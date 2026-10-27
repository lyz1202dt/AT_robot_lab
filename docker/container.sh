#!/usr/bin/env bash

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yaml"
ENV_FILE="${SCRIPT_DIR}/.env.base"
SERVICE_NAME="robot-lab"
CONTAINER_NAME="robot-lab"
TMP_DIR="${SCRIPT_DIR}/.tmp"
X11_OVERRIDE_FILE="${TMP_DIR}/docker-compose.x11.yaml"
GPU_OVERRIDE_FILE="${TMP_DIR}/docker-compose.gpu.yaml"

mkdir -p "${TMP_DIR}"

print_usage() {
    cat <<'EOF'
Usage:
  ./docker/container.sh build
  ./docker/container.sh start
  ./docker/container.sh enter
  ./docker/container.sh stop
  ./docker/container.sh restart
  ./docker/container.sh config

Commands:
  build    Build the docker image only.
  start    Build and start the container in detached mode. If X11 is available, enable GUI forwarding.
  enter    Enter the running container with an interactive bash shell.
  stop     Stop and remove the container.
  restart  Restart the container.
  config   Print the effective docker compose configuration.
EOF
}

has_command() {
    command -v "$1" >/dev/null 2>&1
}

host_has_nvidia_gpu() {
    [[ -c /dev/nvidiactl || -c /dev/nvidia0 || -d /proc/driver/nvidia/gpus ]] || return 1
    has_command nvidia-smi || return 1
    nvidia-smi >/dev/null 2>&1
}

gpu_mode() {
    if [[ "${DISABLE_GPU:-0}" == "1" ]]; then
        echo "disabled"
        return
    fi

    if [[ "${FORCE_GPU:-0}" == "1" ]]; then
        echo "forced"
        return
    fi

    if host_has_nvidia_gpu; then
        echo "enabled"
    else
        echo "disabled"
    fi
}

shell_proxy_configured() {
    [[ -n "${HTTP_PROXY:-}" || -n "${HTTPS_PROXY:-}" || -n "${ALL_PROXY:-}" || \
       -n "${http_proxy:-}" || -n "${https_proxy:-}" || -n "${all_proxy:-}" ]]
}

setup_proxy_env() {
    if [[ "${DISABLE_CLASH_PROXY:-0}" == "1" ]]; then
        echo "[INFO] Clash proxy integration disabled by DISABLE_CLASH_PROXY=1."
        export CLASH_PROXY_SCHEME_ARG=""
        export CLASH_PROXY_BUILD_HOST_ARG=""
        export CLASH_PROXY_RUNTIME_HOST_ARG=""
        export CLASH_PROXY_PORT_ARG=""
        export CLASH_NO_PROXY_ARG=""
        export BUILD_HTTP_PROXY_ARG=""
        export BUILD_HTTPS_PROXY_ARG=""
        export BUILD_ALL_PROXY_ARG=""
        export BUILD_NO_PROXY_ARG=""
        export HTTP_PROXY=""
        export HTTPS_PROXY=""
        export ALL_PROXY=""
        export NO_PROXY=""
        export http_proxy=""
        export https_proxy=""
        export all_proxy=""
        export no_proxy=""
        return
    fi

    if shell_proxy_configured; then
        export HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}"
        export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY}}}"
        export ALL_PROXY="${ALL_PROXY:-${all_proxy:-}}"
        export NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
        export http_proxy="${http_proxy:-${HTTP_PROXY}}"
        export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
        export all_proxy="${all_proxy:-${ALL_PROXY}}"
        export no_proxy="${no_proxy:-${NO_PROXY}}"
        export BUILD_HTTP_PROXY_ARG="${HTTP_PROXY}"
        export BUILD_HTTPS_PROXY_ARG="${HTTPS_PROXY}"
        export BUILD_ALL_PROXY_ARG="${ALL_PROXY}"
        export BUILD_NO_PROXY_ARG="${NO_PROXY}"
        export CLASH_PROXY_SCHEME_ARG=""
        export CLASH_PROXY_BUILD_HOST_ARG=""
        export CLASH_PROXY_RUNTIME_HOST_ARG=""
        export CLASH_PROXY_PORT_ARG=""
        export CLASH_NO_PROXY_ARG=""
        echo "[INFO] Using proxy settings from current shell environment."
        return
    fi

    export CLASH_PROXY_SCHEME_ARG="${CLASH_PROXY_SCHEME_ARG:-${CLASH_PROXY_SCHEME:-http}}"
    export CLASH_PROXY_BUILD_HOST_ARG="${CLASH_PROXY_BUILD_HOST_ARG:-${CLASH_PROXY_BUILD_HOST:-host.docker.internal}}"
    export CLASH_PROXY_RUNTIME_HOST_ARG="${CLASH_PROXY_RUNTIME_HOST_ARG:-${CLASH_PROXY_RUNTIME_HOST:-127.0.0.1}}"
    export CLASH_PROXY_PORT_ARG="${CLASH_PROXY_PORT_ARG:-${CLASH_PROXY_PORT:-7890}}"
    export CLASH_NO_PROXY_ARG="${CLASH_NO_PROXY_ARG:-${CLASH_NO_PROXY:-localhost,127.0.0.1,::1,host.docker.internal}}"
    export BUILD_HTTP_PROXY_ARG=""
    export BUILD_HTTPS_PROXY_ARG=""
    export BUILD_ALL_PROXY_ARG=""
    export BUILD_NO_PROXY_ARG=""

    local runtime_proxy_url="${CLASH_PROXY_SCHEME_ARG}://${CLASH_PROXY_RUNTIME_HOST_ARG}:${CLASH_PROXY_PORT_ARG}"

    export HTTP_PROXY="${HTTP_PROXY:-${runtime_proxy_url}}"
    export HTTPS_PROXY="${HTTPS_PROXY:-${runtime_proxy_url}}"
    export ALL_PROXY="${ALL_PROXY:-${runtime_proxy_url}}"
    export NO_PROXY="${NO_PROXY:-${CLASH_NO_PROXY_ARG}}"
    export http_proxy="${http_proxy:-${HTTP_PROXY}}"
    export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
    export all_proxy="${all_proxy:-${ALL_PROXY}}"
    export no_proxy="${no_proxy:-${NO_PROXY}}"
}

compose_cmd() {
    if docker compose version >/dev/null 2>&1; then
        docker compose "$@"
    elif has_command docker-compose; then
        docker-compose "$@"
    else
        echo "[ERROR] Neither 'docker compose' nor 'docker-compose' is available."
        exit 1
    fi
}

require_docker() {
    if ! has_command docker; then
        echo "[ERROR] Docker is not installed or not in PATH."
        exit 1
    fi
}

container_running() {
    docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | grep -qx 'true'
}

container_exists() {
    docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1
}

wait_for_container_running() {
    local retries="${1:-20}"
    local delay_seconds="${2:-0.25}"
    local attempt

    for ((attempt = 1; attempt <= retries; attempt++)); do
        if container_running; then
            return 0
        fi
        sleep "${delay_seconds}"
    done

    return 1
}

write_x11_override() {
    cat > "${X11_OVERRIDE_FILE}" <<EOF
services:
  ${SERVICE_NAME}:
    environment:
      - DISPLAY=${DISPLAY:-}
      - TERM=${TERM:-xterm-256color}
      - QT_X11_NO_MITSHM=1
EOF

    if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
        cat >> "${X11_OVERRIDE_FILE}" <<EOF
      - XAUTHORITY=${XAUTHORITY}
    volumes:
      - type: bind
        source: /tmp/.X11-unix
        target: /tmp/.X11-unix
      - type: bind
        source: /etc/localtime
        target: /etc/localtime
        read_only: true
      - type: bind
        source: ${XAUTHORITY}
        target: ${XAUTHORITY}
        read_only: true
EOF
    else
        cat >> "${X11_OVERRIDE_FILE}" <<'EOF'
    volumes:
      - type: bind
        source: /tmp/.X11-unix
        target: /tmp/.X11-unix
      - type: bind
        source: /etc/localtime
        target: /etc/localtime
        read_only: true
EOF
    fi
}

write_gpu_override() {
    cat > "${GPU_OVERRIDE_FILE}" <<EOF
services:
  ${SERVICE_NAME}:
    devices:
      - "nvidia.com/gpu=all"
EOF
}

enable_x11_access() {
    if has_command xhost; then
        xhost +SI:localuser:root >/dev/null 2>&1 || xhost +local:root >/dev/null 2>&1 || true
    fi
}

compose_args() {
    local args=("--file" "${COMPOSE_FILE}" "--env-file" "${ENV_FILE}")
    local resolved_gpu_mode

    resolved_gpu_mode="$(gpu_mode)"
    if [[ "${resolved_gpu_mode}" == "forced" ]]; then
        echo "[INFO] GPU support forced on by FORCE_GPU=1." >&2
        write_gpu_override
        args+=("--file" "${GPU_OVERRIDE_FILE}")
    elif [[ "${resolved_gpu_mode}" == "enabled" ]]; then
        echo "[INFO] NVIDIA GPU detected on host. Enabling container GPU access." >&2
        write_gpu_override
        args+=("--file" "${GPU_OVERRIDE_FILE}")
    else
        if [[ "${DISABLE_GPU:-0}" == "1" ]]; then
            echo "[INFO] GPU support disabled by DISABLE_GPU=1." >&2
        else
            echo "[INFO] NVIDIA GPU not detected on host. Starting container without GPU access." >&2
        fi
    fi

    if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
        write_x11_override
        args+=("--file" "${X11_OVERRIDE_FILE}")
    fi
    printf '%s\n' "${args[@]}"
}

run_compose() {
    mapfile -t args < <(compose_args)
    compose_cmd "${args[@]}" "$@"
}

cmd_build() {
    require_docker
    setup_proxy_env
    echo "[INFO] Building ${SERVICE_NAME} image..."
    if [[ "${DISABLE_CLASH_PROXY:-0}" == "1" ]]; then
        echo "[INFO] Build proxy: disabled"
    elif shell_proxy_configured; then
        echo "[INFO] Build proxy: ${HTTP_PROXY:-${HTTPS_PROXY:-${ALL_PROXY:-configured in shell}}}"
    else
        echo "[INFO] Build proxy: ${CLASH_PROXY_SCHEME_ARG}://${CLASH_PROXY_BUILD_HOST_ARG}:${CLASH_PROXY_PORT_ARG}"
    fi
    # Compose's Bake integration does not forward host-network permission.
    # Keep Compose's resolved build settings and explicitly grant it to Bake.
    run_compose build --print "${SERVICE_NAME}" | \
        docker buildx bake --file - --allow=network.host --load "${SERVICE_NAME}"
}

cmd_start() {
    require_docker
    setup_proxy_env
    if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
        echo "[INFO] X11 detected. Enabling GUI forwarding for the container."
        enable_x11_access
    else
        echo "[INFO] X11 not detected. Container will start without GUI forwarding."
    fi
    if [[ "${DISABLE_CLASH_PROXY:-0}" == "1" ]]; then
        echo "[INFO] Runtime proxy: disabled"
    elif shell_proxy_configured; then
        echo "[INFO] Runtime proxy: ${HTTP_PROXY:-${HTTPS_PROXY:-${ALL_PROXY:-configured in shell}}}"
    else
        echo "[INFO] Runtime proxy: ${HTTP_PROXY}"
    fi
    if ! docker image inspect "${SERVICE_NAME}" >/dev/null 2>&1; then
        cmd_build
    fi
    run_compose up -d --no-build --pull never "${SERVICE_NAME}"
    if wait_for_container_running; then
        echo "[INFO] Container '${CONTAINER_NAME}' is ready."
    else
        echo "[ERROR] Container '${CONTAINER_NAME}' was created but is not running yet."
        echo "[INFO] Check its status with: docker ps -a | grep '${CONTAINER_NAME}'"
        exit 1
    fi
}

cmd_enter() {
    require_docker
    if ! container_running; then
        if container_exists; then
            echo "[ERROR] Container '${CONTAINER_NAME}' exists but is not running. Start it with './docker/container.sh start'."
        else
            echo "[ERROR] Container '${CONTAINER_NAME}' does not exist. Start it with './docker/container.sh start'."
        fi
        exit 1
    fi

    local exec_args=("-it")
    if [[ -n "${DISPLAY:-}" ]]; then
        exec_args+=("-e" "DISPLAY=${DISPLAY}")
    fi
    if [[ -n "${TERM:-}" ]]; then
        exec_args+=("-e" "TERM=${TERM}")
    fi
    if [[ -n "${XAUTHORITY:-}" ]]; then
        exec_args+=("-e" "XAUTHORITY=${XAUTHORITY}")
    fi

    if [[ "${DISABLE_CLASH_PROXY:-0}" == "1" ]]; then
        docker exec "${exec_args[@]}" "${CONTAINER_NAME}" env \
            -u HTTP_PROXY \
            -u HTTPS_PROXY \
            -u ALL_PROXY \
            -u NO_PROXY \
            -u http_proxy \
            -u https_proxy \
            -u all_proxy \
            -u no_proxy \
            /bin/bash --noprofile --norc
    else
        docker exec "${exec_args[@]}" "${CONTAINER_NAME}" /bin/bash
    fi
}

cmd_stop() {
    require_docker
    run_compose down
    echo "[INFO] Container '${CONTAINER_NAME}' has been stopped and removed."
}

cmd_restart() {
    cmd_stop
    cmd_start
}

cmd_config() {
    require_docker
    setup_proxy_env
    run_compose config
}

main() {
    local command="${1:-}"
    case "${command}" in
        build)
            cmd_build
            ;;
        start)
            cmd_start
            ;;
        enter)
            cmd_enter
            ;;
        stop)
            cmd_stop
            ;;
        restart)
            cmd_restart
            ;;
        config)
            cmd_config
            ;;
        -h|--help|help|"")
            print_usage
            ;;
        *)
            echo "[ERROR] Unknown command: ${command}"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
