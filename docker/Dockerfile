ARG ISAACLAB_BASE_IMAGE_ARG
ARG BUILD_HTTP_PROXY_ARG
ARG BUILD_HTTPS_PROXY_ARG
ARG BUILD_ALL_PROXY_ARG
ARG BUILD_NO_PROXY_ARG
ARG CLASH_PROXY_SCHEME_ARG=http
ARG CLASH_PROXY_BUILD_HOST_ARG=127.0.0.1
ARG CLASH_PROXY_RUNTIME_HOST_ARG=127.0.0.1
ARG CLASH_PROXY_PORT_ARG=7890
ARG CLASH_NO_PROXY_ARG=localhost,127.0.0.1,::1

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ARG BUILD_HTTP_PROXY_ARG
ARG BUILD_HTTPS_PROXY_ARG
ARG BUILD_ALL_PROXY_ARG
ARG BUILD_NO_PROXY_ARG
ARG CLASH_PROXY_SCHEME_ARG
ARG CLASH_PROXY_BUILD_HOST_ARG
ARG CLASH_PROXY_RUNTIME_HOST_ARG
ARG CLASH_PROXY_PORT_ARG
ARG CLASH_NO_PROXY_ARG

ENV DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH=${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}
ENV BUILD_HTTP_PROXY_ARG=${BUILD_HTTP_PROXY_ARG}
ENV BUILD_HTTPS_PROXY_ARG=${BUILD_HTTPS_PROXY_ARG}
ENV BUILD_ALL_PROXY_ARG=${BUILD_ALL_PROXY_ARG}
ENV BUILD_NO_PROXY_ARG=${BUILD_NO_PROXY_ARG}
ENV CLASH_PROXY_SCHEME_ARG=${CLASH_PROXY_SCHEME_ARG}
ENV CLASH_PROXY_BUILD_HOST_ARG=${CLASH_PROXY_BUILD_HOST_ARG}
ENV CLASH_PROXY_RUNTIME_HOST_ARG=${CLASH_PROXY_RUNTIME_HOST_ARG}
ENV CLASH_PROXY_PORT_ARG=${CLASH_PROXY_PORT_ARG}
ENV CLASH_NO_PROXY_ARG=${CLASH_NO_PROXY_ARG}

USER root

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY ./ ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}

# Ensure the Isaac Lab core packages are installed into the Python environment used by Isaac Sim.
# Some custom/base images only provide aliases to Isaac Sim's python and do not pre-install the
# editable Isaac Lab packages, which causes `ModuleNotFoundError: No module named 'isaaclab'`.
# flatdict 4.0.1 uses pkg_resources while building, which setuptools 82 removed.
# Preinstall it without build isolation so Isaac Lab's dependency install can reuse it.
RUN bash -i -c 'source "${HOME}/.bashrc" && \
    if [[ -n "${BUILD_HTTP_PROXY_ARG}" || -n "${BUILD_HTTPS_PROXY_ARG}" || -n "${BUILD_ALL_PROXY_ARG}" ]]; then \
        export HTTP_PROXY="${BUILD_HTTP_PROXY_ARG}" && \
        export HTTPS_PROXY="${BUILD_HTTPS_PROXY_ARG:-${BUILD_HTTP_PROXY_ARG}}" && \
        export ALL_PROXY="${BUILD_ALL_PROXY_ARG}" && \
        export http_proxy="${HTTP_PROXY}" && \
        export https_proxy="${HTTPS_PROXY}" && \
        export all_proxy="${ALL_PROXY}" && \
        export NO_PROXY="${BUILD_NO_PROXY_ARG}" && \
        export no_proxy="${BUILD_NO_PROXY_ARG}"; \
    elif [[ -n "${CLASH_PROXY_SCHEME_ARG}" && -n "${CLASH_PROXY_BUILD_HOST_ARG}" && -n "${CLASH_PROXY_PORT_ARG}" ]]; then \
        export CLASH_PROXY_BUILD_URL="${CLASH_PROXY_SCHEME_ARG}://${CLASH_PROXY_BUILD_HOST_ARG}:${CLASH_PROXY_PORT_ARG}" && \
        export HTTP_PROXY="${CLASH_PROXY_BUILD_URL}" && \
        export HTTPS_PROXY="${CLASH_PROXY_BUILD_URL}" && \
        export ALL_PROXY="${CLASH_PROXY_BUILD_URL}" && \
        export http_proxy="${CLASH_PROXY_BUILD_URL}" && \
        export https_proxy="${CLASH_PROXY_BUILD_URL}" && \
        export all_proxy="${CLASH_PROXY_BUILD_URL}" && \
        export NO_PROXY="${CLASH_NO_PROXY_ARG}" && \
        export no_proxy="${CLASH_NO_PROXY_ARG}"; \
    else \
        unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; \
    fi && \
    export ISAACLAB_PATH=${ISAACLAB_PATH:-/workspace/isaaclab} && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install "setuptools<82" && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-build-isolation "flatdict==4.0.1" && \
    ${ISAACLAB_PATH}/isaaclab.sh --install && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -c "import flatdict; import isaaclab"'

# Install the robot_lab extension into the same Python environment.
# Pin TensorDict after all dependency installs for compatibility with PyTorch 2.7.
# Verify the training runner imports during the build.
RUN bash -i -c 'source "${HOME}/.bashrc" && \
    if [[ -n "${BUILD_HTTP_PROXY_ARG}" || -n "${BUILD_HTTPS_PROXY_ARG}" || -n "${BUILD_ALL_PROXY_ARG}" ]]; then \
        export HTTP_PROXY="${BUILD_HTTP_PROXY_ARG}" && \
        export HTTPS_PROXY="${BUILD_HTTPS_PROXY_ARG:-${BUILD_HTTP_PROXY_ARG}}" && \
        export ALL_PROXY="${BUILD_ALL_PROXY_ARG}" && \
        export http_proxy="${HTTP_PROXY}" && \
        export https_proxy="${HTTPS_PROXY}" && \
        export all_proxy="${ALL_PROXY}" && \
        export NO_PROXY="${BUILD_NO_PROXY_ARG}" && \
        export no_proxy="${BUILD_NO_PROXY_ARG}"; \
    elif [[ -n "${CLASH_PROXY_SCHEME_ARG}" && -n "${CLASH_PROXY_BUILD_HOST_ARG}" && -n "${CLASH_PROXY_PORT_ARG}" ]]; then \
        export CLASH_PROXY_BUILD_URL="${CLASH_PROXY_SCHEME_ARG}://${CLASH_PROXY_BUILD_HOST_ARG}:${CLASH_PROXY_PORT_ARG}" && \
        export HTTP_PROXY="${CLASH_PROXY_BUILD_URL}" && \
        export HTTPS_PROXY="${CLASH_PROXY_BUILD_URL}" && \
        export ALL_PROXY="${CLASH_PROXY_BUILD_URL}" && \
        export http_proxy="${CLASH_PROXY_BUILD_URL}" && \
        export https_proxy="${CLASH_PROXY_BUILD_URL}" && \
        export all_proxy="${CLASH_PROXY_BUILD_URL}" && \
        export NO_PROXY="${CLASH_NO_PROXY_ARG}" && \
        export no_proxy="${CLASH_NO_PROXY_ARG}"; \
    else \
        unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; \
    fi && \
    export ISAACLAB_PATH=${ISAACLAB_PATH:-/workspace/isaaclab} && \
    cd ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/robot_lab && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e . && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-deps "tensordict==0.12.1" && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -c "import tensordict; from rsl_rl.runners import DistillationRunner, OnPolicyRunner"'

# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
WORKDIR /workspace
