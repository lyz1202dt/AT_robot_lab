ARG ISAACLAB_BASE_IMAGE_ARG
ARG CLASH_PROXY_SCHEME_ARG=http
ARG CLASH_PROXY_BUILD_HOST_ARG=127.0.0.1
ARG CLASH_PROXY_RUNTIME_HOST_ARG=127.0.0.1
ARG CLASH_PROXY_PORT_ARG=7890
ARG CLASH_NO_PROXY_ARG=localhost,127.0.0.1,::1

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ARG CLASH_PROXY_SCHEME_ARG
ARG CLASH_PROXY_BUILD_HOST_ARG
ARG CLASH_PROXY_RUNTIME_HOST_ARG
ARG CLASH_PROXY_PORT_ARG
ARG CLASH_NO_PROXY_ARG

ENV DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH=${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}
ENV CLASH_PROXY_BUILD_URL=${CLASH_PROXY_SCHEME_ARG}://${CLASH_PROXY_BUILD_HOST_ARG}:${CLASH_PROXY_PORT_ARG}
ENV CLASH_PROXY_RUNTIME_URL=${CLASH_PROXY_SCHEME_ARG}://${CLASH_PROXY_RUNTIME_HOST_ARG}:${CLASH_PROXY_PORT_ARG}
ENV HTTP_PROXY=${CLASH_PROXY_RUNTIME_URL}
ENV HTTPS_PROXY=${CLASH_PROXY_RUNTIME_URL}
ENV ALL_PROXY=${CLASH_PROXY_RUNTIME_URL}
ENV http_proxy=${CLASH_PROXY_RUNTIME_URL}
ENV https_proxy=${CLASH_PROXY_RUNTIME_URL}
ENV all_proxy=${CLASH_PROXY_RUNTIME_URL}
ENV NO_PROXY=${CLASH_NO_PROXY_ARG}
ENV no_proxy=${CLASH_NO_PROXY_ARG}

USER root

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY ../ ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}

# Ensure the Isaac Lab core packages are installed into the Python environment used by Isaac Sim.
# Some custom/base images only provide aliases to Isaac Sim's python and do not pre-install the
# editable Isaac Lab packages, which causes `ModuleNotFoundError: No module named 'isaaclab'`.
RUN HTTP_PROXY=${CLASH_PROXY_BUILD_URL} \
    HTTPS_PROXY=${CLASH_PROXY_BUILD_URL} \
    ALL_PROXY=${CLASH_PROXY_BUILD_URL} \
    http_proxy=${CLASH_PROXY_BUILD_URL} \
    https_proxy=${CLASH_PROXY_BUILD_URL} \
    all_proxy=${CLASH_PROXY_BUILD_URL} \
    NO_PROXY=${CLASH_NO_PROXY_ARG} \
    no_proxy=${CLASH_NO_PROXY_ARG} \
    bash -i -c "source ${HOME}/.bashrc && \
    export ISAACLAB_PATH=${ISAACLAB_PATH:-/workspace/isaaclab} && \
    ${ISAACLAB_PATH}/isaaclab.sh --install && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir flatdict"

# Install the robot_lab extension into the same Python environment.
RUN HTTP_PROXY=${CLASH_PROXY_BUILD_URL} \
    HTTPS_PROXY=${CLASH_PROXY_BUILD_URL} \
    ALL_PROXY=${CLASH_PROXY_BUILD_URL} \
    http_proxy=${CLASH_PROXY_BUILD_URL} \
    https_proxy=${CLASH_PROXY_BUILD_URL} \
    all_proxy=${CLASH_PROXY_BUILD_URL} \
    NO_PROXY=${CLASH_NO_PROXY_ARG} \
    no_proxy=${CLASH_NO_PROXY_ARG} \
    bash -i -c "source ${HOME}/.bashrc && \
    export ISAACLAB_PATH=${ISAACLAB_PATH:-/workspace/isaaclab} && \
    cd ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/robot_lab && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e ."

# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
WORKDIR /workspace
