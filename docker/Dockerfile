ARG ISAACLAB_BASE_IMAGE_ARG=isaac-lab-base

FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

SHELL ["/bin/bash", "-c"]

ARG DOCKER_TWIST2_LAB_PATH_ARG
ENV DOCKER_TWIST2_LAB_PATH=${DOCKER_TWIST2_LAB_PATH_ARG}

USER root

# Copy the repository into the image so the editable install is available
# even before the bind mount overlays the workspace at runtime.
COPY . ${DOCKER_TWIST2_LAB_PATH}

RUN bash -lc "source ${HOME}/.bashrc && \
    cd ${DOCKER_TWIST2_LAB_PATH} && \
    python -m pip install -e source/twist2_lab"

WORKDIR ${DOCKER_TWIST2_LAB_PATH}
