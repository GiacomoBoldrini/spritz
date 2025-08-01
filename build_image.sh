#!/bin/bash
set -euo pipefail

# ---------------------------
# User-configurable variables
# ---------------------------

# CERN username (auto-detect from $USER)
USER_NAME="${USER}"

# Base output directory (default: your EOS space)
BASE_DIR="/eos/user/${USER_NAME:0:1}/${USER_NAME}"

# Docker image name
IMAGE_NAME="spritz-env"

# Output paths
SIF_PATH="${BASE_DIR}/${IMAGE_NAME}.sif"

# Optional: Apptainer cache dir (using EOS by default)
export APPTAINER_CACHEDIR="${BASE_DIR}/apptainer-cache"

mkdir -p "$APPTAINER_CACHEDIR"

echo "==> Building Docker image: ${IMAGE_NAME}"
docker build --progress=plain --no-cache -t "${IMAGE_NAME}" .

echo "==> Building Singularity image: ${SIF_PATH}"
# Stream Docker image directly into Singularity (no intermediate tar)
docker save "${IMAGE_NAME}:latest" | singularity build "${SIF_PATH}" docker-archive:/dev/stdin

echo "==> Build complete: ${SIF_PATH}"
