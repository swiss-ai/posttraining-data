#!/usr/bin/env bash

set -euo pipefail

image_tag="v0.5.18-rocm720-mi30x"
image_directory="${BEVERIN_IMAGE_DIR:-/capstor/scratch/cscs/${USER}/ce-images}"
image_path="${image_directory}/sglang-${image_tag}.sqfs"

mkdir -p "${image_directory}"
if [[ -f "${image_path}" ]]; then
    echo "Container already exists: ${image_path}"
    exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    allocation=(
        srun
        --partition=mi300
        --nodes=1
        --ntasks=1
        --gpus-per-task=1
        --cpus-per-task=4
        --time=01:00:00
    )
    if [[ -n "${BEVERIN_ACCOUNT:-}" ]]; then
        allocation+=(--account="${BEVERIN_ACCOUNT}")
    fi
    echo "Starting a Beverin MI300 allocation for temporary Podman storage."
    exec "${allocation[@]}" bash "${script_path}"
fi

# Vadret home storage does not support the overlay extended attributes Podman
# needs. CSCS recommends node-local /dev/shm for rootless container storage.
podman_work="/dev/shm/${USER}/apertus-podman-${SLURM_JOB_ID}"
storage_config="${podman_work}/storage.conf"
mkdir -p \
    "${podman_work}/runroot" \
    "${podman_work}/root" \
    "${podman_work}/tmp" \
    "${podman_work}/enroot-cache" \
    "${podman_work}/enroot-runtime" \
    "${podman_work}/enroot-data"
cat >"${storage_config}" <<EOF
[storage]
driver = "overlay"
runroot = "${podman_work}/runroot"
graphroot = "${podman_work}/root"
EOF
export CONTAINERS_STORAGE_CONF="${storage_config}"
export TMPDIR="${podman_work}/tmp"
export ENROOT_CACHE_PATH="${podman_work}/enroot-cache"
export ENROOT_RUNTIME_PATH="${podman_work}/enroot-runtime"
export ENROOT_DATA_PATH="${podman_work}/enroot-data"

echo "Pulling lmsysorg/sglang:${image_tag}"
podman pull "docker://lmsysorg/sglang:${image_tag}"
enroot import \
    -x mount \
    -o "${image_path}" \
    "podman://lmsysorg/sglang:${image_tag}"

echo "Created ${image_path}"
