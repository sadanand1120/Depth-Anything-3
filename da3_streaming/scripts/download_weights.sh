#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_DIR="${SCRIPT_DIR}/../weights"
mkdir -p "${WEIGHTS_DIR}"

download_model() {
    local hf_repo="$1"
    local local_dir="$2"
    local target_dir="${WEIGHTS_DIR}/${local_dir}"

    mkdir -p "${target_dir}"
    echo "Downloading ${hf_repo} -> ${target_dir}"
    curl -L "https://huggingface.co/${hf_repo}/resolve/main/config.json" -o "${target_dir}/config.json"
    curl -L "https://huggingface.co/${hf_repo}/resolve/main/model.safetensors" -o "${target_dir}/model.safetensors"
}

# SALAD (~ 340 MiB)
echo "Downloading SALAD weights -> ${WEIGHTS_DIR}/dino_salad.ckpt"
curl -L "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt" -o "${WEIGHTS_DIR}/dino_salad.ckpt"

download_model "depth-anything/DA3NESTED-GIANT-LARGE-1.1" "da3nestedgiantlarge1.1"
download_model "depth-anything/DA3-GIANT-1.1" "da3giant1.1"
download_model "depth-anything/DA3METRIC-LARGE" "da3metriclarge"
download_model "depth-anything/DA3-LARGE-1.1" "da3large1.1"
download_model "depth-anything/DA3-BASE" "da3base"
download_model "depth-anything/DA3-SMALL" "da3small"
