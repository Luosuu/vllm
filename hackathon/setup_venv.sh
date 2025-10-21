#!/usr/bin/env bash

# . hackathon/setup_venv.sh
# set -euo pipefail # exit at any failure

(
    set -x  # enable command tracing inside subshell
    # build vllm from source using torch2.9 
    # create .venv first by `uv venv`
    # uv pip install torch==2.9 
    # uv pip install torchaudio
    uv pip install torchvision 
    . .venv/bin/activate
    python use_existing_torch.py

    uv pip install -r requirements/common.txt 
    uv pip install -r requirements/cuda.txt 
    uv pip install -r requirements/build.txt 
    uv pip install -vvv  -e .  --no-build-isolation 

    # Installed triton_kernels like,
    git clone https://github.com/triton-lang/triton.git 
    cd triton
    git checkout release/3.5.x
    cd ../
    uv pip install triton/python/triton_kernels --no-deps 

    # install deep_ep
    cd tools/ep_kernels/
    TORCH_CUDA_ARCH_LIST=10.0 PIP_CMD="uv pip" bash install_python_libraries.sh
    cd -
)