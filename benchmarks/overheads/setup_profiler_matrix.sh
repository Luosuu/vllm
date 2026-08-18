#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${VENV:-"$REPO_ROOT/.venv-profiler-overhead"}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
VLLM_BUILD_MODE=${VLLM_BUILD_MODE:-source}
FLASHINFER_CUBIN_VERSION=${FLASHINFER_CUBIN_VERSION:-$(
  sed -n 's/^flashinfer-cubin==//p' "$REPO_ROOT/requirements/cuda.txt"
)}

command -v uv >/dev/null || {
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
}
command -v nsys >/dev/null || command -v rocprofv3 >/dev/null || {
  echo "a system profiler is required: install nsight-systems-cli (NVIDIA)" \
    "or rocprofiler-sdk for rocprofv3 (AMD)." >&2
  exit 1
}

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python "$PYTHON_VERSION" "$VENV"
fi
uv pip install --python "$VENV/bin/python" matplotlib huggingface_hub ninja
case "$VLLM_BUILD_MODE" in
  source)
    uv pip install \
      --python "$VENV/bin/python" \
      --torch-backend=auto \
      -e "$REPO_ROOT"
    ;;
  precompiled)
    VLLM_USE_PRECOMPILED=1 uv pip install \
      --python "$VENV/bin/python" \
      --torch-backend=auto \
      -e "$REPO_ROOT"
    ;;
  *)
    echo "VLLM_BUILD_MODE must be source or precompiled" >&2
    exit 1
    ;;
esac
uv pip install \
  --python "$VENV/bin/python" \
  --extra-index-url https://flashinfer.ai/whl/ \
  "flashinfer-cubin==$FLASHINFER_CUBIN_VERSION"

env -u PYTHONPATH "$VENV/bin/python" -c '
import torch
import triton
import triton.profiler
import vllm
from vllm.platforms import current_platform

required = ("start", "activate", "deactivate", "finalize")
missing = [name for name in required if not hasattr(triton.profiler, name)]
if missing:
    raise RuntimeError(f"Triton Proton API is missing: {missing}")
print(
    f"vLLM {vllm.__version__}; PyTorch {torch.__version__}; "
    f"Triton {triton.__version__}; {current_platform}"
)
'
if command -v nsys >/dev/null; then
  nsys --version
fi
if command -v rocprofv3 >/dev/null; then
  rocprofv3 --version
fi

"$VENV/bin/python" -c '
from huggingface_hub import snapshot_download

models = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mixtral-8x7B-v0.1",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3.5-27B",
)
for model in models:
    print(f"Caching configuration for {model}", flush=True)
    snapshot_download(
        repo_id=model,
        allow_patterns=(
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
            "vocab.txt",
        ),
    )
'

echo "Environment ready: $VENV"
echo "Run: $VENV/bin/python benchmarks/overheads/run_profiler_matrix.py"

if [[ ${1:-} == "--run" ]]; then
  shift
  read -r -a graph_modes <<<"${GRAPH_MODES:-cudagraph eager}"
  for graph_mode in "${graph_modes[@]}"; do
    "$VENV/bin/python" "$REPO_ROOT/benchmarks/overheads/run_profiler_matrix.py" \
      "$@" \
      --graph-mode "$graph_mode"
  done
fi
