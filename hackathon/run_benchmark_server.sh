# Single node EP deployment with pplx backend
(   
    set -x
    . .venv/bin/activate
    export NCCL_DEBUG=WARN
    # export VLLM_TORCH_PROFILER_DIR=./vllm_profile_torch
    export model_dir=${model_dir:-/mnt/local/localcache00/gpt-oss-120b}
    export YAML_CONFIG="GPT-OSS_Blackwell.yaml"
    export VLLM_ALL2ALL_BACKEND=deepep_high_throughput
    # export CUDA_LAUNCH_BLOCKING=1
    vllm serve $model_dir \
        --config ${YAML_CONFIG} \
        --tensor-parallel-size 1 \
        --max-num-seqs 512 \
        --data-parallel-size 8 \
        --enable-expert-parallel
)
