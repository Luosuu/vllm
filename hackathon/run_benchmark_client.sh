PROTON_CONFIG='{"name_prefix": "120b_dpep8_triton_kernels_deepep", "backend": "cupti", "context": "shadow"}'
. .venv/bin/activate
model_dir=${model_dir:-/mnt/local/localcache00/gpt-oss-120b}
echo $model_dir
vllm bench serve \
  --extra-body "$PROTON_CONFIG" \
  --host 0.0.0.0 \
  --port 8000 \
  --model $model_dir \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --ignore-eos \
  --num-prompt 512 \
  --save-result --result-filename vllm_benchmark_serving_results_120b_dpep8_triton_kernels_deepep.json \
  --profile
