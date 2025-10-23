PROTON_CONFIG='{"name_prefix": "bench_pyctx", "backend": "cupti", "context": "python"}'
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
  --random-input-len 512 \
  --random-output-len 128 \
  --ignore-eos \
  --max-concurrency 128 \
  --num-prompt 128 \
  --save-result --result-filename vllm_benchmark_serving_results_dp_ep.json \
  --profile
