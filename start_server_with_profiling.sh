#!/bin/bash

# Script to start vLLM server with auto profiling enabled
# Edit the variables below to customize your setup

# Auto profiling configuration
export VLLM_LLMPROF_ENABLE_AUTO_PROFILE=true
export VLLM_LLMPROF_PROFILE_REQUESTS_INTERVAL=500
export VLLM_LLMPROF_PROFILE_DIR="benchmark_profiles/rps_500_cpu_timed_eager_kvcache_annotated"
export VLLM_LLMPROF_DEBUG=true

# Force V0 engine (our profiling code is in V0 engine)
export VLLM_USE_V1=0

# Server configuration
MODEL="meta-llama/Llama-3.1-8B-Instruct"
HOST="localhost"
PORT=8039

echo "🔧 Starting vLLM server with auto profiling enabled..."
echo "   VLLM_LLMPROF_ENABLE_AUTO_PROFILE=$VLLM_LLMPROF_ENABLE_AUTO_PROFILE"
echo "   VLLM_LLMPROF_PROFILE_REQUESTS_INTERVAL=$VLLM_LLMPROF_PROFILE_REQUESTS_INTERVAL"
echo "   VLLM_LLMPROF_PROFILE_DIR=$VLLM_LLMPROF_PROFILE_DIR"
echo "   VLLM_LLMPROF_DEBUG=$VLLM_LLMPROF_DEBUG"
echo "   VLLM_USE_V1=$VLLM_USE_V1"
echo
echo "🚀 Server configuration:"
echo "   MODEL=$MODEL"
echo "   HOST=$HOST"
echo "   PORT=$PORT"
echo

# Start the server
echo "📡 Starting vLLM server..."
vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --disable-log-requests \
    --max-model-len 64000 \
    --dtype bfloat16 \
    --trust-remote-code \
    --enforce-eager