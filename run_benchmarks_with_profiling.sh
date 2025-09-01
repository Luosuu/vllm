#!/bin/bash

# Script to run benchmark_serving.py with auto profiling enabled
# 
# IMPORTANT: You need to start the vLLM server with auto profiling environment variables!
# 
# Example server startup:
# VLLM_LLMPROF_ENABLE_AUTO_PROFILE=true \
# VLLM_LLMPROF_PROFILE_REQUESTS_INTERVAL=50 \
# VLLM_LLMPROF_PROFILE_DIR="benchmark_profiles" \
# vllm serve meta-llama/Llama-3.1-8B-Instruct --swap-space 16 --disable-log-requests

# Benchmark configuration - modify these as needed
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME="sonnet"
DATASET_PATH="benchmarks/sonnet.txt"
NUM_PROMPTS=10000
REQUEST_RATE=500
HOST="127.0.0.1"
PORT=8039
BACKEND="vllm"
PROFILE_DIR="benchmark_profiles/rps_50"

echo "📊 Benchmark configuration:"
echo "   MODEL=$MODEL"
echo "   DATASET_NAME=$DATASET_NAME"
echo "   NUM_PROMPTS=$NUM_PROMPTS"
echo "   REQUEST_RATE=$REQUEST_RATE"
echo "   HOST=$HOST"
echo "   PORT=$PORT"
echo
echo "⚠️  IMPORTANT: Make sure vLLM server is running with these environment variables:"
echo "   VLLM_LLMPROF_ENABLE_AUTO_PROFILE=true"
echo "   VLLM_LLMPROF_PROFILE_REQUESTS_INTERVAL=50"
echo "   VLLM_LLMPROF_PROFILE_DIR=\"$PROFILE_DIR\""
echo
echo "💡 Server startup command:"
echo "   VLLM_LLMPROF_ENABLE_AUTO_PROFILE=true \\"
echo "   VLLM_LLMPROF_PROFILE_REQUESTS_INTERVAL=50 \\"
echo "   VLLM_LLMPROF_PROFILE_DIR=\"$PROFILE_DIR\" \\"
echo "   vllm serve $MODEL --swap-space 16 --disable-log-requests"
echo

# Run benchmark_serving.py with configured arguments
echo "📊 Running benchmark_serving.py..."
python benchmarks/benchmark_serving.py \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE"

echo
echo "✅ Benchmark completed! Check $PROFILE_DIR for profile files."
echo "💡 View profiles with: proton-viewer -m time/s <profile_file.hatchet>"

### Result on A100 with CUDA Graph Rps 50

# ============ Serving Benchmark Result ============
# Successful requests:                     3806      
# Request rate configured (RPS):           50.00     
# Benchmark duration (s):                  252.49    
# Total input tokens:                      1937770   
# Total generated tokens:                  570900    
# Request throughput (req/s):              15.07     
# Output token throughput (tok/s):         2261.06   
# Total Token throughput (tok/s):          9935.62   
# ---------------Time to First Token----------------
# Mean TTFT (ms):                          40847.65  
# Median TTFT (ms):                        49370.18  
# P99 TTFT (ms):                           49819.53  
# -----Time per Output Token (excl. 1st token)------
# Mean TPOT (ms):                          108.41    
# Median TPOT (ms):                        111.15    
# P99 TPOT (ms):                           112.43    
# ---------------Inter-token Latency----------------
# Mean ITL (ms):                           108.41    
# Median ITL (ms):                         72.16     
# P99 ITL (ms):                            291.45    
# ==================================================

# Egaer Mode: Rps 500
# ============ Serving Benchmark Result ============
# Successful requests:                     1093      
# Request rate configured (RPS):           500.00    
# Benchmark duration (s):                  77.61     
# Total input tokens:                      556334    
# Total generated tokens:                  163950    
# Request throughput (req/s):              14.08     
# Output token throughput (tok/s):         2112.43   
# Total Token throughput (tok/s):          9280.58   
# ---------------Time to First Token----------------
# Mean TTFT (ms):                          30901.71  
# Median TTFT (ms):                        33787.65  
# P99 TTFT (ms):                           59580.99  
# -----Time per Output Token (excl. 1st token)------
# Mean TPOT (ms):                          107.24    
# Median TPOT (ms):                        116.04    
# P99 TPOT (ms):                           119.40    
# ---------------Inter-token Latency----------------
# Mean ITL (ms):                           107.24    
# Median ITL (ms):                         62.22     
# P99 ITL (ms):                            213.28    
# ==================================================