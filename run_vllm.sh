#!/bin/bash

# Model and environment settings
export CUDA_VISIBLE_DEVICES=2
MODEL="/data/nvme6n1p1/adal_workspace/small_llm/models/chatting_llm/try_1/checkpoint-566"
TOKENIZER="/data/nvme6n1p1/adal_workspace/small_llm/models/chatting_llm/try_1/checkpoint-566"
PORT=8009
HOST="0.0.0.0"
SEED=0

# vLLM configuration parameters
GPU_MEMORY_UTILIZATION=0.80
MAX_NUM_BATCHED_TOKENS=1024
MAX_MODEL_LEN=1024
DTYPE="auto"
TENSOR_PARALLEL_SIZE=1
BLOCK_SIZE=16
KV_CACHE_DTYPE="auto"
SWAP_SPACE=32
MAX_NUM_SEQS=5
# Construct the vLLM command
CMD="vllm serve $MODEL \
  --host $HOST \
  --port $PORT \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
  --max-model-len $MAX_MODEL_LEN \
  --trust-remote-code \
  --tokenizer $TOKENIZER \
  --seed $SEED \
  --dtype $DTYPE \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --swap-space $SWAP_SPACE \
  --block-size $BLOCK_SIZE \
  --kv-cache-dtype $KV_CACHE_DTYPE \
  --max-num-seqs $MAX_NUM_SEQS \
  --disable-frontend-multiprocessing"

# Execute the command
eval $CMD