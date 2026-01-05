#!/bin/bash

export VLLM_RPC_TIMEOUT=30000
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LMCACHE_CONFIG_FILE=/home/henry/LMCache/benchmarks/multi-round-qa/example.yaml

CONDA_BASE="/data01/henry/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate llm

lmcache_vllm serve $1 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --port 8000 \

echo "LMCache server killed"