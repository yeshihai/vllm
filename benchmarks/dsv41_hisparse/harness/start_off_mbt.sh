#!/bin/bash
# OFF (no HiSparse) launcher; MBT is the scheduler token budget.
MBT="${1:?usage: start_off_mbt.sh MBT}"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH

source /etc/taiji_sering_env/taiji_sering_env
export NODE_IP_LIST=${NODE_IP_LIST:-28.48.51.38}
export LOCAL_IP=${LOCAL_IP:-28.48.51.38}
export CHIEF_IP=${CHIEF_IP:-28.48.51.38}
export LOCAL_HOSTNAME=${LOCAL_HOSTNAME:-taiji-serving-custom-6095215-0}
export SKIP_EXPORT_IP_LIST=1
export ROOT=/dockerdata/yeshihai_dsv41_buffer
export PYTHONPATH="$ROOT/src/vllm-pr"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/vllm_dev
export PATH="/root/miniconda3/envs/vllm_dev/bin:$PATH"
export VLLM_USE_V2_MODEL_RUNNER=1
export LD_PRELOAD="/root/miniconda3/envs/vllm_dev/lib/libstdc++.so.6"
export VLLM_KV_CACHE_LAYOUT=BLHNC
export VLLM_NO_ENABLE_PREFIX_CACHING=0
export VLLM_NO_ENABLE_LOG_REQUEST=1
export VLLM_MAX_NUM_BATCHED_TOKENS="$MBT"
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export MOE_BACKEND=marlin
export MODEL_NAME="DeepSeek-V4.1-Flash-1017-H20-Perf-noxzhao DeepSeek-V4-Flash-H20-vllm-0702-profile-noxzhao"
export MODEL_PATH="/dockerdata/models/DeepSeek-V4.1-Flash/"
unset http_proxy https_proxy
exec /root/miniconda3/envs/vllm_dev/bin/vllm serve "$MODEL_PATH" \
  --served-model-name $MODEL_NAME \
  --host 0.0.0.0 --port 8021 --trust-remote-code \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --tokenizer-mode deepseek_v41 \
  --max-model-len 1048576 --max-num-seqs 32 --max-num-batched-tokens "$MBT" \
  --gpu-memory-utilization 0.92 --kv-cache-dtype fp8_ds_mla --load-format auto \
  --scheduler-reserve-full-isl --enable-prefix-caching \
  --kernel-config '{"linear_backend":"marlin","moe_backend":"marlin","enable_jit_warmup":false}'
