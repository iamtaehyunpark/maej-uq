#!/usr/bin/env bash
# vLLM serving only, from the yllm env.
set -u
SNAP=/data5/user/hf_cache/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
GPU=${1:-0}; PORT=${2:-8000}
LOG=/data5/kje/MULTIAGENT/maej-uq/runs/vllm_${PORT}.log
mkdir -p "$(dirname "$LOG")"
curl -s -m 5 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "already up on $PORT"; exit 0; }
CUDA_VISIBLE_DEVICES=$GPU setsid nohup /opt/anaconda3/envs/yllm/bin/vllm serve "$SNAP" \
  --served-model-name Qwen/Qwen3.6-35B-A3B \
  --max-model-len 40960 --max-num-seqs 8 \
  --gpu-memory-utilization 0.93 --enable-prefix-caching \
  --port "$PORT" > "$LOG" 2>&1 &
for i in $(seq 1 160); do
  sleep 15
  curl -s -m 5 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "up after $((i*15))s"; exit 0; }
  pgrep -f "[v]llm serve.*--port $PORT" >/dev/null || { echo "DIED"; tail -25 "$LOG"; exit 1; }
done
echo "TIMEOUT"; exit 1
