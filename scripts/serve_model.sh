#!/usr/bin/env bash
# serve_model.sh <snapshot-dir> <served-name> <gpus> <port> [tp]
set -u
SNAP=$1; NAME=$2; GPUS=$3; PORT=$4; TP=${5:-1}
LOG=/data5/kje/MULTIAGENT/maej-uq/runs/vllm_${PORT}.log
mkdir -p "$(dirname "$LOG")"
curl -s -m 5 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "already up on $PORT"; exit 0; }
# NCCL segfaults inside ncclNetPluginInit on this host while probing for a
# network plugin. Forcing the socket transport skips that probe. Verified with a
# two-GPU all-reduce on 2,3: baseline and NCCL_NET_PLUGIN=none both crash,
# NCCL_NET=Socket succeeds. Single-GPU serving never reaches NCCL, which is why
# this only ever showed up on tensor-parallel launches.
NCCL_NET=Socket \
NCCL_DEBUG=WARN \
CUDA_VISIBLE_DEVICES=$GPUS setsid nohup /opt/anaconda3/envs/yllm/bin/vllm serve "$SNAP" \
  --served-model-name "$NAME" \
  --tensor-parallel-size "$TP" \
  --max-model-len 40960 --max-num-seqs 8 \
  --gpu-memory-utilization 0.92 --enable-prefix-caching \
  --port "$PORT" > "$LOG" 2>&1 &
for i in $(seq 1 200); do
  sleep 15
  curl -s -m 5 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "$NAME up on $PORT after $((i*15))s"; exit 0; }
  pgrep -f "[v]llm serve.*--port $PORT" >/dev/null || { echo "DIED: $NAME"; tail -25 "$LOG"; exit 1; }
done
echo "TIMEOUT: $NAME"; exit 1
