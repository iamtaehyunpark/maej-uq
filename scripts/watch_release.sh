#!/usr/bin/env bash
# watch_release.sh <pid> <port-pattern> <label>
# Waits for an already-running judging job to end, then frees its server.
# Needed for jobs launched before the teardown trap existed.
set -u
PID=$1; PORTPAT=$2; LABEL=$3
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "[$(date +%T)] $LABEL job $PID ended; releasing $PORTPAT"
pkill -f "[v]llm serve.*--port $PORTPAT" 2>/dev/null
sleep 20
pkill -9 -f "[v]llm serve.*--port $PORTPAT" 2>/dev/null
sleep 5
echo "[$(date +%T)] $LABEL released"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
