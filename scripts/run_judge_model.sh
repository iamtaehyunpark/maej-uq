#!/usr/bin/env bash
# run_judge_model.sh <model-name> <port> <tag>
#
# Runs both answer settings for one judge, then releases its GPU.
#
# The teardown is a trap rather than a line at the end: a judging pass can take
# hours, and a run that is killed, times out, or dies partway would otherwise
# leave a 70GB server resident on a shared box with nothing pointed at it. EXIT
# fires on normal completion and on INT/TERM alike.
set -u
R=/data5/kje/MULTIAGENT/maej-uq; J=/opt/anaconda3/envs/Jagent/bin/python
D=/data5/kje/MULTIAGENT/data; O=$R/runs/judges
NAME=$1; PORT=$2; TAG=$3
mkdir -p "$O"; cd "$R" || exit 1

release() {
  local code=$?
  echo "[$(date +%T)] releasing $TAG (port $PORT, exit $code)"
  # [v] keeps the pattern from matching this script's own command line.
  pkill -f "[v]llm serve.*--port $PORT" 2>/dev/null
  for i in $(seq 1 20); do
    pgrep -f "[v]llm serve.*--port $PORT" >/dev/null || break
    sleep 3
  done
  pgrep -f "[v]llm serve.*--port $PORT" >/dev/null && {
    echo "[$(date +%T)] $TAG did not stop; forcing"
    pkill -9 -f "[v]llm serve.*--port $PORT" 2>/dev/null
    sleep 5
  }
  echo "[$(date +%T)] $TAG released"
}
trap release EXIT INT TERM

for arm in nogt gt; do
  FLAG=""; [ "$arm" = "gt" ] && FLAG="--with-gt"
  echo "[$(date +%T)] $TAG $arm starting"
  JUDGE_MODEL="$NAME" JUDGE_BASE_URL="http://localhost:$PORT/v1" \
    $J tools/llm_judge.py run "$D" "$O/${TAG}_${arm}.jsonl" $FLAG > "$O/${TAG}_${arm}.log" 2>&1
  echo "[$(date +%T)] $TAG $arm exit=$?"
done
echo "${TAG}_DONE"
