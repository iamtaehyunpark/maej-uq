#!/usr/bin/env bash
set -u
R=/data5/kje/MULTIAGENT/maej-uq; J=/opt/anaconda3/envs/Jagent/bin/python
D=/data5/kje/MULTIAGENT/data; O=$R/runs/judges
mkdir -p "$O"; cd "$R" || exit 1

# Release both servers when this finishes, is killed, or dies partway.
release() {
  echo "[$(date +%T)] releasing 35B servers (8000, 8001)"
  pkill -f "[v]llm serve.*--port 800[01]" 2>/dev/null
  sleep 20
  pkill -9 -f "[v]llm serve.*--port 800[01]" 2>/dev/null
}
trap release EXIT INT TERM
bash serve.sh 1 8001 >> "$O/serve8001.log" 2>&1
rm -f "$O/nogt.jsonl" "$O/gt.jsonl"
( JUDGE_BASE_URL=http://localhost:8000/v1 $J tools/llm_judge.py run "$D" "$O/nogt.jsonl" > "$O/nogt.log" 2>&1; echo "nogt exit=$?" ) &
( JUDGE_BASE_URL=http://localhost:8001/v1 $J tools/llm_judge.py run "$D" "$O/gt.jsonl" --with-gt > "$O/gt.log" 2>&1; echo "gt exit=$?" ) &
wait
echo JUDGES_DONE
wc -l "$O"/*.jsonl
