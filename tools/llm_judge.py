"""The three Who&When attribution methods, ported from the authors' code.

Follows Automated_FA/Lib/local_model.py rather than my own reading of the
paper, because several details differ and each one changes results:

* **Binary search names the halves the other way round.** In their code
  `upper half` is the *earlier* range (start..mid) and `lower half` is the
  *later* one (mid+1..end); an ambiguous answer defaults to the upper (earlier)
  half. My earlier version defined it the opposite way, which inverts where a
  directionally-biased judge ends up.
* **Step-by-step numbers the steps in the transcript** ("Step 3 - Agent: ...")
  and names the step under judgment, instead of leaving the model to count.
* **The verdict test is strict** -- startswith("1. yes") / ("1. no").
* **The transcript is `agent: content` lines**, not a JSON-like block.
* **Generation samples** (temperature 0.6, top_p 0.95, 1024 new tokens).

Their local path always puts the ground-truth answer in the prompt; the paper's
Appendix G also gives a without-answer variant, so both are kept behind
--with-gt.

Usage:
  python tools/llm_judge.py run <data-root> <out.jsonl> [--methods a,b] [--with-gt] [--limit N]
  python tools/llm_judge.py report <out.jsonl>
"""

import json
import os
import random
import re
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor

MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3.6-35B-A3B")
BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1")
METHODS = ("all_at_once", "step_by_step", "binary_search")
WIDTH = int(os.environ.get("STEP_WIDTH", "8"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "80000"))

SYSTEM = "You are a helpful assistant skilled in analyzing conversations."


def load(data_root, name, subset):
    import pandas as pd

    df = pd.read_parquet("%s/who_and_when/%s.parquet" % (data_root, name))
    out = []
    for _, r in df.iterrows():
        key = "role" if subset == "hc" else "name"     # their index_agent
        steps = [{"agent": str(s.get(key) or s.get("role") or "Unknown Agent").strip(),
                  "content": str(s.get("content") or "")} for s in list(r["history"])]
        gt = r["ground_truth"] if "ground_truth" in df.columns else r.get("groundtruth", "")
        out.append({"subset": subset, "file_id": str(r["question_ID"]),
                    "problem": str(r["question"]), "ground_truth": str(gt), "steps": steps,
                    "gold_step": int(r["mistake_step"]),
                    "gold_agent": str(r["mistake_agent"]).strip()})
    return out


def render(steps, lo, hi, numbered=False):
    """`agent: content` lines, as in their code.

    They apply no length cap and rely on the model's context. A 130-step
    hand-crafted log overflows a 40k-token window, so the oldest entries are
    emptied when necessary -- never removed, because the step index the model
    reports is positional.
    """
    parts = []
    for i in range(lo, hi + 1):
        c = " ".join((steps[i]["content"] or "").split())
        parts.append(("Step %d - %s: %s" % (i, steps[i]["agent"], c)) if numbered
                     else ("%s: %s" % (steps[i]["agent"], c)))
    total = sum(len(p) for p in parts)
    i = 0
    while total > MAX_CHARS and i < len(parts) - 1:
        short = parts[i].split(":", 1)[0] + ": [omitted for length]"
        total -= len(parts[i]) - len(short)
        parts[i] = short
        i += 1
    return "\n".join(parts)


def make_client():
    """Served endpoint by default; a local transformers pipeline with
    JUDGE_BACKEND=hf, which is how the authors' code runs local models."""
    if os.environ.get("JUDGE_BACKEND") == "hf":
        import torch
        from transformers import pipeline

        pipe = pipeline("text-generation", model=MODEL, torch_dtype=torch.bfloat16,
                        device_map="auto")
        pipe._terminators = [pipe.tokenizer.eos_token_id,
                             pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        return pipe

    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key="not-needed", timeout=600.0, max_retries=5)


def ask(client, prompt, max_tokens=1024):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    if hasattr(client, "_terminators"):                 # transformers pipeline
        out = client(msgs, max_new_tokens=max_tokens, eos_token_id=client._terminators,
                     do_sample=True, temperature=0.6, top_p=0.95,
                     pad_token_id=client.tokenizer.eos_token_id)
        gen = out[0]["generated_text"]
        return (gen[-1]["content"] if isinstance(gen, list) else str(gen)).strip()

    r = client.chat.completions.create(
        model=MODEL, messages=msgs,
        max_tokens=max_tokens, temperature=0.6, top_p=0.95,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (r.choices[0].message.content or "").strip()


def all_at_once(client, rec, with_gt):
    ans = ("The Answer for the problem is: %s\n" % rec["ground_truth"]) if with_gt else ""
    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history "
        "when solving a real world problem. The problem is:  %s \n%s"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n%s"
        "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for "
        "the wrong solution to the real world problem. If there are no agents that make obvious "
        "mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation "
        'structured as follows: {\n"agent a": "xx",\n"agent b": "xxxx",\n"agent c": "xxxxx",\n'
        '"agent a": "xxxxxxx"\n},\n'
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the "
        "speech of each agent. If the mistake is in agent c's speech, the step number is 2. If "
        "the second speech by 'agent a' contains the mistake, the step number is 3, and so on. "
        "Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n, Step Number: (Your "
        "prediction)\n, Reason for Mistake: (Your reason)\n."
        % (rec["problem"], ans, render(rec["steps"], 0, len(rec["steps"]) - 1))
    )
    text = ask(client, prompt)
    agent, step = parse_answer(text)
    return {"pred_agent": agent, "pred_step": step, "n_calls": 1, "raw": text[:2500]}


def step_by_step(client, rec, with_gt):
    """Their loop: the first step whose verdict starts '1. yes'.

    Candidates are issued in windows so the server batches them; the answer is
    unchanged, since each is judged on its own prefix and the lowest hit in the
    earliest window wins.
    """
    ans = ("The Answer for the problem is: %s\n" % rec["ground_truth"]) if with_gt else ""
    n, calls = len(rec["steps"]), 0

    def one(i):
        return (
            "You are an AI assistant tasked with evaluating the correctness of each step in an "
            "ongoing multi-agent conversation aimed at solving a real-world problem. The problem "
            "being addressed is: %s. %s"
            "Here is the conversation history up to the current step:\n%s\n"
            "The most recent step (%d) was by '%s'.\n"
            "Your task is to determine whether this most recent agent's action (Step %d) contains "
            "an error that could hinder the problem-solving process or lead to an incorrect "
            "solution. Please respond with 'Yes' or 'No' and provide a clear explanation for your "
            "judgment. Note: Please avoid being overly critical in your evaluation. Focus on "
            "errors that clearly derail the process."
            "Attention: Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
            % (rec["problem"], ans, render(rec["steps"], 0, i, numbered=True),
               i, rec["steps"][i]["agent"], i)
        )

    for base in range(0, n, WIDTH):
        idx = list(range(base, min(base + WIDTH, n)))
        if hasattr(client, "_terminators"):          # one model, no batching
            texts = [ask(client, one(i)) for i in idx]
        else:
            with ThreadPoolExecutor(max_workers=len(idx)) as ex:
                texts = list(ex.map(lambda i: ask(client, one(i)), idx))
        calls += len(idx)
        for i, text in zip(idx, texts):
            if text.lower().strip().startswith("1. yes"):
                return {"pred_agent": rec["steps"][i]["agent"], "pred_step": i,
                        "n_calls": calls, "raw": text[:600]}
    return {"pred_agent": None, "pred_step": None, "n_calls": calls, "raw": "no error found"}


def binary_search(client, rec, with_gt):
    """Their recursion. Note the naming: 'upper half' is the EARLIER range."""
    ans = ("The Answer for the problem is: %s\n" % rec["ground_truth"]) if with_gt else ""
    start, end, calls, trace = 0, len(rec["steps"]) - 1, 0, []
    while start < end:
        mid = start + (end - start) // 2
        prompt = (
            "You are an AI assistant tasked with analyzing a segment of a multi-agent "
            "conversation. Multiple agents are collaborating to address a user query, with the "
            "goal of resolving the query through their collective dialogue.\n"
            "Your primary task is to identify the location of the most critical mistake within "
            "the provided segment. Determine which half of the segment contains the single step "
            "where this crucial error occurs, ultimately leading to the failure in resolving the "
            "user's query.\n"
            "The problem to address is as follows: %s\n%s"
            "Review the following conversation segment from step %d to step %d:\n\n%s\n\n"
            "Based on your analysis, predict whether the most critical error is more likely to be "
            "located in the upper half (from step %d to step %d) or the lower half (from step %d "
            "to step %d) of this segment.\n"
            "Please simply output either 'upper half' or 'lower half'. You should not output "
            "anything else."
            % (rec["problem"], ans, start, end, render(rec["steps"], start, end),
               start, mid, mid + 1, end)
        )
        out = ask(client, prompt, max_tokens=16).lower().strip()
        calls += 1
        if "upper half" in out:
            trace.append("upper"); end = mid
        elif "lower half" in out:
            trace.append("lower"); start = min(mid + 1, end)
        else:
            trace.append("ambiguous"); end = mid        # their default
    return {"pred_agent": rec["steps"][start]["agent"], "pred_step": start,
            "n_calls": calls, "raw": ">".join(trace)}


RUNNERS = {"all_at_once": all_at_once, "step_by_step": step_by_step,
           "binary_search": binary_search}


def parse_answer(text):
    """Agent and step from a free-form answer, tolerating markdown labels."""
    agent = step = None
    m = re.search(r"Agent\s*(?:Name|Identification)\s*\**\s*:\s*\**\s*(.+)", text, re.I)
    if m:
        agent = m.group(1).strip().split("\n")[0]
    m = re.search(r"Step\s*(?:Number|Index)?\s*\**\s*:\s*\**\s*(-?\d+)", text, re.I)
    if m:
        step = int(m.group(1))
    if agent is None:
        m = re.search(r"^\s*1[\.\)]\s*\**\s*(.+)$", text, re.M)
        if m:
            agent = m.group(1).strip()
    if step is None:
        m = re.search(r"^\s*2[\.\)]\s*\**\s*(-?\d+)", text, re.M)
        if m:
            step = int(m.group(1))
    if step is None:
        m = re.search(r"step\s*(?:number)?\s*(?:is|:)?\s*\**\s*(\d+)", text, re.I)
        if m:
            step = int(m.group(1))
    if agent:
        agent = agent.strip().strip("*").strip(" .*():,")
        if len(agent) > 60:
            b = re.search(r"\*\*([^*]{2,60})\*\*", text)
            agent = (b.group(1) if b else agent.split()[0]).strip(" .*():,")
    return agent or None, step


def cmd_run(argv):
    data_root, out_path = argv[0], argv[1]
    methods = (argv[argv.index("--methods") + 1].split(",")
               if "--methods" in argv else list(METHODS))
    with_gt = "--with-gt" in argv
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None

    recs = load(data_root, "Algorithm-Generated", "alg") + load(data_root, "Hand-Crafted", "hc")
    if limit:
        recs = ([r for r in recs if r["subset"] == "alg"][:limit]
                + [r for r in recs if r["subset"] == "hc"][:limit])

    seen = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                d = json.loads(line)
                seen.add((d.get("judge"), d["method"], d["subset"], d["file_id"]))

    client = make_client()
    t0, done = time.time(), 0
    with open(out_path, "a") as fh:
        for method in methods:
            for rec in recs:
                if (MODEL, method, rec["subset"], rec["file_id"]) in seen:
                    continue
                try:
                    out = RUNNERS[method](client, rec, with_gt)
                except Exception as e:                       # noqa: BLE001
                    out = {"pred_agent": None, "pred_step": None, "n_calls": 0,
                           "raw": "ERROR: %s" % e}
                out.update({"judge": MODEL, "method": method, "with_gt": with_gt,
                            "subset": rec["subset"], "file_id": rec["file_id"],
                            "gold_step": rec["gold_step"], "gold_agent": rec["gold_agent"],
                            "n_steps": len(rec["steps"])})
                fh.write(json.dumps(out) + "\n")
                fh.flush()
                done += 1
                if done % 20 == 0:
                    sys.stderr.write("  %s: %d done, %.0fs\n" % (method, done, time.time() - t0))
                    sys.stderr.flush()
    sys.stderr.write("wrote %d rows to %s\n" % (done, out_path))
    return 0


def boot(v, n=2000, seed=0):
    rng = random.Random(seed)
    m = [sum(v[rng.randrange(len(v))] for _ in range(len(v))) / len(v) for _ in range(n)]
    m.sort()
    return m[int(.025 * n)], m[int(.975 * n)]


def fmt(v):
    lo, hi = boot(v)
    return "%.3f [%.3f,%.3f]" % (st.fmean(v), lo, hi)


def _speaker(role):
    """The actual speaker of a step.

    Hand-crafted roles encode more than identity: "Orchestrator (thought)",
    "Orchestrator (termination condition)", "Orchestrator (-> WebSurfer)". The
    last form names a delegation TARGET, so a plain substring test credits a
    hit whenever the Orchestrator delegates to the faulty agent -- the speaker
    is the Orchestrator. That inflated hand-crafted agent accuracy by 4 of 58
    trajectories (0.672 vs the correct 0.603).
    """
    return re.sub(r"\s*\(.*?\)\s*", "", str(role or "")).strip().lower()


def amatch(gold, picked):
    g, p = str(gold or "").strip().lower(), _speaker(picked)
    return bool(g and p and (g in p or p in g))


def cmd_report(argv):
    rows = [json.loads(l) for l in open(argv[0]) if l.strip()]
    keys = sorted({(r.get("judge", "?"), r["method"], r["with_gt"], r["subset"]) for r in rows})
    print("| judge | method | answer | corpus | logs | step | agent | no answer | calls |")
    print("|---|---|---|---|---|---|---|---|---|")
    for judge, method, gt, sub in keys:
        sel = [r for r in rows
               if (r.get("judge", "?"), r["method"], r["with_gt"], r["subset"])
               == (judge, method, gt, sub)]
        S = [1.0 if r["pred_step"] == r["gold_step"] else 0.0 for r in sel]
        A = [1.0 if amatch(r["pred_agent"], r["gold_agent"]) else 0.0 for r in sel]
        bad = sum(1 for r in sel if r["pred_step"] is None and r["pred_agent"] is None)
        print("| %s | %s | %s | %s | %d | %s | %s | %d | %.1f |"
              % (judge.split("/")[-1], method, "shown" if gt else "hidden", sub,
                 len(sel), fmt(S), fmt(A), bad,
                 st.fmean([r.get("n_calls", 0) for r in sel])))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        sys.exit(cmd_run(sys.argv[2:]))
    if cmd == "report":
        sys.exit(cmd_report(sys.argv[2:]))
    print(__doc__)
    sys.exit(2)
