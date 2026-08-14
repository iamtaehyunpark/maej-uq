"""The three Who&When failure-attribution methods, run against a local judge.

Prompts are transcribed verbatim from Appendix G of arXiv:2505.00212 and the
control flow from Appendix A (Algorithm 1 for step-by-step, Algorithm 2 for
binary search). Each method has a without-ground-truth and a with-ground-truth
variant, exactly as the paper gives two prompts per method.

Step numbering follows the paper's own convention, stated in G.1: the first
entry is step 0 ("If the mistake is in agent c's speech, the step number is 2"
for a log of agent a / agent b / agent c). That matches Who&When's released
`mistake_step`, which is a 0-based index into `history` — verified empirically:
the annotated agent matches `history[mistake_step]` for 97.6% of
algorithm-generated and 91.4% of hand-crafted logs, against 17.5% / 32.8% under
1-based indexing.

Cost differs sharply by method and is reported per run:
  all_at_once    1 call per log
  binary_search  ~log2(n) calls per log
  step_by_step   up to n calls per log (early-exits at the first flagged step)

Runs in the `Jagent` env against a `yllm` vLLM server.

Usage:
  python tools/llm_judge.py run <data-root> <out.jsonl> [--methods all_at_once,...]
                                [--with-gt] [--limit N]
  python tools/llm_judge.py report <out.jsonl>
"""

import json
import os
import random
import re
import statistics as st
import sys
import time

MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3.6-35B-A3B")
BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1")
STEP_CHARS = 1500        # per-entry cap when rendering a log
LOG_CHARS = 90000        # whole-log cap for all_at_once / binary_search
METHODS = ("all_at_once", "step_by_step", "binary_search")
STEP_WIDTH = int(os.environ.get("STEP_WIDTH", "8"))   # parallel window for step_by_step

# --- prompts, verbatim from Appendix G --------------------------------------

ALL_AT_ONCE = """You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.
The problem is: {problem}.
{gt_line}Identify which agent made an error, at which step, and explain the reason for the error.
Here's the conversation: {failure_log}
Based on this conversation, please predict the following:
1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.
2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows:
{{
"agent a": "xx",
"agent b": "xxxx",
"agent c": "xxxxx",
"agent a": "xxxxxxx"
}},
each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.
3. The reason for your prediction. Please answer in the format:
Agent Name: (Your prediction)
Step Number: (Your prediction)
Reason for Mistake: (Your reason)"""

BINARY_SEARCH = """You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.
Your primary task is to identify location of the most critical mistake, and determine the single step in the conversation where this error occurs, ultimately leading to the failure in resolving the user's query.
The problem to address is as follows: {problem}.
{gt_line}Review the following conversation range
{range_description}: {sliced_log}.
Based on your analysis, predict whether the error is more likely to be located in the upper or lower half of the segment. lower half is defined as the range {lower_half_range} and upper half is defined as the range {upper_half_range}.
Please simply output either 'upper half' or 'lower half'.
You should not output anything else."""

STEP_BY_STEP = """You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem.
The problem being addressed is: {problem}.
Here is the conversation history up to the current step: {failure_log}.
{gt_line}Your task is to determine whether the most recent agent's action contains an error that could hinder the problem-solving process. Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment.
Note: Please avoid being overly critical in your evaluation.
Attention: Respond in the format:
1. Yes/No. 2. Reason for the judgment."""

GT_LINE = "The Answer for the problem is: {ground_truth}.\n"


# --- data -------------------------------------------------------------------


def load(data_root, name, subset):
    import pandas as pd

    df = pd.read_parquet("%s/who_and_when/%s.parquet" % (data_root, name))
    out = []
    for _, r in df.iterrows():
        steps = []
        for s in list(r["history"]):
            agent = (s.get("name") or s.get("role") or "") if subset == "alg" else (s.get("role") or "")
            steps.append({"agent": str(agent).strip(), "content": str(s.get("content") or "")})
        gt = r["ground_truth"] if "ground_truth" in df.columns else r.get("groundtruth", "")
        out.append({
            "subset": subset, "file_id": str(r["question_ID"]), "problem": str(r["question"]),
            "ground_truth": str(gt), "steps": steps,
            "gold_step": int(r["mistake_step"]), "gold_agent": str(r["mistake_agent"]).strip(),
        })
    return out


def render(steps, lo, hi, budget=LOG_CHARS):
    """Entries lo..hi inclusive, in the shape the all-at-once prompt illustrates.

    Oldest entries lose their content first when over budget, so the numbering
    the prompt asks the model to count stays intact — dropping entries outright
    would silently shift every step index after the cut.
    """
    parts = []
    for i in range(lo, hi + 1):
        c = " ".join((steps[i]["content"] or "").split())
        if len(c) > STEP_CHARS:
            c = c[:STEP_CHARS // 2] + " ...[elided]... " + c[-STEP_CHARS // 2:]
        parts.append('"%s": "%s"' % (steps[i]["agent"], c.replace('"', "'")))
    total = sum(len(p) for p in parts)
    i = 0
    while total > budget and i < len(parts) - 1:
        short = '"%s": "...[omitted for length]..."' % steps[lo + i]["agent"]
        total -= len(parts[i]) - len(short)
        parts[i] = short
        i += 1
    return "{\n" + ",\n".join(parts) + "\n}"


# --- transport --------------------------------------------------------------


def make_client():
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key="not-needed", timeout=600.0, max_retries=5)


def ask(client, prompt, max_tokens=512):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (r.choices[0].message.content or "").strip()


# --- parsing ----------------------------------------------------------------


def parse_all_at_once(text):
    """Accept both the requested labelled form and the numbered form.

    The prompt asks for ``Agent Name: ... / Step Number: ...``, but a judge that
    reads the three numbered questions often answers ``1. <agent>`` / ``2. <n>``
    instead. Parsing only the labelled form scores every such answer as a
    failure of the method rather than of the parser.
    """
    agent = step = None
    m = re.search(r"Agent\s*Name\s*:\s*\**\s*(.+)", text, re.I)
    if m:
        agent = m.group(1).strip().strip("*").split("\n")[0].strip(" .*()")
    m = re.search(r"Step\s*Number\s*:\s*\**\s*(-?\d+)", text, re.I)
    if m:
        step = int(m.group(1))
    if agent is None:
        m = re.search(r"^\s*1[\.\)]\s*\**\s*(.+)$", text, re.M)
        if m:
            agent = m.group(1).strip().strip("*").strip(" .*()")
    if step is None:
        m = re.search(r"^\s*2[\.\)]\s*\**\s*(-?\d+)", text, re.M)
        if m:
            step = int(m.group(1))
    # An agent field that swallowed a whole sentence is not a name.
    if agent and len(agent) > 60:
        agent = agent.split()[0].strip(" .*():,")
    return agent, step


def parse_yes_no(text):
    """First standalone Yes/No. Returns True for Yes, False for No, None if absent."""
    m = re.search(r"\b(yes|no)\b", text, re.I)
    return None if not m else (m.group(1).lower() == "yes")


def parse_half(text):
    t = text.lower()
    lo, up = t.find("lower half"), t.find("upper half")
    if lo < 0 and up < 0:
        return None
    if lo < 0:
        return "upper"
    if up < 0:
        return "lower"
    return "lower" if lo < up else "upper"


# --- the three methods ------------------------------------------------------


def run_all_at_once(client, rec, with_gt):
    gt = GT_LINE.format(ground_truth=rec["ground_truth"]) if with_gt else ""
    p = ALL_AT_ONCE.format(problem=rec["problem"], gt_line=gt,
                           failure_log=render(rec["steps"], 0, len(rec["steps"]) - 1))
    text = ask(client, p)
    agent, step = parse_all_at_once(text)
    return {"pred_agent": agent, "pred_step": step, "n_calls": 1, "raw": text[:600]}


def run_step_by_step(client, rec, with_gt, width=STEP_WIDTH):
    """Algorithm 1: the first step the judge calls an error.

    Evaluated in parallel windows rather than one call at a time. The answer is
    identical -- the lowest-indexed step flagged Yes -- because within a window
    every candidate is judged on its own prefix and the smallest hit wins, and
    windows are processed in order. Only the cost changes: more calls issued,
    but the server batches them instead of sitting idle behind one request. On a
    130-step log against a 70B that is the difference between minutes and an
    hour.
    """
    from concurrent.futures import ThreadPoolExecutor

    gt = GT_LINE.format(ground_truth=rec["ground_truth"]) if with_gt else ""
    n = len(rec["steps"])
    calls = 0
    for base in range(0, n, width):
        idx = list(range(base, min(base + width, n)))
        prompts = [STEP_BY_STEP.format(problem=rec["problem"], gt_line=gt,
                                       failure_log=render(rec["steps"], 0, i))
                   for i in idx]
        with ThreadPoolExecutor(max_workers=len(idx)) as ex:
            texts = list(ex.map(lambda q: ask(client, q), prompts))
        calls += len(idx)
        for i, text in zip(idx, texts):
            if parse_yes_no(text):
                return {"pred_agent": rec["steps"][i]["agent"], "pred_step": i,
                        "n_calls": calls, "raw": text[:300]}
    # "No error found" -- the paper leaves this unspecified; scored as a miss
    # rather than silently defaulting to a step.
    return {"pred_agent": None, "pred_step": None, "n_calls": calls,
            "raw": "no error found"}


def run_binary_search(client, rec, with_gt):
    """Algorithm 2, driven by the upper/lower-half question the G.2 prompt asks."""
    gt = GT_LINE.format(ground_truth=rec["ground_truth"]) if with_gt else ""
    lo, hi, calls, trace, unparsed = 0, len(rec["steps"]) - 1, 0, [], 0
    while lo < hi:
        mid = (lo + hi) // 2
        p = BINARY_SEARCH.format(
            problem=rec["problem"], gt_line=gt,
            range_description="steps %d to %d" % (lo, hi),
            sliced_log=render(rec["steps"], lo, hi),
            lower_half_range="steps %d to %d" % (lo, mid),
            upper_half_range="steps %d to %d" % (mid + 1, hi),
        )
        raw = ask(client, p, max_tokens=8)
        half = parse_half(raw)
        calls += 1
        # Recorded per decision: a judge that answers the same direction every
        # time walks the search to one end of the log regardless of content, and
        # that is indistinguishable from localisation unless the answers are kept.
        trace.append(half or raw[:20])
        if half is None:
            unparsed += 1
        if half == "upper":
            lo = mid + 1
        else:            # 'lower', and an unparseable answer keeps the earlier half
            hi = mid
    return {"pred_agent": rec["steps"][lo]["agent"], "pred_step": lo,
            "n_calls": calls, "n_unparsed_halves": unparsed,
            "raw": ">".join(trace)}


RUNNERS = {"all_at_once": run_all_at_once, "step_by_step": run_step_by_step,
           "binary_search": run_binary_search}


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

    # Resume by key rather than blind append. The file is opened in append mode
    # so an interrupted pass can continue, but appending alone silently doubles
    # every row if the driver is started twice -- which happened once already.
    seen = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                d = json.loads(line)
                seen.add((d.get("judge"), d["method"], d["subset"], d["file_id"]))
        if seen:
            sys.stderr.write("resuming: %d rows already present, skipping those\n" % len(seen))

    client = make_client()
    t0 = time.time()
    done = skipped = 0
    with open(out_path, "a") as fh:
        for method in methods:
            for rec in recs:
                if (MODEL, method, rec["subset"], rec["file_id"]) in seen:
                    skipped += 1
                    continue
                try:
                    out = RUNNERS[method](client, rec, with_gt)
                except Exception as e:                      # noqa: BLE001
                    out = {"pred_agent": None, "pred_step": None, "n_calls": 0,
                           "raw": "ERROR: %s" % e}
                out.update({"judge": MODEL,
                            "method": method, "with_gt": with_gt, "subset": rec["subset"],
                            "file_id": rec["file_id"], "gold_step": rec["gold_step"],
                            "gold_agent": rec["gold_agent"], "n_steps": len(rec["steps"])})
                fh.write(json.dumps(out) + "\n")
                fh.flush()
                done += 1
                if done % 20 == 0:
                    sys.stderr.write("  %s: %d done, %.0fs\n" % (method, done, time.time() - t0))
                    sys.stderr.flush()
    sys.stderr.write("wrote %d rows (%d already present) to %s\n"
                     % (done, skipped, out_path))
    return 0


# --- scoring ----------------------------------------------------------------


def boot(v, n=2000, seed=0):
    rng = random.Random(seed)
    m = []
    for _ in range(n):
        s = [v[rng.randrange(len(v))] for _ in range(len(v))]
        m.append(sum(s) / len(s))
    m.sort()
    return m[int(.025 * n)], m[int(.975 * n)]


def fmt(v):
    lo, hi = boot(v)
    return "%.3f [%.3f,%.3f]" % (st.fmean(v), lo, hi)


def amatch(a, b):
    a, b = str(a or "").strip().lower(), str(b or "").strip().lower()
    return bool(a and b and (a in b or b in a))


def cmd_report(argv):
    rows = [json.loads(l) for l in open(argv[0]) if l.strip()]
    keys = sorted({(r.get("judge", "?"), r["method"], r["with_gt"], r["subset"]) for r in rows})
    print("| judge | method | answer | corpus | logs | spots the step | spots the agent | unparsed | calls/log |")
    print("|---|---|---|---|---|---|---|---|---|")
    for judge, method, gt, sub in keys:
        sel = [r for r in rows
               if (r.get("judge", "?"), r["method"], r["with_gt"], r["subset"])
               == (judge, method, gt, sub)]
        S = [1.0 if r["pred_step"] == r["gold_step"] else 0.0 for r in sel]
        A = [1.0 if amatch(r["pred_agent"], r["gold_agent"]) else 0.0 for r in sel]
        bad = sum(1 for r in sel if r["pred_step"] is None and r["pred_agent"] is None)
        calls = st.fmean([r["n_calls"] for r in sel])
        print("| %s | %s | %s | %s | %d | %s | %s | %d | %.1f |"
              % (judge.split("/")[-1], method, "shown" if gt else "hidden",
                 "algorithm-generated" if sub == "alg" else "hand-crafted",
                 len(sel), fmt(S), fmt(A), bad, calls))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        sys.exit(cmd_run(sys.argv[2:]))
    if cmd == "report":
        sys.exit(cmd_report(sys.argv[2:]))
    print(__doc__)
    sys.exit(2)
