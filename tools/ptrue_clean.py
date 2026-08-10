"""Pure P(True) on Who&When. Standalone: no harness import, no rules, no
normalization layer.

Runs in the `Jagent` env (python 3.9, openai 1.66.3) against a `yllm` vLLM
server. Nothing here is shared with `src/masattr` — that is the point. If this
agrees with the harness, the harness is corroborated by an independent path; if
it disagrees, one of them is wrong and we find out.

Three things it does differently from the harness's judge client, each of which
was a defect there:

1. **Chat endpoint.** The judge is an instruct model. Posting raw text to
   `/completions` skips its chat template, leaving it outside the format it was
   trained in — which is what pushed the answer tokens out of the head of the
   distribution in the first place.
2. **Thinking off via the template's own switch**
   (`chat_template_kwargs={"enable_thinking": False}`), not by prefilling a
   literal `<think></think>` string into the prompt.
3. **Probabilities summed over spellings**, not maxed, and the mass actually
   sitting on {True, False} recorded on every row. If the model is not
   answering the question, that number says so rather than a ratio of two tail
   probabilities hiding it.

Label convention: `mistake_step` is a 0-based index into `history`. Verified
empirically — the annotated agent matches `history[mistake_step]` for 97.6% of
algorithm-generated and 91.4% of hand-crafted logs, against 17.5% / 32.8% under
1-based indexing.

Usage:
  python tools/ptrue_clean.py score <data-root> <out.jsonl> [--limit N] [--with-gt]
  python tools/ptrue_clean.py report <out.jsonl>
"""

import json
import math
import statistics as st
import sys
import time
from collections import Counter

MODEL = "Qwen/Qwen3.6-35B-A3B"
BASE_URL = __import__("os").environ.get("PTRUE_BASE_URL", "http://localhost:8000/v1")
STEP_CHARS = 2000       # per-step content cap
PREFIX_CHARS = 60000    # total transcript cap; oldest steps drop content first

SYSTEM = (
    "You are auditing a transcript from a multi-agent system that failed to "
    "solve its task. You will see the task and the transcript up to and "
    "including one step. Judge only that step, given everything before it. "
    "Do not use hindsight you do not have."
)


def load(data_root, name, subset):
    import pandas as pd

    df = pd.read_parquet("%s/who_and_when/%s.parquet" % (data_root, name))
    out = []
    for i, r in df.iterrows():
        hist = list(r["history"])
        steps = []
        for s in hist:
            agent = (s.get("name") or s.get("role") or "") if subset == "alg" else (s.get("role") or "")
            steps.append({"agent": str(agent).strip(), "content": str(s.get("content") or "")})
        gt = r["ground_truth"] if "ground_truth" in df.columns else r.get("groundtruth", "")
        out.append(
            {
                "subset": subset,
                "file_id": str(r["question_ID"]),
                "question": str(r["question"]),
                "ground_truth": str(gt),
                "steps": steps,
                "gold_step": int(r["mistake_step"]),
                "gold_agent": str(r["mistake_agent"]).strip(),
            }
        )
    return out


def render(step, i, cap=STEP_CHARS):
    c = step["content"]
    if len(c) > cap:
        h = cap // 2
        c = c[:h] + "\n…[%d chars elided]…\n" % (len(c) - cap) + c[-h:]
    return "[step %d | agent=%s]\n%s\n" % (i, step["agent"], c)


def build_user(rec, t, with_gt):
    """Task, transcript through step t, then the question. Prefix only."""
    head = ["[task]", rec["question"]]
    if with_gt:
        head += ["", "[reference answer]", rec["ground_truth"]]
    head += ["", "[transcript]"]

    body = [render(rec["steps"][i], i) for i in range(t + 1)]
    # Oldest-first shrink if the transcript is too long. The step being judged
    # is never shortened beyond the per-step cap.
    total = sum(len(b) for b in body)
    i = 0
    while total > PREFIX_CHARS and i < len(body) - 1:
        short = "[step %d | agent=%s]\n…[content omitted for length]…\n" % (
            i, rec["steps"][i]["agent"],
        )
        total -= len(body[i]) - len(short)
        body[i] = short
        i += 1

    step = rec["steps"][t]
    tail = [
        "",
        "Question: Is step %d by '%s' correct and appropriate given the context "
        "above?" % (t, step["agent"]),
        "Answer with exactly one word: True or False.",
    ]
    return "\n".join(head) + "\n" + "".join(body) + "\n".join(tail)


def make_client():
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key="not-needed", timeout=600.0, max_retries=5)


def p_true(client, user):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        max_tokens=1,
        temperature=0.0,
        logprobs=True,
        top_logprobs=20,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    ch = r.choices[0]
    top = []
    if ch.logprobs and ch.logprobs.content:
        top = ch.logprobs.content[0].top_logprobs or []

    pt = pf = 0.0
    for item in top:
        w = str(item.token).strip().lower()
        p = math.exp(float(item.logprob))
        if w == "true":
            pt += p
        elif w == "false":
            pf += p
    answered = pt + pf
    return {
        "p_true": (pt / answered) if answered > 0 else None,
        "mass_on_answer": answered,
        "top_token": str(top[0].token) if top else "",
    }


def cmd_score(argv):
    data_root = argv[0]
    out_path = argv[1]
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    with_gt = "--with-gt" in argv

    recs = load(data_root, "Algorithm-Generated", "alg") + load(data_root, "Hand-Crafted", "hc")
    if limit:
        alg = [r for r in recs if r["subset"] == "alg"][:limit]
        hc = [r for r in recs if r["subset"] == "hc"][:limit]
        recs = alg + hc

    client = make_client()
    n = 0
    t0 = time.time()
    with open(out_path, "w") as fh:
        for k, rec in enumerate(recs):
            for t in range(len(rec["steps"])):
                r = p_true(client, build_user(rec, t, with_gt))
                r.update(
                    {
                        "subset": rec["subset"],
                        "file_id": rec["file_id"],
                        "step_idx": t,
                        "agent": rec["steps"][t]["agent"],
                        "gold_step": rec["gold_step"],
                        "gold_agent": rec["gold_agent"],
                        "n_steps": len(rec["steps"]),
                        "with_gt": with_gt,
                    }
                )
                fh.write(json.dumps(r) + "\n")
                n += 1
            fh.flush()
            if (k + 1) % 10 == 0:
                el = time.time() - t0
                sys.stderr.write(
                    "  %d/%d files, %d steps, %.0fs (%.1f steps/s)\n"
                    % (k + 1, len(recs), n, el, n / max(el, 1))
                )
                sys.stderr.flush()
    sys.stderr.write("wrote %d rows to %s\n" % (n, out_path))
    return 0


def boot(vals, n_boot=2000, seed=0):
    import random

    rng = random.Random(seed)
    m = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        m.append(sum(sample) / len(sample))
    m.sort()
    return m[int(0.025 * n_boot)], m[int(0.975 * n_boot)]


def cmd_report(argv):
    """Top-1 and top-2 accuracy: rank steps by P(True) ascending (least
    confident first) and ask whether the labelled step, and the labelled agent,
    are in the top k."""
    rows = [json.loads(l) for l in open(argv[0]) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault((r["subset"], r["file_id"]), []).append(r)

    miss = [r for r in rows if r["p_true"] is None]
    mass = [r["mass_on_answer"] for r in rows if r["p_true"] is not None]
    print("rows %d   trajectories %d   with_gt=%s"
          % (len(rows), len(by), rows[0].get("with_gt")))
    print("rows with neither True nor False in top-20: %d (%.2f%%)"
          % (len(miss), 100.0 * len(miss) / len(rows)))
    if mass:
        print("mass on True+False: mean %.4f  median %.4f  min %.4f"
              % (st.fmean(mass), st.median(mass), min(mass)))
    print("most common first token:", Counter(r["top_token"] for r in rows).most_common(4))
    print()

    hdr = "%-6s %6s   %-22s %-22s %-22s %-22s"
    print(hdr % ("logs", "files", "step top-1", "step top-2", "agent top-1", "agent top-2"))
    for subset in ("alg", "hc"):
        s1, s2, a1, a2 = [], [], [], []
        for (sub, _f), sc in by.items():
            if sub != subset:
                continue
            sc = [x for x in sc if x["p_true"] is not None]
            if len(sc) < 2:
                continue
            gs, ga = sc[0]["gold_step"], str(sc[0]["gold_agent"]).strip().lower()
            if not any(x["step_idx"] == gs for x in sc):
                continue  # released label points outside the trajectory
            ranked = sorted(sc, key=lambda x: x["p_true"])  # least confident first

            def agent_hit(k):
                names = [str(x["agent"]).strip().lower() for x in ranked[:k]]
                return 1.0 if any(ga and n and (ga in n or n in ga) for n in names) else 0.0

            s1.append(1.0 if ranked[0]["step_idx"] == gs else 0.0)
            s2.append(1.0 if gs in [x["step_idx"] for x in ranked[:2]] else 0.0)
            a1.append(agent_hit(1))
            a2.append(agent_hit(2))
        if not s1:
            continue
        cells = []
        for v in (s1, s2, a1, a2):
            lo, hi = boot(v)
            cells.append("%.3f [%.3f,%.3f]" % (st.fmean(v), lo, hi))
        print(hdr % (subset, len(s1), cells[0], cells[1], cells[2], cells[3]))
    print()
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "score":
        sys.exit(cmd_score(sys.argv[2:]))
    if cmd == "report":
        sys.exit(cmd_report(sys.argv[2:]))
    print(__doc__)
    sys.exit(2)
