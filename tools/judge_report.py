"""Assemble the judge x method grid into docs/JUDGE_RESULTS.md.

Three judges, three attribution methods, two answer settings, two corpora.
Reads every runs/judges/*.jsonl, so the tables cannot disagree with the runs.

Usage: python tools/judge_report.py <runs/judges dir> [> docs/JUDGE_RESULTS.md]
"""

import collections
import json
import glob
import os
import random
import statistics as st
import sys

JUDGES = [
    ("Qwen/Qwen3.5-9B", "Qwen3.5-9B", "9B dense"),
    ("Qwen/Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B", "35B MoE, ~3B active"),
    ("meta-llama/Llama-3.3-70B-Instruct", "Llama-3.3-70B", "70B dense"),
]
METHODS = ["all_at_once", "step_by_step", "binary_search"]
CORPORA = [("alg", "algorithm-generated"), ("hc", "hand-crafted")]


def boot(v, n=2000, seed=0):
    rng = random.Random(seed)
    m = []
    for _ in range(n):
        s = [v[rng.randrange(len(v))] for _ in range(len(v))]
        m.append(sum(s) / len(s))
    m.sort()
    return m[int(.025 * n)], m[int(.975 * n)]


def fmt(v):
    if not v:
        return "—"
    lo, hi = boot(v)
    return "%.3f [%.3f,%.3f]" % (st.fmean(v), lo, hi)


def amatch(a, b):
    a, b = str(a or "").strip().lower(), str(b or "").strip().lower()
    return bool(a and b and (a in b or b in a))


def load(dirpath):
    rows, seen = [], set()
    for f in sorted(glob.glob(os.path.join(dirpath, "*.jsonl"))):
        if os.sep + "backup" + os.sep in f:
            continue
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r.get("judge"), r["method"], r["subset"], r["file_id"], r["with_gt"])
            if k in seen:      # the same cell may appear in a rerun file
                continue
            seen.add(k)
            rows.append(r)
    return rows


def cells(rows, judge, method, gt, sub):
    sel = [r for r in rows
           if r.get("judge") == judge and r["method"] == method
           and r["with_gt"] == gt and r["subset"] == sub]
    if not sel:
        return None
    S = [1.0 if r["pred_step"] == r["gold_step"] else 0.0 for r in sel]
    A = [1.0 if amatch(r["pred_agent"], r["gold_agent"]) else 0.0 for r in sel]
    none = sum(1 for r in sel if r["pred_step"] is None and r["pred_agent"] is None)
    return {"n": len(sel), "S": S, "A": A, "none": none,
            "calls": st.fmean([r.get("n_calls", 0) for r in sel])}


def main(argv):
    d = argv[1] if len(argv) > 1 else "runs/judges"
    rows = load(d)

    out = ["""# LLM-judge failure attribution — three judges, three methods

The three attribution methods from *Which Agent Causes Task Failures and When?*
(arXiv:2505.00212), run against three local judges. Prompts are transcribed
verbatim from the paper's Appendix G and the control flow from Appendix A;
each method has a without-answer and a with-answer variant, as the paper gives
two prompts per method.

**These are not the paper's numbers.** The paper uses GPT-4o. Everything here
is a local open-weights judge, so the grid is a judge-capability comparison
under the paper's own methods, not a reproduction of its figures.

Corpus: Who&When — 126 algorithm-generated logs and 58 hand-crafted ones.
*Spots the step* is exact match against the annotated mistake step, *spots the
agent* against the annotated agent. Intervals are bootstrapped over logs, 2,000
resamples. `mistake_step` is 0-based, matching the paper's own convention in
G.1.

| judge | size |
|---|---|"""]
    for jid, short, desc in JUDGES:
        if any(r.get("judge") == jid for r in rows):
            out.append("| `%s` | %s |" % (short, desc))

    for gt, glabel in ((False, "Answer hidden"), (True, "Answer shown")):
        out += ["", "## %s" % glabel, ""]
        for sub, slabel in CORPORA:
            out += ["**%s**" % slabel, "",
                    "| judge | method | logs | spots the step | spots the agent | no answer | calls/log |",
                    "|---|---|---|---|---|---|---|"]
            for jid, short, _ in JUDGES:
                for m in METHODS:
                    c = cells(rows, jid, m, gt, sub)
                    if not c:
                        out.append("| %s | %s | — | *not run* | | | |" % (short, m))
                        continue
                    out.append("| %s | %s | %d | %s | %s | %d | %.1f |"
                               % (short, m, c["n"], fmt(c["S"]), fmt(c["A"]),
                                  c["none"], c["calls"]))
            out.append("")

    # binary-search decision trace: a judge that answers one direction every
    # time walks to one end of the log regardless of content.
    tr = [r for r in rows if r["method"] == "binary_search" and r.get("raw")]
    if tr:
        out += ["## Appendix — what binary search actually decided", "",
                "Each call asks the judge to pick the upper or lower half. A judge "
                "that answers the same direction every time converges on one end of "
                "the log whatever it contains, which is indistinguishable from "
                "localisation unless the individual choices are recorded.", "",
                "| judge | answer | 'lower' | 'upper' | unreadable | converged to step 0 |",
                "|---|---|---|---|---|---|"]
        for jid, short, _ in JUDGES:
            for gt, glab in ((False, "hidden"), (True, "shown")):
                sel = [r for r in tr if r.get("judge") == jid and r["with_gt"] == gt]
                if not sel:
                    continue
                c = collections.Counter()
                for r in sel:
                    for d_ in str(r["raw"]).split(">"):
                        c["lower" if d_ == "lower" else "upper" if d_ == "upper" else "bad"] += 1
                tot = max(sum(c.values()), 1)
                z = sum(1 for r in sel if r["pred_step"] == 0)
                out.append("| %s | %s | %.0f%% | %.0f%% | %d | %d/%d |"
                           % (short, glab, 100 * c["lower"] / tot, 100 * c["upper"] / tot,
                              c["bad"], z, len(sel)))
        out.append("")

    out += ["## How this was produced", "",
            "| item | value |", "|---|---|",
            "| serving | vLLM (`yllm` env); 70B tensor-parallel over 2 GPUs |",
            "| client | OpenAI SDK (`Jagent` env), `temperature=0` |",
            "| reasoning | disabled via `chat_template_kwargs={'enable_thinking': false}` |",
            "| code | `tools/llm_judge.py`; rebuild this file with `tools/judge_report.py` |",
            "",
            "Parsing accepts both the labelled form the prompt requests "
            "(`Agent Name:` / `Step Number:`) and the numbered form judges often "
            "return instead (`1. <agent>` / `2. <n>`); scoring only the former "
            "would charge a formatting mismatch against the method. Where "
            "step-by-step walks a whole log without ever flagging an error, the "
            "paper leaves the outcome unspecified — it is counted as *no answer* "
            "and scored as a miss rather than defaulted to a step.",
            ]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
