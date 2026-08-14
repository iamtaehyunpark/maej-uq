"""Two-stage attribution on raw P(True): argmin picks the agent, first-crossing
picks the step inside that agent.

Stage 1 takes the agent owning the globally lowest-scoring step -- the rule that
does best at naming the agent. Stage 2 keeps only that agent's steps, in order,
and returns the first one falling below a threshold built from that agent's own
scores; if none crosses, it falls back to that agent's lowest step.

The agent column is identical to plain argmin by construction, so the only
question this asks is whether restricting the search to one agent's trace
improves the step.

Several thresholds are swept because "its own threshold" has no single
definition. They are compared against plain argmin and plain first-crossing on
the same data.

Usage: python tools/ptrue_two_stage.py <ptrue_nogt.jsonl> <ptrue_gt.jsonl>
"""

import json
import random
import re
import statistics as st
import sys


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


def load(path):
    by = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["p_true"] is None:
            continue
        by.setdefault((r["subset"], r["file_id"]), []).append(r)
    out = {}
    for k, sc in by.items():
        sc.sort(key=lambda x: x["step_idx"])
        if len(sc) >= 2:
            out[k] = sc
    return out


#: Threshold built from the selected agent's own scores.
THRESHOLDS = {
    "mean": lambda p: st.fmean(p),
    "median": lambda p: st.median(p),
    "mean-0.5sd": lambda p: st.fmean(p) - 0.5 * (st.pstdev(p) or 0.0),
    "mean-1sd": lambda p: st.fmean(p) - (st.pstdev(p) or 0.0),
}


def two_stage(sc, thr_name):
    """argmin -> that agent -> first crossing within it."""
    lowest = min(sc, key=lambda x: x["p_true"])
    agent = lowest["agent"]
    mine = [x for x in sc if str(x["agent"]).strip() == str(agent).strip()]
    if not mine:
        return lowest["agent"], lowest["step_idx"]
    p = [x["p_true"] for x in mine]
    thr = THRESHOLDS[thr_name](p)
    for x in mine:                       # in step order
        if x["p_true"] < thr:
            return agent, x["step_idx"]
    return agent, min(mine, key=lambda x: x["p_true"])["step_idx"]


def plain_argmin(sc):
    x = min(sc, key=lambda x: x["p_true"])
    return x["agent"], x["step_idx"]


def plain_first_crossing(sc):
    """Whole trajectory, threshold from the whole trajectory."""
    p = [x["p_true"] for x in sc]
    thr = st.fmean(p)
    for x in sc:
        if x["p_true"] < thr:
            return x["agent"], x["step_idx"]
    x = min(sc, key=lambda x: x["p_true"])
    return x["agent"], x["step_idx"]


def score(data, sub, fn):
    S, A = [], []
    for (s, _f), sc in data.items():
        if s != sub:
            continue
        gs, ga = sc[0]["gold_step"], sc[0]["gold_agent"]
        a, st_ = fn(sc)
        S.append(1.0 if st_ == gs else 0.0)
        A.append(1.0 if amatch(ga, a) else 0.0)
    return S, A


def main(argv):
    for path, label in ((argv[1], "answer hidden"), (argv[2], "answer shown")):
        data = load(path)
        print("\n## %s\n" % label)
        print("| rule | corpus | spots the step | spots the agent |")
        print("|---|---|---|---|")
        rules = [("argmin (baseline)", plain_argmin),
                 ("first crossing, whole trajectory", plain_first_crossing)]
        rules += [("argmin -> agent -> first crossing (%s)" % t,
                   (lambda t: (lambda sc: two_stage(sc, t)))(t)) for t in THRESHOLDS]
        for name, fn in rules:
            for sub, slab in (("alg", "algorithm-generated"), ("hc", "hand-crafted")):
                S, A = score(data, sub, fn)
                if S:
                    print("| %s | %s | %s | %s |" % (name, slab, fmt(S), fmt(A)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
