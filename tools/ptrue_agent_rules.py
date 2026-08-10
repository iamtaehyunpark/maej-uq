"""Which way of aggregating P(True) per agent names the faulty agent best?

`agent_first` selects the agent whose *best* step is still worst (minimax), then
picks a step inside that agent. That is one choice among many, and it was never
compared against the others. Each selector below scores every agent from its own
steps' P(True) and takes the worst; the step reported is that agent's
lowest-scoring step.
"""
import json, sys, statistics as st, random
from collections import defaultdict

def boot(v, n=2000, seed=0):
    rng = random.Random(seed); m = []
    for _ in range(n):
        s = [v[rng.randrange(len(v))] for _ in range(len(v))]
        m.append(sum(s) / len(s))
    m.sort(); return m[int(.025*n)], m[int(.975*n)]

def fmt(v):
    lo, hi = boot(v); return "%.3f [%.3f,%.3f]" % (st.fmean(v), lo, hi)

def amatch(a, b):
    a = str(a).strip().lower(); b = str(b).strip().lower()
    return bool(a and b and (a in b or b in a))

SELECTORS = {
    "worst single step (min)":      lambda ps: min(ps),
    "best step still worst (max)":  lambda ps: max(ps),
    "mean":                         lambda ps: st.fmean(ps),
    "median":                       lambda ps: st.median(ps),
    "mean of its 2 worst steps":    lambda ps: st.fmean(sorted(ps)[:2]),
    "fraction of steps below 0.5":  lambda ps: -sum(1 for p in ps if p < 0.5) / len(ps),
    "count of steps below 0.5":     lambda ps: -sum(1 for p in ps if p < 0.5),
    "total suspicion, sum(1-p)":    lambda ps: -sum(1 - p for p in ps),
}

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault((r["subset"], r["file_id"]), []).append(r)
    out = {}
    for k, sc in by.items():
        sc = [x for x in sc if x["p_true"] is not None]
        if len(sc) < 2: continue
        gs = sc[0]["gold_step"]
        if not any(x["step_idx"] == gs for x in sc): continue
        out[k] = sorted(sc, key=lambda x: x["step_idx"])
    return out

def run(files, subset, sel):
    A, S = [], []
    for (sub, _), sc in files.items():
        if sub != subset: continue
        gs, ga = sc[0]["gold_step"], sc[0]["gold_agent"]
        per = defaultdict(list)
        for x in sc: per[str(x["agent"]).strip()].append(x)
        score = {a: sel([y["p_true"] for y in xs]) for a, xs in per.items()}
        worst = min(score, key=lambda a: score[a])
        A.append(1.0 if amatch(ga, worst) else 0.0)
        pick = min(per[worst], key=lambda y: y["p_true"])
        S.append(1.0 if pick["step_idx"] == gs else 0.0)
    return A, S

nogt, gt = load(sys.argv[1]), load(sys.argv[2])
for sub, lab in (("alg", "algorithm-generated"), ("hc", "hand-crafted")):
    print("\n**%s**\n" % lab)
    print("| agent selector | agent, hidden | step, hidden | agent, shown | step, shown |")
    print("|---|---|---|---|---|")
    for name, sel in SELECTORS.items():
        a1, s1 = run(nogt, sub, sel)
        a2, s2 = run(gt, sub, sel)
        print("| %s | %s | %s | %s | %s |" % (name, fmt(a1), fmt(s1), fmt(a2), fmt(s2)))
