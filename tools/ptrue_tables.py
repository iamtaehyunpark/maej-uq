"""Emit the top-1 and top-2 tables, P(True) against content-free baselines."""
import json, sys, statistics as st, random

def boot(v, n=2000, seed=0):
    rng = random.Random(seed); m=[]
    for _ in range(n):
        s=[v[rng.randrange(len(v))] for _ in range(len(v))]
        m.append(sum(s)/len(s))
    m.sort(); return m[int(.025*n)], m[int(.975*n)]

def fmt(v):
    lo,hi = boot(v); return "%.3f [%.3f,%.3f]" % (st.fmean(v), lo, hi)

def load(path):
    rows=[json.loads(l) for l in open(path) if l.strip()]
    by={}
    for r in rows: by.setdefault((r["subset"], r["file_id"]),[]).append(r)
    out={}
    for k,sc in by.items():
        sc=[x for x in sc if x["p_true"] is not None]
        if len(sc)<2: continue
        gs=sc[0]["gold_step"]
        if not any(x["step_idx"]==gs for x in sc): continue
        out[k]=sorted(sc,key=lambda x:x["step_idx"])
    return out

def amatch(a,b):
    a=str(a).strip().lower(); b=str(b).strip().lower()
    return bool(a and b and (a in b or b in a))

def hits(files, subset, pick, k):
    """pick(steps,k) -> list of chosen step dicts"""
    S,A=[],[]
    for (sub,_),sc in files.items():
        if sub!=subset: continue
        gs, ga = sc[0]["gold_step"], sc[0]["gold_agent"]
        ch = pick(sc,k)
        S.append(1.0 if gs in [x["step_idx"] for x in ch] else 0.0)
        A.append(1.0 if any(amatch(ga,x["agent"]) for x in ch) else 0.0)
    return S,A

def rand_exp(files, subset, k):
    S,A=[],[]
    for (sub,_),sc in files.items():
        if sub!=subset: continue
        n=len(sc); gs,ga=sc[0]["gold_step"],sc[0]["gold_agent"]
        S.append(min(k,n)/n)
        m=sum(1 for x in sc if amatch(ga,x["agent"]))
        miss=1.0
        for i in range(min(k,n)):
            av=n-m-i
            miss = 0.0 if av<0 else miss*av/(n-i)
        A.append(1.0-miss)
    return S,A

nogt, gt = load(sys.argv[1]), load(sys.argv[2])
by_p = lambda sc,k: sorted(sc,key=lambda x:x["p_true"])[:k]
first = lambda sc,k: sc[:k]
last  = lambda sc,k: sc[-k:]

def busiest(sc,k):
    from collections import Counter
    c=Counter(str(x["agent"]).strip().lower() for x in sc)
    top=[a for a,_ in c.most_common(k)]
    return [next(x for x in sc if str(x["agent"]).strip().lower()==a) for a in top]

for k in (1,2):
    print("\n### Top-%d\n" % k)
    print("| method | corpus | names the step | names the agent |")
    print("|---|---|---|---|")
    for sub,label in (("alg","algorithm-generated"),("hc","hand-crafted")):
        for name,files in (("P(True), answer hidden",nogt),("P(True), answer shown",gt)):
            S,A=hits(files,sub,by_p,k)
            print("| **%s** | %s | **%s** | **%s** |" % (name,label,fmt(S),fmt(A)))
        for name,fn in (("baseline: first step(s)",first),("baseline: last step(s)",last),
                        ("baseline: busiest agent(s)",busiest)):
            S,A=hits(nogt,sub,fn,k)
            print("| %s | %s | %s | %s |" % (name,label,fmt(S),fmt(A)))
        S,A=rand_exp(nogt,sub,k)
        print("| baseline: random step(s) | %s | %s | %s |" % (label,fmt(S),fmt(A)))
