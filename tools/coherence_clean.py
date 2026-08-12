"""Judge-free coherence baselines, recomputed. Standalone, no harness import.

The pilot's versions of these two fields were not measuring what they claimed.
Two defects, both fixed here:

1. **Step 0 was scored 0.5 as a "neutral" value.** It has no prefix to compare
   against, so no score is meaningful — but 0.5 sits near the *bottom* of both
   fields' real distributions (only 1.0% of algorithm-generated embedding scores
   fall below it), which made step 0 the most suspicious step in almost every
   trajectory. The selection rule then picked it in 92% of files, turning the
   whole field into a constant "blame the first step" predictor. Here step 0 is
   excluded from the candidate set outright rather than given a value.

2. **The NLI premise was truncated from the wrong end.** Five prefix steps come
   to ~2,000 tokens against a 512-token limit, and the default `longest_first`
   strategy trims the tail — so only the *oldest* step survived and steps t-4..t-1,
   including the one immediately before, were discarded. `truncation_side="left"`
   keeps the recent context instead.

Scoring is top-1 and top-2 only: rank candidate steps by score ascending and
check whether the labelled step, and the labelled agent, are in the top k. No
normalization, no attribution rules.

Usage:
  python tools/coherence_clean.py score <data-root> <out.jsonl>
  python tools/coherence_clean.py report <out.jsonl>
"""

import json
import random
import statistics as st
import sys
import time

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
NLI_PREMISE_STEPS = 5
HYP_MAX = 256          # tokens reserved for the step being judged
MAX_CHARS = 2000


def load(data_root, name, subset):
    import pandas as pd

    df = pd.read_parquet("%s/who_and_when/%s.parquet" % (data_root, name))
    out = []
    for _, r in df.iterrows():
        steps = []
        for s in list(r["history"]):
            agent = (s.get("name") or s.get("role") or "") if subset == "alg" else (s.get("role") or "")
            steps.append({"agent": str(agent).strip(), "content": str(s.get("content") or "")})
        out.append({
            "subset": subset, "file_id": str(r["question_ID"]), "steps": steps,
            "gold_step": int(r["mistake_step"]), "gold_agent": str(r["mistake_agent"]).strip(),
        })
    return out


class Embedder:
    def __init__(self, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(EMBED_MODEL).to(device).eval()
        self.device = device

    def encode(self, texts):
        torch = self.torch
        b = self.tok(list(texts), padding=True, truncation=True, max_length=512,
                     return_tensors="pt").to(self.device)
        with torch.no_grad():
            h = self.model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(e, dim=-1)


class NLI:
    def __init__(self, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        # Keep the END of the premise: the steps nearest the one being judged.
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL, truncation_side="left")
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device).eval()
        self.device = device
        labels = getattr(self.model.config, "id2label", {}) or {}
        self.idx = next((i for i, n in labels.items()
                         if str(n).lower().startswith("contradiction")), 0)

    def contradiction(self, premise, hypothesis):
        """Budget the two sides explicitly.

        ``only_first`` cannot always satisfy the limit — a long hypothesis
        leaves the premise nothing to give — so the split is done by hand: the
        hypothesis keeps its head up to ``HYP_MAX`` tokens, and the premise
        takes the remainder **from its tail**, which is the context nearest the
        step being judged.
        """
        torch = self.torch
        hyp_ids = self.tok(hypothesis, add_special_tokens=False)["input_ids"][:HYP_MAX]
        budget = 512 - len(hyp_ids) - 4  # room for [CLS]/[SEP] markers
        prem_ids = self.tok(premise, add_special_tokens=False)["input_ids"][-budget:]
        b = self.tok(
            self.tok.decode(prem_ids), self.tok.decode(hyp_ids),
            truncation=True, max_length=512, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**b).logits[0]
        return float(torch.softmax(logits, dim=-1)[self.idx])


def cmd_score(argv):
    import torch

    data_root, out_path = argv[0], argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recs = load(data_root, "Algorithm-Generated", "alg") + load(data_root, "Hand-Crafted", "hc")
    emb, nli = Embedder(dev), NLI(dev)

    n = 0
    t0 = time.time()
    with open(out_path, "w") as fh:
        for k, rec in enumerate(recs):
            steps = rec["steps"]
            E = emb.encode([(s["content"] or "")[:MAX_CHARS] for s in steps])
            # Step 0 has no prefix and is not a candidate. It is emitted with
            # null scores so the row count still matches the corpus, and skipped
            # at selection time.
            fh.write(json.dumps({
                "subset": rec["subset"], "file_id": rec["file_id"], "step_idx": 0,
                "agent": steps[0]["agent"], "embed_divergence": None,
                "nli_contradiction": None, "candidate": False,
                "gold_step": rec["gold_step"], "gold_agent": rec["gold_agent"],
                "n_steps": len(steps),
            }) + "\n")
            n += 1
            for t in range(1, len(steps)):
                cent = torch.nn.functional.normalize(E[:t].mean(0, keepdim=True), dim=-1)
                cos = float((E[t:t + 1] * cent).sum())
                p_emb = (1.0 + cos) / 2.0          # high = coheres with prefix
                prefix = steps[max(0, t - NLI_PREMISE_STEPS):t]
                premise = "\n".join((s["content"] or "")[:MAX_CHARS] for s in prefix)
                p_nli = 1.0 - nli.contradiction(premise, (steps[t]["content"] or "")[:MAX_CHARS])
                fh.write(json.dumps({
                    "subset": rec["subset"], "file_id": rec["file_id"], "step_idx": t,
                    "agent": steps[t]["agent"], "embed_divergence": p_emb,
                    "nli_contradiction": p_nli, "candidate": True,
                    "gold_step": rec["gold_step"], "gold_agent": rec["gold_agent"],
                    "n_steps": len(steps),
                }) + "\n")
                n += 1
            fh.flush()
            if (k + 1) % 20 == 0:
                el = time.time() - t0
                sys.stderr.write("  %d/%d files, %d steps, %.0fs\n" % (k + 1, len(recs), n, el))
                sys.stderr.flush()
    sys.stderr.write("wrote %d rows to %s\n" % (n, out_path))
    return 0


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
    a, b = str(a).strip().lower(), str(b).strip().lower()
    return bool(a and b and (a in b or b in a))


def cmd_report(argv):
    rows = [json.loads(l) for l in open(argv[0]) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault((r["subset"], r["file_id"]), []).append(r)

    print("rows %d   trajectories %d" % (len(rows), len(by)))
    excl = sum(1 for r in rows if not r["candidate"])
    print("step-0 rows excluded from selection: %d\n" % excl)

    for field in ("embed_divergence", "nli_contradiction"):
        print("### %s\n" % field)
        print("| corpus | files | step top-1 | step top-2 | agent top-1 | agent top-2 |")
        print("|---|---|---|---|---|---|")
        for sub, lab in (("alg", "algorithm-generated"), ("hc", "hand-crafted")):
            s1, s2, a1, a2, z = [], [], [], [], 0
            for (su, _f), sc in by.items():
                if su != sub:
                    continue
                cand = [x for x in sc if x["candidate"] and x[field] is not None]
                if len(cand) < 2:
                    continue
                gs, ga = cand[0]["gold_step"], cand[0]["gold_agent"]
                # A file whose labelled step is 0 (or out of range) stays in the
                # denominator as a miss on the step column. Dropping it would
                # quietly grade this field on an easier corpus than P(True),
                # which can select step 0.
                selectable = any(x["step_idx"] == gs for x in cand)
                ranked = sorted(cand, key=lambda x: x[field])
                s1.append(1.0 if selectable and ranked[0]["step_idx"] == gs else 0.0)
                s2.append(1.0 if selectable and gs in [x["step_idx"] for x in ranked[:2]] else 0.0)
                a1.append(1.0 if amatch(ga, ranked[0]["agent"]) else 0.0)
                a2.append(1.0 if any(amatch(ga, x["agent"]) for x in ranked[:2]) else 0.0)
                z += 0 if selectable else 1
            if not s1:
                continue
            # Matched controls on the identical candidate set (step 0 excluded,
            # unselectable-gold files kept as misses), so these are comparable
            # to the rows above rather than to a differently-filtered corpus.
            r1, r2, ra1, ra2 = [], [], [], []
            for (su, _f), sc in by.items():
                if su != sub:
                    continue
                cand = [x for x in sc if x["candidate"] and x[field] is not None]
                if len(cand) < 2:
                    continue
                gs, ga = cand[0]["gold_step"], cand[0]["gold_agent"]
                sel = any(x["step_idx"] == gs for x in cand)
                nc = len(cand)
                r1.append((1.0 / nc) if sel else 0.0)
                r2.append((min(2, nc) / nc) if sel else 0.0)
                m = sum(1 for x in cand if amatch(ga, x["agent"]))
                ra1.append(m / nc)
                miss = 1.0
                for i in range(min(2, nc)):
                    av = nc - m - i
                    miss = 0.0 if av < 0 else miss * av / (nc - i)
                ra2.append(1.0 - miss)
            print("| %s | %d | %s | %s | %s | %s |"
                  % (lab, len(s1), fmt(s1), fmt(s2), fmt(a1), fmt(a2)))
            print("| *random baseline, same candidates* | %d | %s | %s | %s | %s |"
                  % (len(r1), fmt(r1), fmt(r2), fmt(ra1), fmt(ra2)))
            if z:
                print("| *(%d files have a labelled step this field cannot "
                      "select; kept as misses)* | | | | | |" % z)
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
