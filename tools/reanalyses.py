"""The four re-analyses bundled into the baseline report.

All are recomputed from artifacts already on disk — no model calls.

(i)   base-rate audit: B1 rows against the reference row's orchestrator/worker
      cells, plus the step-ownership share that drives them.
(ii)  final-step scatter: alg/final scores against normalized gold position.
(iii) type composition of predicted vs gold steps.
(iv)  tolerance curves for the reference row.

Usage: python tools/reanalyses.py <runs-root> <data-root>
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from masattr.baselines.naive import PREDICTORS  # noqa: E402
from masattr.eval.ci import bootstrap_ci  # noqa: E402
from masattr.eval.scorers import exact_agent, exact_step  # noqa: E402
from masattr.judge.score import load_scores  # noqa: E402
from masattr.loaders.whowhen_ag import load as load_ag  # noqa: E402
from masattr.loaders.whowhen_hc import load as load_hc  # noqa: E402
from masattr.typing.normalize import collapse_orchestrator, is_orchestrator  # noqa: E402


def corpus(data_root: Path):
    ag = load_ag(data_root / "who_and_when/Algorithm-Generated.parquet", anomaly_policy="flag")[0]
    hc = load_hc(data_root / "who_and_when/Hand-Crafted.parquet", anomaly_policy="flag")[0]
    return {r.key: r for r in ag + hc}


def audit(meta, runs: Path) -> list[str]:
    out = ["## (i) Base-rate audit", ""]
    hc = [r for r in meta.values() if r.subset == "hc"]
    out += ["| row | fault | n | agent | step |", "|---|---|---|---|---|"]
    for name, fn in PREDICTORS.items():
        for grp in ("orchestrator", "worker"):
            sel = [r for r in hc if is_orchestrator(r.label_mistake_agent) == (grp == "orchestrator")]
            u = [
                (exact_agent(fn(r)[0], r.label_mistake_agent), exact_step(fn(r)[1], r.label_mistake_step))
                for r in sel
            ]
            a = bootstrap_ci(u, lambda x: sum(i for i, _ in x) / len(x), n_boot=2000)
            s = bootstrap_ci(u, lambda x: sum(j for _, j in x) / len(x), n_boot=2000)
            out.append(
                f"| {name} | {grp} | {len(u)} | {a.point:.3f} [{a.lo:.3f}, {a.hi:.3f}] | "
                f"{s.point:.3f} [{s.lo:.3f}, {s.hi:.3f}] |"
            )
    out += ["", "| fault | n | mean share of steps owned by the gold agent |", "|---|---|---|"]
    for grp in ("orchestrator", "worker"):
        sel = [r for r in hc if is_orchestrator(r.label_mistake_agent) == (grp == "orchestrator")]
        shares = [
            sum(1 for s in r.steps if collapse_orchestrator(s.agent) == collapse_orchestrator(r.label_mistake_agent))
            / r.n_steps
            for r in sel
        ]
        out.append(f"| {grp} | {len(sel)} | {st.fmean(shares):.3f} |")
    return out


def final_scatter(meta, runs: Path) -> list[str]:
    """alg/final score vs normalized gold position, binned."""
    out = ["", "## (ii) alg/final score vs normalized gold position", ""]
    path = next((runs / "main/scores").glob("alg__*ptrue*nogt.jsonl"), None)
    if not path:
        return out + ["*reference scores not found*"]
    rows = [r for r in load_scores(path) if r.type_norm == "final"]
    buckets = defaultdict(list)
    for r in rows:
        rec = meta.get(r.key)
        if not rec or rec.n_steps < 2:
            continue
        pos = rec.label_mistake_step / (rec.n_steps - 1)
        buckets[min(int(pos * 5), 4)].append(r.p_raw)
    out += ["| gold position | n | mean final-step score |", "|---|---|---|"]
    for b in sorted(buckets):
        v = buckets[b]
        out.append(f"| [{b/5:.1f}, {(b+1)/5:.1f}) | {len(v)} | {st.fmean(v):.3f} |")
    return out


def type_composition(meta, runs: Path) -> list[str]:
    out = ["", "## (iii) Type composition, predicted vs gold steps", ""]
    for gt in ("nogt", "gt"):
        res_path = runs / f"main/e1_{gt}/results.json"
        if not res_path.exists():
            continue
        res = json.loads(res_path.read_text())
        for label, cfg in sorted(res["configs"].items()):
            subset = [p for p in label.split(" · ") if p.startswith("subset=")][0].split("=")[1]
            preds = cfg["predictions"].get("changepoint_single", {})
            pred_t, gold_t = Counter(), Counter()
            for key, p in preds.items():
                rec = meta.get(key)
                if not rec:
                    continue
                gold_t[rec.steps[rec.label_mistake_step].type_norm] += 1
                if p.get("pred_step") is not None and 0 <= p["pred_step"] < rec.n_steps:
                    pred_t[rec.steps[p["pred_step"]].type_norm] += 1
            types = sorted(set(pred_t) | set(gold_t))
            out.append(f"**GT {gt}, {subset}** — " + ", ".join(
                f"{t}: pred {pred_t[t]} / gold {gold_t[t]}" for t in types
            ))
    return out


def tolerance_curve(runs: Path) -> list[str]:
    out = ["", "## (iv) Tolerance curves, reference row (primary rule)", "",
           "| GT | subset | exact | |Δ|≤1 | |Δ|≤2 |", "|---|---|---|---|---|"]
    for gt in ("nogt", "gt"):
        res_path = runs / f"main/e1_{gt}/results.json"
        if not res_path.exists():
            continue
        res = json.loads(res_path.read_text())
        for label, cfg in sorted(res["configs"].items()):
            subset = [p for p in label.split(" · ") if p.startswith("subset=")][0].split("=")[1]
            sc = cfg["scores"].get("changepoint_single", {})
            vals = [sc.get(f"{k}/all", {}).get("step_acc") for k in ("exact", "tol1", "tol2")]
            out.append(
                f"| {gt} | {subset} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in vals) + " |"
            )
    return out


def main(argv: list[str]) -> int:
    runs = Path(argv[1] if len(argv) > 1 else "runs")
    data = Path(argv[2] if len(argv) > 2 else "data")
    meta = corpus(data)
    lines = ["# Re-analyses", ""]
    lines += audit(meta, runs)
    lines += final_scatter(meta, runs)
    lines += type_composition(meta, runs)
    lines += tolerance_curve(runs)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
