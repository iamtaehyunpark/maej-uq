"""Derive lookahead-shift (delta) score fields and test them against gold.

Two jobs, deliberately in one command because the second is meaningless without
the first's provenance:

1. ``delta[t] = p_arm[t] - p_base[t]`` written as ordinary score JSONLs, which
   then feed ``e0`` (LOO folds + field sanity) and ``e1`` (the rule table)
   unchanged. The delta rows are what land in the master table.

2. The correlation readout. "Does the shift spot the fault?" is two questions at
   two different levels and they can disagree, so both are reported:

   * **step level** — AUROC of ``-delta`` against *is this the gold step*. This
     is the field-quality number: can the shift rank the faulty step above the
     rest of its own trajectory? Computed within-file (each trajectory
     contributes its own ranking) and pooled.
   * **file level** — does the shift at the *selected* step predict whether that
     selection was **correct**? This is the one that makes delta a confidence
     signal rather than merely another field: if attributions that came with a
     large negative shift are right more often than attributions that came with
     a flat one, delta buys selective prediction even where it loses on
     accuracy. Reported as mean delta split by correct/incorrect, the AUROC
     between those two groups, and a risk-coverage curve over delta magnitude.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from ..attribute.rules import PRIMARY, attribute, resolve_method
from ..eval.ci import bootstrap_ci
from ..eval.scorers import exact_agent, exact_step
from ..judge.delta import derive_delta
from ..judge.score import by_file, load_scores
from ..normalize.apply import apply_folds, thresholds_for
from ..normalize.fit import load_folds
from ._shared import add_common, emit, flatten, load_records, open_manifest


def build_parser() -> argparse.ArgumentParser:
    p = add_common(argparse.ArgumentParser(prog="masattr delta", description=__doc__))
    p.add_argument("--base", nargs="+", required=True, help="W0 (C3) score JSONL(s)")
    p.add_argument(
        "--arm",
        nargs="+",
        required=True,
        help="lookahead score JSONL(s) — paired to --base by subset and GT setting",
    )
    p.add_argument("--delta-dir", default="runs/delta/fields", help="where delta JSONLs are written")
    p.add_argument(
        "--folds",
        help="folds fit on the DELTA field (run e0 on the emitted files first). "
        "Without it only the raw-score correlations are reported, not the rules.",
    )
    p.add_argument("--method", default=PRIMARY, help="rule used for the file-level readout")
    return p


def _auroc(pos: list[float], neg: list[float]) -> float | None:
    """Rank AUROC of ``pos`` over ``neg``; ties count a half."""
    if not pos or not neg:
        return None
    wins = sum(
        1.0 if a > b else 0.5 if a == b else 0.0
        for a in pos
        for b in neg
    )
    return wins / (len(pos) * len(neg))


def step_level(rows, gold_step: dict[str, int]) -> dict:
    """AUROC of -delta against is-gold-step, within-file and pooled."""
    per_file, pooled_pos, pooled_neg = [], [], []
    for key, scores in by_file(rows).items():
        g = gold_step.get(key)
        if g is None or not 0 <= g < len(scores):
            continue
        # -delta: a bigger drop when the lookahead is appended should rank higher
        pos = [-s.p_raw for s in scores if s.step_idx == g and s.parse_ok]
        neg = [-s.p_raw for s in scores if s.step_idx != g and s.parse_ok]
        a = _auroc(pos, neg)
        if a is not None:
            per_file.append(a)
        pooled_pos += pos
        pooled_neg += neg
    ci = bootstrap_ci(per_file, lambda x: st.fmean(x), n_boot=2000) if per_file else None
    return {
        "n_files": len(per_file),
        "within_file_auroc": st.fmean(per_file) if per_file else None,
        "within_file_ci": {"lo": ci.lo, "hi": ci.hi} if ci else None,
        "pooled_auroc": _auroc(pooled_pos, pooled_neg),
        "mean_delta_gold": st.fmean(pooled_pos) * -1 if pooled_pos else None,
        "mean_delta_other": st.fmean(pooled_neg) * -1 if pooled_neg else None,
    }


def file_level(grouped: dict, records, method: str, per_file_thr: dict) -> dict:
    """Does the shift at the selected step predict that selection being right?"""
    by_key = {r.key: r for r in records}
    preds = attribute(grouped, method=method, per_file=per_file_thr)
    sel_correct, sel_wrong, coverage = [], [], []
    for key, pred in preds.items():
        rec = by_key.get(key)
        scores = grouped.get(key)
        if rec is None or not scores or pred.step is None:
            continue
        # the delta the rule's own choice sat on — raw, not z-scored, so the
        # magnitudes stay comparable across files
        d = next((s.p_raw for s in scores if s.step_idx == pred.step), None)
        if d is None:
            continue
        ok = exact_step(pred.step, rec.label_mistake_step) and exact_agent(
            pred.agent, rec.label_mistake_agent
        )
        (sel_correct if ok else sel_wrong).append(d)
        coverage.append((d, ok))
    if not coverage:
        return {"n_selected": 0, "n_correct": 0, "risk_coverage": []}

    # risk-coverage: keep the most-negative-delta fraction, measure accuracy there
    coverage.sort(key=lambda t: t[0])
    curve = []
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        k = max(1, int(round(frac * len(coverage))))
        kept = coverage[:k]
        curve.append(
            {
                "coverage": round(k / len(coverage), 3) if coverage else None,
                "n": k,
                "accuracy": round(sum(1 for _, ok in kept if ok) / k, 4),
            }
        )
    return {
        "n_selected": len(coverage),
        "n_correct": len(sel_correct),
        "mean_delta_when_correct": st.fmean(sel_correct) if sel_correct else None,
        "mean_delta_when_wrong": st.fmean(sel_wrong) if sel_wrong else None,
        # -delta again: the hypothesis is that correct picks sit on bigger drops
        "auroc_correct_vs_wrong": _auroc(
            [-d for d in sel_correct], [-d for d in sel_wrong]
        ),
        "risk_coverage": curve,
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = open_manifest("delta_field", args)
    records = flatten(load_records(args))
    gold_step = {r.key: r.label_mistake_step for r in records}

    base_rows = [r for p in args.base for r in load_scores(p)]
    arm_rows = [r for p in args.arm for r in load_scores(p)]

    results: dict = {"fields": {}}
    out_dir = Path(args.delta_dir)
    folds = load_folds(args.folds) if args.folds else None
    per_file_thr = thresholds_for(folds) if folds else {}

    # one delta field per (subset, GT setting, arm) present in the inputs
    groups: dict[tuple, tuple[list, list]] = {}
    for r in base_rows:
        groups.setdefault((r.subset, r.with_gt), ([], []))[0].append(r)
    for r in arm_rows:
        g = groups.get((r.subset, r.with_gt))
        if g is None:
            raise SystemExit(
                f"lookahead rows for ({r.subset}, gt={r.with_gt}) have no matching "
                "W0 base rows; pass both arms of the same cell"
            )
        g[1].append(r)

    for (subset, with_gt), (b, a) in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if not a:
            raise SystemExit(f"no lookahead rows for ({subset}, gt={with_gt})")
        rows = derive_delta(b, a)
        arm_name = rows[0].lookahead
        tag = f"{subset}__delta_{arm_name}_{'gt' if with_gt else 'nogt'}"
        path = out_dir / f"{tag}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r.to_dict()) + "\n")

        entry = {
            "path": str(path),
            "n_rows": len(rows),
            "arm": arm_name,
            "subset": subset,
            "with_gt": with_gt,
            "step_level": step_level(rows, gold_step),
        }
        if folds:
            grouped = by_file(rows)
            apply_folds(grouped, folds)  # writes p_norm in place
            entry["file_level"] = file_level(
                grouped, [r for r in records if r.subset == subset], args.method, per_file_thr
            )
        results["fields"][tag] = entry
        manifest.note(f"{tag}: {len(rows)} delta rows from arm {arm_name!r}")

    emit(manifest, results, _render(results), args.out_dir)
    return 0


def _render(results: dict) -> str:
    lines = [
        "# Lookahead-shift (delta) fields",
        "",
        "`delta[t] = p_arm[t] - p_base[t]`. Negative = the step lost credibility "
        "once what followed it was appended.",
        "",
        "## Step level — can the shift rank the gold step within its own trajectory?",
        "",
        "| field | n files | within-file AUROC | pooled AUROC | mean δ gold | mean δ other |",
        "|---|---|---|---|---|---|",
    ]

    def f(x, p=3):
        return "—" if x is None else f"{x:.{p}f}"

    for tag, e in sorted(results["fields"].items()):
        s = e["step_level"]
        ci = s["within_file_ci"]
        w = f(s["within_file_auroc"]) + (f" [{ci['lo']:.3f},{ci['hi']:.3f}]" if ci else "")
        lines.append(
            f"| {tag} | {s['n_files']} | {w} | {f(s['pooled_auroc'])} | "
            f"{f(s['mean_delta_gold'], 4)} | {f(s['mean_delta_other'], 4)} |"
        )

    if any("file_level" in e for e in results["fields"].values()):
        lines += [
            "",
            "## File level — does the shift at the selected step predict a correct selection?",
            "",
            "| field | n | correct | mean δ correct | mean δ wrong | AUROC |",
            "|---|---|---|---|---|---|",
        ]
        for tag, e in sorted(results["fields"].items()):
            g = e.get("file_level")
            if not g:
                continue
            lines.append(
                f"| {tag} | {g['n_selected']} | {g['n_correct']} | "
                f"{f(g['mean_delta_when_correct'], 4)} | {f(g['mean_delta_when_wrong'], 4)} | "
                f"{f(g['auroc_correct_vs_wrong'])} |"
            )
        lines += ["", "### Risk-coverage (keep the most-negative δ fraction)", ""]
        for tag, e in sorted(results["fields"].items()):
            g = e.get("file_level")
            if not g:
                continue
            pts = ", ".join(f"{c['coverage']:.1f}→{c['accuracy']:.3f}" for c in g["risk_coverage"])
            lines.append(f"- **{tag}** — {pts}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
