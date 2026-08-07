"""E0 — score-field sanity and threshold stability (spec v2.1 §2). Runs first.

Two questions, both asked before any attribution number exists:

**(a) Is there a field to localize?** Per subset and act type, the raw score
distribution, plus degeneracy checks — constant scores, saturation at the ends,
a near-binary field. A judge that answers 0 or 1 for everything carries no
ranking information, and no attribution rule can rescue that.

**(b) Is the crossing threshold stable?** Normalization statistics and the
crossing threshold are fit by leave-one-file-out CV, so there is one threshold
per fold. Their cross-fold coefficient of variation says whether "the"
threshold is a real quantity or an artefact of which files happened to be in
the training half.

The decision that follows is **pre-registered in ``specs/e0_criteria.json``**
and read before the numbers are computed: if the worst threshold CV exceeds the
registered bound, the primary rule switches from ``first_crossing`` to the
threshold-free set and ``first_crossing`` demotes to an ablation. The decision
and the criterion hash go in the run manifest, so the rule cannot be chosen
after the outcome is visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..normalize.apply import (
    apply_folds,
    field_sanity,
    render_field,
    render_stability,
    worst_threshold_cv,
)
from ..normalize.fit import fit_all_subsets, save_folds, stability
from ..specs import E0_CRITERIA_FILE, e0_criteria, require_status, sha
from ._shared import add_common, emit, flatten, load_records, open_manifest, read_scores

DECISION_FILE = "e0_decision.json"


def build_parser() -> argparse.ArgumentParser:
    p = add_common(argparse.ArgumentParser(prog="masattr e0", description=__doc__))
    p.add_argument("--scores", nargs="+", required=True, help="step-score JSONL(s)")
    p.add_argument("--label-policy", default="prefix", choices=("prefix", "point"))
    p.add_argument("--folds-out", default="runs/normalize/folds.json")
    p.add_argument(
        "--allow-draft-criteria",
        action="store_true",
        help="run against unregistered criteria (exploration only; never a result)",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = open_manifest("e0_field", args)

    criteria = e0_criteria()
    if not args.allow_draft_criteria:
        require_status(
            "e0_criteria",
            criteria,
            "registered",
            "E0's decision rule must be fixed before E0 runs, or the rule can be "
            "chosen after the outcome.",
        )
    else:
        manifest.note(
            "criteria were NOT registered for this run — exploratory only, not an E0 result"
        )

    records = flatten(load_records(args))
    manifest.record_anomalies(records)
    scores: dict = {}
    for path in args.scores:
        scores.update(read_scores(path))
    if not scores:
        raise SystemExit(f"no score rows in {args.scores}")

    # (a) is there a field?
    sanity = field_sanity(records, scores, policy=args.label_policy)

    # (b) is the threshold stable?
    folds = fit_all_subsets(records, scores, policy=args.label_policy)
    apply_folds(scores, folds, strict=False)
    stab = stability(folds)
    worst_cell, worst_cv = worst_threshold_cv(stab)

    bound = float(criteria.get("max_threshold_cv", 0.25))
    max_degenerate = int(criteria.get("max_degenerate_cells", 0))
    threshold_unstable = worst_cv > bound
    field_degenerate = len(sanity["degenerate"]) > max_degenerate

    primary = (
        criteria.get("threshold_free_primary", "relative_crossing")
        if threshold_unstable
        else criteria.get("threshold_primary", "first_crossing")
    )
    decision = {
        "primary_rule": primary,
        "threshold_unstable": threshold_unstable,
        "worst_threshold_cv": worst_cv,
        "worst_threshold_cell": worst_cell,
        "max_threshold_cv": bound,
        "field_degenerate": field_degenerate,
        "n_degenerate_cells": len(sanity["degenerate"]),
        "max_degenerate_cells": max_degenerate,
        "threshold_free_rules": criteria.get(
            "threshold_free_rules", ["argmin", "changepoint", "relative_crossing"]
        ),
        "criterion_hash": sha(json.dumps(criteria, indent=2, sort_keys=True)),
        "criteria_status": criteria.get("status"),
        "label_policy": args.label_policy,
    }

    folds_path = save_folds(folds, args.folds_out)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / DECISION_FILE).write_text(json.dumps(decision, indent=2), encoding="utf-8")
    manifest.note(
        f"E0 decision: primary rule = {primary} "
        f"(worst threshold CV {worst_cv:.3f} vs bound {bound}); "
        f"criterion hash {decision['criterion_hash']}"
    )

    results = {
        "field_sanity": sanity,
        "stability": stab,
        "decision": decision,
        "folds": str(folds_path),
        "n_folds": len(folds),
    }
    verdict = (
        f"**Threshold is unstable** (worst CV {worst_cv:.3f} at {worst_cell} > {bound}) → "
        f"primary rule switches to **{primary}**; first_crossing demotes to an ablation."
        if threshold_unstable
        else f"**Threshold is stable** (worst CV {worst_cv:.3f} at {worst_cell} ≤ {bound}) → "
        f"primary rule stays **{primary}**."
    )
    if field_degenerate:
        verdict += (
            f"\n\n**Field degeneracy exceeds the registered bound** "
            f"({len(sanity['degenerate'])} > {max_degenerate}): "
            + "; ".join(sanity["degenerate"])
            + ". No rule can localize a field that does not vary."
        )

    md = "\n".join(
        [
            "# E0 — score field and threshold stability",
            "",
            f"{len(folds)} leave-one-file-out folds, label policy `{args.label_policy}`, "
            f"criteria `{decision['criterion_hash']}` ({criteria.get('status')})",
            "",
            render_field(sanity),
            "",
            render_stability(stab, bound),
            "",
            "## Pre-registered decision",
            "",
            verdict,
        ]
    )
    emit(manifest, results, md, args.out_dir)
    if field_degenerate:
        return 3  # the field itself is unusable
    return 2 if threshold_unstable else 0


def load_decision(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
