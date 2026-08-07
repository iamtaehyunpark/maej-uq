"""E0 — score-field sanity. Runs first, and decides nothing.

The primary attribution rule is fixed by ``specs/rule_directive.md``, so E0 no
longer gates E1: it reports, and a human reads it. Three questions, all asked
before any attribution number exists:

**(a) Is there a field to localize?** Per subset and act type, the raw score
distribution, plus degeneracy checks — constant scores, saturation at the ends,
a near-binary field. A judge that answers 0 or 1 for everything carries no
ranking information, and no attribution rule can rescue that.

**(b) What do the per-type distributions look like?** Level and spread per act
type, which is what typed normalization exists to remove and what E4 ablates.

**(c) How stable is the leave-one-out threshold?** Normalization statistics and
a crossing threshold are fit per fold, so there is one threshold per held-out
file. Their cross-fold coefficient of variation is reported because
``first_crossing`` is an **E3 ablation row** that rests on that threshold — the
stability number says how much weight that row can carry. The primary rule takes
nothing from it.

E0 exits non-zero only when the field itself is degenerate beyond the registered
bound: a constant or saturated field cannot be localized by any rule, and that
is worth stopping for.
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
from ..specs import criteria as load_criteria
from ._shared import add_common, emit, flatten, load_records, open_manifest, read_scores


def build_parser() -> argparse.ArgumentParser:
    p = add_common(argparse.ArgumentParser(prog="masattr e0", description=__doc__))
    p.add_argument("--scores", nargs="+", required=True, help="step-score JSONL(s)")
    p.add_argument("--label-policy", default="prefix", choices=("prefix", "point"))
    p.add_argument("--folds-out", default="runs/normalize/folds.json")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = open_manifest("e0_field", args)

    criteria = load_criteria()

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

    max_degenerate = int(criteria.get("max_degenerate_cells", 0))
    field_degenerate = len(sanity["degenerate"]) > max_degenerate

    folds_path = save_folds(folds, args.folds_out)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    results = {
        "field_sanity": sanity,
        "stability": stab,
        "worst_threshold_cv": worst_cv,
        "worst_threshold_cell": worst_cell,
        "field_degenerate": field_degenerate,
        "n_degenerate_cells": len(sanity["degenerate"]),
        "max_degenerate_cells": max_degenerate,
        "folds": str(folds_path),
        "n_folds": len(folds),
        "label_policy": args.label_policy,
        "decides_nothing": (
            "the primary rule is fixed by specs/rule_directive.md; this run reports"
        ),
    }
    manifest.note(
        f"E0 is sanity-only: {len(sanity['degenerate'])} degenerate cells, worst "
        f"leave-one-out threshold CV {worst_cv:.3f} at {worst_cell}"
    )

    notes = [
        f"Worst cross-fold threshold CV: **{worst_cv:.3f}** at `{worst_cell}`. "
        "This bears on the `first_crossing` **ablation row** only — the primary "
        "rule reads no threshold.",
    ]
    if field_degenerate:
        notes.append(
            f"**Field degeneracy exceeds the registered bound** "
            f"({len(sanity['degenerate'])} > {max_degenerate}): "
            + "; ".join(sanity["degenerate"])
            + ". No rule can localize a field that does not vary."
        )
    else:
        notes.append("Field is usable: no cell is constant, saturated, or near-binary.")

    md = "\n".join(
        [
            "# E0 — score-field sanity",
            "",
            f"{len(folds)} leave-one-file-out folds, label policy `{args.label_policy}`. "
            "E0 decides nothing: the primary rule is fixed by "
            "`specs/rule_directive.md`.",
            "",
            render_field(sanity),
            "",
            render_stability(stab),
            "",
            "## Read-out",
            "",
            "\n\n".join(notes),
        ]
    )
    emit(manifest, results, md, args.out_dir)
    return 3 if field_degenerate else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
