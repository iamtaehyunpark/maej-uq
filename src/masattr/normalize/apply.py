"""Apply fold statistics to step scores (spec v2.1 §2).

Each file's steps are z-scored under the statistics of *its own* fold — fit on
every other file — and the result is written to ``p_norm``. The raw score stays
on the row untouched: downstream code, and anyone reading the JSONL, can see
both, and no stage has to trust that normalization did what it says.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..record import Record
from .fit import FoldStats, NormalizationError, step_labels

#: Fraction of a type's scores sitting at the extremes before the field is
#: called saturated — a judge answering 0 or 1 for everything carries no ranking
#: information, whatever its accuracy looks like.
SATURATION_EPS = 1e-3
SATURATION_FRACTION = 0.95

#: Below this spread a type's raw scores are effectively constant.
DEGENERATE_SD = 1e-3


def apply_folds(
    scores_by_file: Mapping[str, list],
    folds: Mapping[str, FoldStats],
    *,
    typed: bool = True,
    strict: bool = True,
) -> int:
    """Write ``p_norm`` onto every score row, in place. Returns rows normalized."""
    n = 0
    for key, rows in scores_by_file.items():
        fold = folds.get(key)
        if fold is None:
            if strict:
                raise NormalizationError(
                    f"{key}: no fold statistics. Every file must be normalized under "
                    "a fold fit without it; scoring a file under statistics that saw "
                    "it is the leak leave-one-out exists to prevent."
                )
            continue
        for row in rows:
            row.p_norm = fold.z(row.p_raw, row.type_norm, typed=typed)
            n += 1
    return n


def thresholds_for(
    folds: Mapping[str, FoldStats], *, typed: bool = True
) -> dict[str, dict[str, float] | float]:
    """Per-file crossing thresholds, in the same z-space as ``p_norm``."""
    out: dict[str, dict[str, float] | float] = {}
    for key, fold in folds.items():
        if typed and fold.thresholds:
            out[key] = {**fold.thresholds, "": fold.threshold}
        else:
            out[key] = fold.threshold
    return out


# --- score-field sanity (E0a) ----------------------------------------------


def field_sanity(
    records: Sequence[Record],
    scores_by_file: Mapping[str, list],
    *,
    policy: str = "prefix",
) -> dict[str, Any]:
    """Distributions of the raw field per subset and type, with degeneracy checks.

    A field that is constant, or saturated at the ends, cannot be localized no
    matter which rule reads it — so this runs before any attribution number and
    says plainly whether there is a signal to localize.
    """
    by_key = {r.key: r for r in records}
    buckets: dict[tuple[str, str], list[float]] = {}
    labelled: dict[tuple[str, str], list[tuple[float, bool]]] = {}

    for key, rows in scores_by_file.items():
        rec = by_key.get(key)
        if rec is None:
            continue
        labels = step_labels(rec, policy)
        for row in rows:
            buckets.setdefault((rec.subset, row.type_norm), []).append(row.p_raw)
            if row.step_idx in labels:
                labelled.setdefault((rec.subset, row.type_norm), []).append(
                    (row.p_raw, labels[row.step_idx])
                )

    out: dict[str, Any] = {"policy": policy, "cells": {}, "degenerate": []}
    for (subset, type_norm), values in sorted(buckets.items()):
        arr = np.asarray(values, dtype=float)
        sat = float(
            ((arr <= SATURATION_EPS) | (arr >= 1 - SATURATION_EPS)).mean()
        )
        cell = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "p05": float(np.percentile(arr, 5)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
            "saturated_fraction": sat,
            "n_distinct": int(np.unique(np.round(arr, 6)).size),
        }
        pairs = labelled.get((subset, type_norm), [])
        if pairs:
            from ..eval.ci import auroc

            cell["n_labelled"] = len(pairs)
            cell["auroc_vs_derived_labels"] = auroc(
                [p for p, _ in pairs], [y for _, y in pairs]
            )
        problems = []
        if cell["sd"] < DEGENERATE_SD:
            problems.append("constant")
        if sat >= SATURATION_FRACTION:
            problems.append("saturated")
        if cell["n_distinct"] <= 2 and cell["n"] > 10:
            problems.append("near-binary")
        if problems:
            cell["degenerate"] = problems
            out["degenerate"].append(f"{subset}/{type_norm}: {', '.join(problems)}")
        out["cells"][f"{subset}/{type_norm}"] = cell
    return out


def render_field(sanity: Mapping[str, Any]) -> str:
    lines = [
        "### Score field, per subset and type",
        "",
        "| cell | n | mean | sd | p05 | median | p95 | saturated | distinct | AUROC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, c in sanity["cells"].items():
        auc = c.get("auroc_vs_derived_labels")
        lines.append(
            f"| {name} | {c['n']} | {c['mean']:.3f} | {c['sd']:.3f} | {c['p05']:.3f} | "
            f"{c['median']:.3f} | {c['p95']:.3f} | {c['saturated_fraction']:.1%} | "
            f"{c['n_distinct']} | {'—' if auc is None or auc != auc else f'{auc:.3f}'} |"
        )
    if sanity["degenerate"]:
        lines += ["", "**Degenerate cells:** " + "; ".join(sanity["degenerate"])]
    else:
        lines += ["", "No degenerate cells: every cell varies and none is saturated."]
    lines += [
        "",
        "> AUROC here is against *derived* step labels, not annotations — it says "
        "whether the field carries any within-type signal at all, not how well "
        "attribution will do.",
    ]
    return "\n".join(lines)


def render_stability(stab: Mapping[str, Any], bound: float | None = None) -> str:
    lines = ["### Cross-fold stability of the normalization", ""]
    for subset, row in stab.items():
        g = row["global_threshold"]
        flag = ""
        if bound is not None and g["cv"] > bound:
            flag = f"  ← exceeds the registered bound {bound}"
        lines += [
            f"**{subset}** ({row['n_folds']} folds)",
            "",
            f"global threshold: mean={g['mean']:+.3f} sd={g['sd']:.3f} "
            f"cv={g['cv']:.3f} range=[{g['min']:+.3f}, {g['max']:+.3f}]{flag}",
            "",
            "| type | folds | mean CV | sd CV | threshold CV | threshold mean |",
            "|---|---|---|---|---|---|",
        ]
        for t, c in row["per_type"].items():
            tcv = c["threshold_cv"]
            tmean = c["threshold_mean"]
            lines.append(
                f"| {t} | {c['n_folds_with_type']} | {c['mean_cv']:.4f} | "
                f"{c['sd_cv']:.4f} | {'—' if tcv is None else f'{tcv:.3f}'} | "
                f"{'—' if tmean is None else f'{tmean:+.3f}'} |"
            )
        lines.append("")
    return "\n".join(lines)


def worst_threshold_cv(stab: Mapping[str, Any]) -> tuple[str, float]:
    """The subset/type whose threshold moves most across folds, and by how much.

    This is the quantity the pre-registered criterion is evaluated against: if
    the threshold is not stable across folds, a rule that depends on one is not
    a rule you can report.
    """
    # Starts below zero so a perfectly stable corpus still names a cell rather
    # than reporting an empty one.
    worst_name, worst = "", -1.0
    for subset, row in stab.items():
        cv = row["global_threshold"]["cv"]
        if cv > worst:
            worst_name, worst = f"{subset}/global", cv
        for t, c in row["per_type"].items():
            tcv = c["threshold_cv"]
            if tcv is not None and tcv > worst:
                worst_name, worst = f"{subset}/{t}", tcv
    return worst_name, max(worst, 0.0)


def iter_rows(scores_by_file: Mapping[str, list]) -> Iterable:
    for rows in scores_by_file.values():
        yield from rows
