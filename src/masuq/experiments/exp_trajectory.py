"""Trajectory track — U vs per-run correctness on the two MATU cells (spec §5).

Primary metrics: AUROC and AUARC of trajectory ``U`` against per-run
correctness, reported per cell. Every aggregator from :mod:`masuq.aggregate` is
reported so noisy-OR is compared against, not merely asserted over, its
alternatives.

Comparability row: MATU's published Qwen2.5 columns (Tables 1–2). The framing
difference has to travel with the number — ours is **per-run, single-pass**;
theirs is **per-task, N=10**. Those are different estimands, and the row exists
to situate our numbers, not to claim a win.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..aggregate import AGGREGATORS, PRIMARY, aggregate_corpus
from ..metrics import CI, auarc, auarc_normalized, auroc, bootstrap_ci
from ..schema import Record


@dataclass
class CellResult:
    subset: str
    n_trajectories: int
    base_accuracy: float
    aggregators: dict[str, dict] = field(default_factory=dict)
    comparability: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "subset": self.subset,
            "n_trajectories": self.n_trajectories,
            "base_accuracy": self.base_accuracy,
            "primary_aggregator": PRIMARY,
            "aggregators": self.aggregators,
            "comparability": self.comparability,
        }

    def render(self) -> str:
        lines = [
            f"## Trajectory track — {self.subset}",
            "",
            f"n={self.n_trajectories}  base accuracy={self.base_accuracy:.3f}  "
            f"(per-run, single-pass)",
            "",
            "| aggregator | AUROC (U vs failure) | AUARC | AUARC norm |",
            "|---|---|---|---|",
        ]
        for name, row in self.aggregators.items():
            mark = " **(primary)**" if name == PRIMARY else ""
            ci = row.get("auroc_ci")
            ci_s = f" [{ci['lo']:.3f}, {ci['hi']:.3f}]" if ci else ""
            lines.append(
                f"| {name}{mark} | {row['auroc']:.4f}{ci_s} | {row['auarc']:.4f} | "
                f"{row['auarc_norm']:.4f} |"
            )
        if self.comparability:
            lines += [
                "",
                "Comparability: " + json.dumps(self.comparability),
                "",
                "> Framing: ours is per-run and single-pass; MATU's published Qwen2.5 "
                "columns are per-task over N=10 runs. Different estimands — the row "
                "situates the number, it does not compare a score.",
            ]
        return "\n".join(lines)


def run_cell(
    records: Sequence[Record],
    scores_by_key: dict[str, list],
    *,
    subset: str,
    use_calibrated: bool = True,
    n_boot: int = 2000,
    seed: int = 0,
    published: dict | None = None,
) -> CellResult:
    labels = {r.key: r.label_correct for r in records if r.label_correct is not None}
    grouped = {k: v for k, v in scores_by_key.items() if k in labels}
    if not grouped:
        raise ValueError(f"{subset}: no labelled trajectories with scores")

    us = aggregate_corpus(grouped, labels, use_calibrated=use_calibrated)
    correct = [bool(u.label_correct) for u in us]
    base_acc = sum(correct) / len(correct)

    rows: dict[str, dict] = {}
    for name in list(AGGREGATORS) + (["noisy_or_no_final"] if "noisy_or_no_final" in us[0].values else []):
        vals = [u.values[name] for u in us]
        # AUROC of U against *failure*: high U should mean the run was wrong.
        units = list(zip(vals, correct))
        ci: CI = bootstrap_ci(
            units,
            lambda s: auroc([v for v, _ in s], [not c for _, c in s]),
            n_boot=n_boot,
            seed=seed,
        )
        rows[name] = {
            "auroc": auroc(vals, [not c for c in correct]),
            "auroc_ci": ci.to_dict(),
            "auarc": auarc(vals, correct),
            "auarc_norm": auarc_normalized(vals, correct),
        }

    return CellResult(
        subset=subset,
        n_trajectories=len(us),
        base_accuracy=base_acc,
        aggregators=rows,
        comparability=published or {},
    )


def run(
    cells: dict[str, tuple[Sequence[Record], dict[str, list]]],
    *,
    use_calibrated: bool = True,
    out_dir: str | Path | None = None,
    n_boot: int = 2000,
    published: dict[str, dict] | None = None,
) -> dict[str, CellResult]:
    results = {
        subset: run_cell(
            recs,
            scores,
            subset=subset,
            use_calibrated=use_calibrated,
            n_boot=n_boot,
            published=(published or {}).get(subset),
        )
        for subset, (recs, scores) in cells.items()
    }
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "trajectory.json").write_text(
            json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2), encoding="utf-8"
        )
        (out / "trajectory.md").write_text(
            "\n\n".join(r.render() for r in results.values()) + "\n", encoding="utf-8"
        )
    return results
