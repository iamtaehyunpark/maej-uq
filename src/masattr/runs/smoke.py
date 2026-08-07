"""Stage-0 smoke — the go/no-go gate before anything else runs.

Ten seeded files, three evidence arms, both GT settings. The question is not
"how accurate is attribution" — it is **is there a field at all**. A saturated
or structureless score field cannot be localized by any rule, and finding that
out after a full 50k-assessment pass would be an expensive way to learn it.

The three arms, in this harness's switches:

===========  ======================================================
W0           ``--policy plain`` — prefix only, no augmentation
W+resp       ``typed`` with peer corroboration, no subtask pointer
W+own        ``typed`` with the subtask pointer, no peer corroboration
===========  ======================================================

Output is one curve per file per arm — the per-step score against step index,
with the annotated mistake step marked — as ASCII in the report and as CSV for
real plotting. Plus the E0 degeneracy read on the smoke sample, which is what
the gate turns on.

Passing means: no degenerate cell, and the field moves. It does **not** mean the
attribution works; that is E1's question.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from ..judge.client import build_client
from ..judge.score import cost_summary, score_corpus
from ..normalize.apply import field_sanity, render_field
from ._shared import add_common, add_judge_args, emit, flatten, load_records, open_manifest, resolve_model

#: The three evidence arms, as (name, policy, subtask_pointer, peer_corroboration).
ARMS = (
    ("W0", "plain", False, False),
    ("W+resp", "typed", False, True),
    ("W+own", "typed", True, False),
)

BLOCKS = " ▁▂▃▄▅▆▇█"


def build_parser() -> argparse.ArgumentParser:
    p = add_judge_args(add_common(argparse.ArgumentParser(prog="masattr smoke", description=__doc__)))
    p.add_argument("--n-files", type=int, default=10, help="seeded sample size")
    p.add_argument("--scores-dir", default="runs/smoke/scores")
    return p


def sample_files(records, n: int, seed: int) -> list:
    """Seeded sample, balanced across subsets so neither idiom is unrepresented."""
    rng = random.Random(seed)
    by_subset: dict[str, list] = {}
    for r in records:
        by_subset.setdefault(r.subset, []).append(r)
    out = []
    subsets = sorted(by_subset)
    per = max(n // max(len(subsets), 1), 1)
    for subset in subsets:
        pool = sorted(by_subset[subset], key=lambda r: r.key)
        out.extend(rng.sample(pool, min(per, len(pool))))
    # Top up deterministically if the split left room.
    rest = [r for r in sorted(records, key=lambda r: r.key) if r not in out]
    out.extend(rest[: max(n - len(out), 0)])
    return out[:n]


def sparkline(values: list[float], lo: float = 0.0, hi: float = 1.0) -> str:
    span = (hi - lo) or 1.0
    return "".join(
        BLOCKS[min(int((min(max(v, lo), hi) - lo) / span * (len(BLOCKS) - 1)), len(BLOCKS) - 1)]
        for v in values
    )


def curve_block(record, rows, arm: str, with_gt: bool) -> str:
    """One file's curve, with the annotated mistake step marked underneath."""
    ps = [r.p_raw for r in rows]
    marker = "".join("^" if i == record.label_mistake_step else "·" for i in range(len(ps)))
    at = ps[record.label_mistake_step] if record.label_mistake_step < len(ps) else float("nan")
    return "\n".join(
        [
            f"`{record.key}`  arm={arm}  gt={'on' if with_gt else 'off'}  "
            f"T={len(ps)}  mistake@{record.label_mistake_step} ({record.label_mistake_agent})",
            "```",
            sparkline(ps),
            marker,
            "```",
            f"p at the annotated step: **{at:.3f}**  ·  "
            f"min {min(ps):.3f}  median {sorted(ps)[len(ps) // 2]:.3f}  max {max(ps):.3f}  "
            f"· rank of the annotated step: {sorted(ps).index(at) + 1}/{len(ps)}"
            if ps
            else "",
        ]
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = open_manifest("smoke", args)
    judge_spec = resolve_model(args.judge)
    manifest.record_models(judge=judge_spec)

    records = sample_files(flatten(load_records(args)), args.n_files, args.seed)
    manifest.record_anomalies(records)
    client = build_client(judge_spec, device=args.device, seed=args.seed)

    scores_dir = Path(args.scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"judge": judge_spec, "files": [r.key for r in records], "arms": {}}
    blocks: list[str] = []
    rows_csv: list[dict] = []
    all_sanity: dict = {}

    for arm, policy, subtask, peers in ARMS:
        for with_gt in (False, True):
            tag = f"{arm.replace('+', '_')}_{'gt' if with_gt else 'nogt'}"
            path = scores_dir / f"smoke__{tag}.jsonl"
            traj = score_corpus(
                records,
                client,
                kind=args.readout,
                policy=policy,
                with_gt=with_gt,
                use_types=not args.no_types,
                subtask_pointer=subtask,
                peer_corroboration=peers,
                budget_chars=args.prefix_budget_chars,
                out_path=path,
            )
            by_key = {t.key: t.scores for t in traj}
            sanity = field_sanity(records, by_key)
            all_sanity[tag] = sanity
            results["arms"][tag] = {
                "scores": str(path),
                "cost": cost_summary(traj),
                "n_degenerate": len(sanity["degenerate"]),
                "degenerate": sanity["degenerate"],
            }
            blocks.append(f"## {arm} · GT {'on' if with_gt else 'off'}\n")
            for rec in records:
                rows = by_key.get(rec.key, [])
                if rows:
                    blocks.append(curve_block(rec, rows, arm, with_gt))
                    for r in rows:
                        rows_csv.append(
                            {
                                "arm": arm,
                                "with_gt": with_gt,
                                "file": rec.key,
                                "step": r.step_idx,
                                "type": r.type_norm,
                                "agent": r.agent,
                                "p_raw": r.p_raw,
                                "is_mistake_step": int(r.step_idx == rec.label_mistake_step),
                            }
                        )
            blocks.append(render_field(sanity))

    csv_path = out_dir / "curves.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]) if rows_csv else ["arm"])
        w.writeheader()
        w.writerows(rows_csv)

    degenerate = {k: v["degenerate"] for k, v in results["arms"].items() if v["degenerate"]}
    passed = not degenerate
    results["gate"] = {
        "passed": passed,
        "degenerate_arms": degenerate,
        "curves_csv": str(csv_path),
    }
    verdict = (
        "**GATE PASSED** — every arm's field varies and none is saturated. The "
        "main pass may proceed; pick one arm per the config-lock rule."
        if passed
        else "**GATE FAILED** — "
        + "; ".join(f"{k}: {', '.join(v)}" for k, v in degenerate.items())
        + ". Fix the probe prompt and re-smoke; nothing else runs."
    )
    manifest.note(f"smoke gate {'passed' if passed else 'FAILED'} on {len(records)} files")

    md = "\n\n".join(
        [
            "# Stage-0 smoke",
            f"{len(records)} seeded files (seed={args.seed}), judge `{judge_spec}`, "
            f"readout `{args.readout}`. Three evidence arms × both GT settings.",
            f"Curves also written to `{csv_path}` for plotting.",
            "## Gate",
            verdict,
            *blocks,
        ]
    )
    emit(manifest, results, md, args.out_dir)
    return 0 if passed else 4


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
