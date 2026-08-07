"""Stage-0 smoke — the go/no-go gate before anything else runs.

Ten seeded files, three evidence arms, both GT settings. The question is not
"how accurate is attribution" — it is **is there a field at all**. A saturated
or structureless score field cannot be localized by any rule, and finding that
out after a full 50k-assessment pass would be an expensive way to learn it.

The three arms differ **only** in the lookahead window appended after step
``t``. Base evidence assembly — including the near-empty-execute rescue — is
identical in all three; conflating the two was a real bug once.

===========  ==========================================================
W0           no lookahead. Prefix-conditional.
W+resp       the following contiguous steps by other agents, capped at 2
W+own        W+resp plus the acting agent's own next appearance
===========  ==========================================================

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
from ..typing.normalize import is_orchestrator
from ._shared import add_common, add_judge_args, emit, flatten, load_records, open_manifest, resolve_model

#: The three evidence arms, as (name, lookahead). Base assembly is ``typed``
#: throughout, so the arms isolate the lookahead window and nothing else.
ARMS = (("W0", "none"), ("W+resp", "resp"), ("W+own", "own"))

BLOCKS = " ▁▂▃▄▅▆▇█"


def build_parser() -> argparse.ArgumentParser:
    p = add_judge_args(add_common(argparse.ArgumentParser(prog="masattr smoke", description=__doc__)))
    p.add_argument("--n-files", type=int, default=10, help="seeded sample size")
    p.add_argument("--scores-dir", default="runs/smoke/scores")
    return p


#: Composition the smoke must cover, as (name, subset, predicate). A random
#: seeded draw can easily miss all three — a 10-file sample of mostly short AG
#: logs would say nothing about the long-trajectory or orchestrator-fault cases
#: the gate exists to probe — so each is seeded in first and the rest fills up.
REQUIRED = (
    ("hc_long", "hc", lambda r: r.n_steps > 80),
    ("alg_early_error", "alg", lambda r: r.label_mistake_step <= 1),
    ("hc_orchestrator_fault", "hc", lambda r: is_orchestrator(r.label_mistake_agent)),
)


def sample_files(records, n: int, seed: int) -> tuple[list, dict[str, str]]:
    """Seeded sample: required cases first, then balanced across subsets.

    Returns the sample and which file satisfied each required case, so the
    report can show the composition was met rather than assert it.
    """
    rng = random.Random(seed)
    by_key = {r.key: r for r in records}
    chosen: list = []
    covered: dict[str, str] = {}

    for name, subset, pred in REQUIRED:
        pool = sorted(
            (r for r in records if r.subset == subset and pred(r) and r.key not in {c.key for c in chosen}),
            key=lambda r: r.key,
        )
        if pool:
            pick = rng.choice(pool)
            chosen.append(pick)
            covered[name] = pick.key
        else:
            covered[name] = ""

    by_subset: dict[str, list] = {}
    for r in records:
        by_subset.setdefault(r.subset, []).append(r)
    target = {s: n // max(len(by_subset), 1) for s in by_subset}
    for subset in sorted(by_subset):
        have = sum(1 for c in chosen if c.subset == subset)
        pool = sorted(
            (r for r in by_subset[subset] if r.key not in {c.key for c in chosen}),
            key=lambda r: r.key,
        )
        take = max(target[subset] - have, 0)
        chosen.extend(rng.sample(pool, min(take, len(pool))))

    rest = [r for r in sorted(records, key=lambda r: r.key) if r.key not in {c.key for c in chosen}]
    chosen.extend(rest[: max(n - len(chosen), 0)])
    _ = by_key
    return chosen[:n], covered


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
    judge_spec = resolve_model(args.judge, args.transport)
    manifest.record_models(judge=judge_spec)

    records, covered = sample_files(flatten(load_records(args)), args.n_files, args.seed)
    missing = [k for k, v in covered.items() if not v]
    if missing:
        manifest.note(f"smoke composition could not cover: {', '.join(missing)}")
    manifest.record_anomalies(records)
    client = build_client(
        judge_spec, device=args.device, seed=args.seed, base_url=args.base_url
    )

    scores_dir = Path(args.scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "judge": judge_spec,
        "files": [r.key for r in records],
        "required_cases": covered,
        "arms": {},
    }
    blocks: list[str] = []
    rows_csv: list[dict] = []
    all_sanity: dict = {}

    for arm, lookahead in ARMS:
        for with_gt in (False, True):
            tag = f"{arm.replace('+', '_')}_{'gt' if with_gt else 'nogt'}"
            path = scores_dir / f"smoke__{tag}.jsonl"
            traj = score_corpus(
                records,
                client,
                kind=args.readout,
                policy=args.policy,
                with_gt=with_gt,
                use_types=not args.no_types,
                lookahead=lookahead,
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
            "**Required composition** — "
            + "; ".join(
                f"{k}: {v or '**NOT COVERED**'}" for k, v in covered.items()
            ),
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
