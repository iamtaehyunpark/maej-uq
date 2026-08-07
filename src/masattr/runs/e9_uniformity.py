"""E9 — uniformity stratification (spec v3 Part C §7). No new model calls.

Recomputed from E1's saved predictions: is the primary rule uniformly good, or
is it carried by one stratum? Accuracy is broken out by the gold step's act
type, by orchestrator vs worker, by subset, by trajectory length, and — where
the release carries it — by task difficulty: **function × role × subset ×
level**.

The level axis is formed *within a scale*. The released column holds two
vocabularies (numeric 1/2/3 on uuid-keyed files, verbal Medium/Hard on
hex64-keyed ones), and a stratum mixing "2" with "Medium" would be an artifact
rather than a difficulty. Files without a level are reported as their own
bucket, not silently folded into one.

There is no domain axis. The benchmark carries no domain labels and the
`question_ID` shape split is a source split rather than a topical one, so the
domain stratification is deferred to the self-generated corpus rather than
faked from an identifier.

A method that only works on ``execute`` steps in short trajectories is a
different claim from one that works across the corpus, and the aggregate number
hides which one you have.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ..attribute.rules import PRIMARY
from ..eval.ci import bootstrap_ci
from ..eval.scorers import exact_agent, exact_step
from ..loaders._common import enrich_levels
from ..typing.normalize import is_orchestrator
from ._shared import add_common, emit, flatten, load_records, open_manifest

LENGTH_BINS = ((0, 10, "short (<10)"), (10, 40, "medium (10-39)"), (40, 10**9, "long (40+)"))


def build_parser() -> argparse.ArgumentParser:
    p = add_common(argparse.ArgumentParser(prog="masattr e9", description=__doc__))
    p.add_argument("--e1-results", required=True, help="results.json written by E1")
    p.add_argument("--method", default=PRIMARY)
    p.add_argument(
        "--levels-from",
        help="their per-trajectory JSON directory, to pick up the level column "
        "the parquet drops: --levels-from alg=<dir> hc=<dir>",
        nargs="+",
        default=[],
    )
    return p


def _length_bin(n: int) -> str:
    for lo, hi, label in LENGTH_BINS:
        if lo <= n < hi:
            return label
    return "?"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = open_manifest("e9_uniformity", args)
    loaded = load_records(args)
    for kv in args.levels_from:
        subset, _, directory = kv.partition("=")
        if subset in loaded and directory:
            loaded[subset] = enrich_levels(loaded[subset], directory)
    records = {r.key: r for r in flatten(loaded)}
    blob = json.loads(Path(args.e1_results).read_text(encoding="utf-8"))

    strata: dict[str, dict[str, list[tuple[bool, bool]]]] = {
        "gold_step_type": defaultdict(list),
        "gold_role": defaultdict(list),
        "trajectory_length": defaultdict(list),
        "subset": defaultdict(list),
        "level": defaultdict(list),
    }

    n_used = 0
    for label, cfg in blob.get("configs", {}).items():
        preds = cfg.get("predictions", {}).get(args.method, {})
        for key, pred in preds.items():
            rec = records.get(key)
            if rec is None:
                continue
            n_used += 1
            gold_agent, gold_step = rec.gold
            hit = (
                exact_agent(pred.get("pred_agent"), gold_agent),
                exact_step(pred.get("pred_step"), gold_step),
            )
            gold_type = rec.steps[gold_step].type_norm
            strata["gold_step_type"][gold_type].append(hit)
            strata["gold_role"][
                "orchestrator" if is_orchestrator(gold_agent) else "worker"
            ].append(hit)
            strata["trajectory_length"][_length_bin(rec.n_steps)].append(hit)
            strata["subset"][rec.subset].append(hit)
            strata["level"][
                f"{rec.level_scale}:{rec.level}" if rec.level else "absent"
            ].append(hit)
        _ = label

    if not n_used:
        raise SystemExit(
            f"no predictions for method {args.method!r} in {args.e1_results} — "
            "E9 recomputes from E1 output and adds no new runs"
        )

    results: dict = {"method": args.method, "n_predictions": n_used, "strata": {}}
    blocks = []
    for name, buckets in strata.items():
        rows = []
        for stratum, units in sorted(buckets.items()):
            a_ci = bootstrap_ci(
                units, lambda u: sum(a for a, _ in u) / len(u), n_boot=args.n_boot, seed=args.seed
            )
            s_ci = bootstrap_ci(
                units,
                lambda u: sum(s for _, s in u) / len(u),
                n_boot=args.n_boot,
                seed=args.seed + 1,
            )
            rows.append(
                {
                    "stratum": stratum,
                    "n": len(units),
                    "agent_acc": a_ci.to_dict(),
                    "step_acc": s_ci.to_dict(),
                }
            )
        results["strata"][name] = rows
        blocks.append(
            "\n".join(
                [f"### by {name}", "", "| stratum | n | agent acc | step acc |", "|---|---|---|---|"]
                + [
                    f"| {r['stratum']} | {r['n']} | "
                    f"{r['agent_acc']['point']:.3f} [{r['agent_acc']['lo']:.3f}, {r['agent_acc']['hi']:.3f}] | "
                    f"{r['step_acc']['point']:.3f} [{r['step_acc']['lo']:.3f}, {r['step_acc']['hi']:.3f}] |"
                    for r in rows
                ]
            )
        )

    md = "\n".join(
        [
            f"# E9 — uniformity stratification ({args.method})",
            "",
            f"Recomputed from {args.e1_results}; {n_used} predictions, no new model calls.",
            "",
            "\n\n".join(blocks),
            "",
            "> Small strata carry wide intervals. Read the CIs before reading the gaps.",
        ]
    )
    emit(manifest, results, md, args.out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
