"""Score a subset with the judge (spec v3 Part C §3).

Writes one JSONL of step scores per subset plus a cost row. The shared-prefix
path is asserted inside the scoring loop, not hoped for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..judge.client import build_client
from ..judge.score import cost_summary, score_corpus
from ._shared import add_common, add_judge_args, load_records, open_manifest, resolve_model


def build_parser() -> argparse.ArgumentParser:
    p = add_judge_args(add_common(argparse.ArgumentParser(prog="masattr judge", description=__doc__)))
    p.add_argument("--limit", type=int, help="score only the first N files (smoke runs)")
    p.add_argument("--scores-dir", default="runs/scores")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = open_manifest("judge", args)
    records = load_records(args)
    judge_spec = resolve_model(args.judge, args.transport)
    manifest.record_models(judge=judge_spec)
    manifest.record_anomalies([r for recs in records.values() for r in recs])
    client = build_client(
        judge_spec, device=args.device, seed=args.seed, base_url=args.base_url
    )

    scores_dir = Path(args.scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{judge_spec.replace(':', '_').replace('/', '-')}_{args.readout}_{args.policy}"
    tag += "_gt" if args.with_gt else "_nogt"
    if args.no_types:
        tag += "_notypes"
    if args.no_subtask_pointer:
        tag += "_nosubtask"
    if args.no_peer_corroboration:
        tag += "_nopeer"
    if args.prefix_window:
        tag += f"_win{args.prefix_window}"
    if args.lookahead != "none":
        tag += f"_{args.lookahead}"

    summary: dict = {}
    for subset, recs in records.items():
        if args.limit:
            recs = recs[: args.limit]
        out_path = scores_dir / f"{subset}__{tag}.jsonl"

        def progress(i, n, ts, subset=subset):
            if i % 25 == 0 or i == n:
                print(f"  {subset}: {i}/{n} files ({len(ts.scores)} steps)", file=sys.stderr)

        results = score_corpus(
            recs,
            client,
            kind=args.readout,
            policy=args.policy,
            with_gt=args.with_gt,
            use_types=not args.no_types,
            subtask_pointer=not args.no_subtask_pointer,
            peer_corroboration=not args.no_peer_corroboration,
            prefix_window=args.prefix_window,
            lookahead=args.lookahead,
            budget_chars=args.prefix_budget_chars,
            out_path=out_path,
            progress=progress,
        )
        summary[subset] = {"scores": str(out_path), "cost": cost_summary(results)}

    print(json.dumps(summary, indent=2))
    manifest.results = summary
    manifest.write(args.out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
