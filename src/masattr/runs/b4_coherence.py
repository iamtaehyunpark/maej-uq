"""B4 — judge-free semantic coherence fields (pilot baseline spec).

Computes ``embed_divergence`` and ``nli_contradiction`` over the corpus and
writes them as score JSONL in the same shape as the judge field, so they enter
the evaluation grid through the ordinary path: E0 for folds, E1 for rules,
scorers and slices. Nothing about the grid knows these came from a judge-free
source.

GT-off only. Neither field reads the query or the reference answer, so a GT arm
would be the identical computation under a different label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..baselines.coherence import Embedder, NLI, score_records
from ..judge.score import cost_summary
from ._shared import add_common, emit, flatten, load_records, open_manifest

FIELDS = ("embed_divergence", "nli_contradiction")


def build_parser() -> argparse.ArgumentParser:
    p = add_common(argparse.ArgumentParser(prog="masattr b4", description=__doc__))
    p.add_argument("--fields", nargs="+", default=list(FIELDS), choices=list(FIELDS))
    p.add_argument("--embed-model", required=True, help="sentence embedding model id or path")
    p.add_argument("--nli-model", required=True, help="NLI cross-encoder id or path")
    p.add_argument("--device")
    p.add_argument("--scores-dir", default="runs/base/b4/scores")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = open_manifest("b4_coherence", args)
    records = flatten(load_records(args))
    manifest.record_anomalies(records)

    scores_dir = Path(args.scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"fields": {}}

    for field in args.fields:
        model = (
            Embedder(args.embed_model, device=args.device)
            if field == "embed_divergence"
            else NLI(args.nli_model, device=args.device)
        )
        by_file = score_records(records, field, model)
        for subset in sorted({r.subset for r in records}):
            path = scores_dir / f"{subset}__{field}_nogt.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for key, rows in by_file.items():
                    if key.startswith(f"{subset}/"):
                        for r in rows:
                            fh.write(json.dumps(r.to_dict()) + "\n")
            results["fields"].setdefault(field, {})[subset] = str(path)
        results["fields"][field]["model"] = model.model_id
        results["fields"][field]["n_rows"] = sum(len(v) for v in by_file.values())

    emit(
        manifest,
        results,
        "# B4 — judge-free coherence fields\n\n"
        + "\n".join(
            f"- `{f}` ({v['model']}): {v['n_rows']} rows"
            for f, v in results["fields"].items()
        )
        + "\n\n> Step 0 has no prefix and takes the neutral value 0.5, flagged "
        "`parse_ok=false`. Inventing a first-step score would feed straight into "
        "first-step-biased rules.",
        args.out_dir,
    )
    _ = cost_summary
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
