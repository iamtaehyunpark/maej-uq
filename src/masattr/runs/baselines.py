"""Who&When baseline reproduction (Part C §6).

all_at_once / step_by_step / binary_search, run twice: with gpt-4o (their
regime) and with our judge model (the capability control). Without both arms,
any gap between their methods and ours is confounded with judge capability.

``--impl repo`` imports their ``inference.py`` and calls their functions —
nothing patched except credentials arriving via flags. ``--impl local`` is a
re-prompting of the same three strategies so the pipeline runs without their
checkout; every row it produces is stamped ``impl=local`` and is **not** the
reproduction.
"""

from __future__ import annotations

import argparse
import json

from ..baselines.whowhen_repo import METHODS, build_generator, run_method
from ..eval.scorers import gold_map, render, score_all
from ._shared import add_common, emit, flatten, held_aside_keys, load_records, open_manifest


def build_parser() -> argparse.ArgumentParser:
    p = add_common(argparse.ArgumentParser(prog="masattr baselines", description=__doc__))
    p.add_argument(
        "--generators",
        nargs="+",
        default=["mock"],
        help="mock | openai:<model> | judge:<judge-spec> — pass both arms",
    )
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--impl", default="local", choices=("repo", "local"))
    p.add_argument("--repo-path", help="Who&When checkout (required for --impl repo)")
    p.add_argument("--api-key", help="passed explicitly; their env-var fallback is not implemented")
    p.add_argument("--device")
    p.add_argument("--limit", type=int, help="first N files per subset (smoke runs)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = open_manifest("baselines", args)
    records_by_subset = load_records(args)
    all_records = flatten(records_by_subset)
    held = held_aside_keys(all_records, args.seed)

    if args.impl == "local":
        manifest.note(
            "impl=local: a re-prompting of the three strategies, NOT the Who&When "
            "reproduction. Do not report these rows as reproduced numbers."
        )

    results: dict = {"impl": args.impl, "runs": []}
    table: dict[str, dict] = {}

    for subset, records in records_by_subset.items():
        gold = gold_map(records)
        for gen_spec in args.generators:
            gen = build_generator(gen_spec, api_key=args.api_key, device=args.device)
            for method in args.methods:
                run = run_method(
                    records,
                    gen,
                    method=method,
                    subset=subset,
                    impl=args.impl,
                    repo_path=args.repo_path,
                    limit=args.limit,
                )
                scored = score_all(
                    {k: _Pred(v) for k, v in run.preds.items()},
                    gold,
                    records,
                    held_aside=held,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                label = f"{method} · {gen.name} · subset={subset} · impl={args.impl}"
                table[label] = scored
                results["runs"].append({**run.to_dict(), "scores": scored})

    md = "\n".join(
        [
            "# Who&When baselines",
            "",
            render(table),
            "",
            f"> impl={args.impl}."
            + (
                " These are NOT reproduced numbers — `--impl local` re-prompts the "
                "strategies rather than calling their code."
                if args.impl == "local"
                else " Their functions were called directly; only credentials were passed in."
            ),
            "",
            "> AgenTracer and StepFinder are cited from published numbers; not reproduced.",
        ]
    )
    emit(manifest, results, md, args.out_dir)
    print(json.dumps({r["method"]: r["llm_calls"] for r in results["runs"]}, indent=2))
    return 0


class _Pred:
    """Adapter so baseline ``(agent, step)`` tuples score through the same path."""

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[str | None, int | None]) -> None:
        self.pair = pair

    def as_pair(self) -> tuple[str | None, int | None]:
        return self.pair


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
