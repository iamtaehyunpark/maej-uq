"""Who&When baseline reproduction (spec v3 Part C §6).

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
from pathlib import Path

from ..baselines.whowhen_repo import (
    METHODS,
    build_generator,
    id_map,
    output_path,
    parse_repo_output,
    run_method,
    run_repo_subprocess,
)
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
    p.add_argument(
        "--repo-data",
        help="their per-trajectory JSON directory, one per subset: "
        "--repo-data alg=<dir> hc=<dir>",
        nargs="+",
        default=[],
    )
    p.add_argument("--repo-model", default="gpt-4o", help="a model id from their ALL_MODELS")
    p.add_argument(
        "--served-base-url",
        help="redirect their OpenAI client here — the capability control: their "
        "prompts and logic, our judge",
    )
    p.add_argument("--served-model", help="model name to rewrite their calls to")
    p.add_argument(
        "--repo-python",
        help="interpreter with their dependencies; theirs is a different "
        "environment from this package's",
    )
    p.add_argument("--api-key", help="passed explicitly on the command line")
    p.add_argument(
        "--api-key-file",
        help="read the key from a file instead, so it stays out of shell history",
    )

    p.add_argument("--device")
    p.add_argument("--limit", type=int, help="first N files per subset (smoke runs)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.api_key_file:
        args.api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
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
    repo_data = dict(kv.split("=", 1) for kv in args.repo_data)

    if args.impl == "repo":
        for subset, records in records_by_subset.items():
            gold = gold_map(records)
            directory = repo_data.get(subset)
            if not directory:
                raise SystemExit(
                    f"--impl repo needs their JSON directory for {subset!r}: "
                    f"--repo-data {subset}=<dir>"
                )
            ids = id_map(directory)
            for method in args.methods:
                receipt = Path(args.out_dir) / f"snapshot_{method}_{subset}.txt"
                receipt.parent.mkdir(parents=True, exist_ok=True)
                run_repo_subprocess(
                    args.repo_path,
                    method=method,
                    model=args.repo_model,
                    directory_path=directory,
                    is_handcrafted=(subset == "hc"),
                    api_key=args.api_key or "not-needed",
                    device=args.device,
                    snapshot_receipt=receipt,
                    base_url=args.served_base_url,
                    model_rewrite=args.served_model,
                    python_exe=args.repo_python,
                )
                snapshots = (
                    sorted(set(receipt.read_text().split())) if receipt.exists() else []
                )
                if snapshots:
                    manifest.note(
                        f"{method}/{subset}: API returned model snapshot(s) "
                        + ", ".join(snapshots)
                    )
                out_file = output_path(
                    args.repo_path,
                    method=method,
                    model=args.repo_model,
                    is_handcrafted=(subset == "hc"),
                )
                preds = parse_repo_output(
                    out_file.read_text(encoding="utf-8"), ids, subset
                )
                if not preds:
                    raise SystemExit(
                        f"parsed no predictions from {out_file}; their output format "
                        "may have changed — check it before trusting any row"
                    )
                scored = score_all(
                    {k: _Pred(v) for k, v in preds.items()},
                    gold,
                    records,
                    held_aside=held,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                arm = args.served_model or args.repo_model
                label = f"{method} · {arm} · subset={subset} · impl=repo"
                table[label] = scored
                results["runs"].append(
                    {
                        "subset": subset,
                        "method": method,
                        "generator": args.served_model or args.repo_model,
                        "impl": "repo",
                        "n": len(preds),
                        "n_gold": len(gold),
                        "output_file": str(out_file),
                        "model_snapshots": snapshots,
                        "scores": scored,
                    }
                )
        md = "\n".join(
            ["# Who&When baselines (their script)", "", render(table), "",
             "> Their `inference.py` was invoked as a subprocess and its result file "
             "parsed on their own contract; nothing of theirs was edited or imported."]
        )
        emit(manifest, results, md, args.out_dir)
        return 0

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
