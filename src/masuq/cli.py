"""Command-line entry point.

The weekend build order of spec §7 maps onto these commands, in order::

    masuq paths                       # where is the data, and is it there
    masuq load --subset alg --assert  # 1. W&W loaders + flags + counts
    masuq load --subset camel_math --assert
    masuq typecheck                   # 2. classifier validation table
    masuq smoke                       # 3. judge harness smoke test
    masuq judge --subset ...          # full scoring run
    masuq exp0                        # 4. falsifier — must run before attribution
    masuq trajectory                  # trajectory track
    masuq attribution                 # attribution track (gated on exp0)
    masuq baselines                   # W&W repro, both judge arms
    masuq audit                       # MATU label spot-audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import RunConfig, load_paths
from .loaders import (
    check_matu_cell,
    check_subset,
    check_whowhen_steps,
    join_labels,
    load_subset,
)
from .schema import Record, corpus_stats, read_jsonl, write_jsonl

SUBSETS = ("camel_math", "autogen_mmlu", "alg", "hc")
MATU_SUBSETS = ("camel_math", "autogen_mmlu")


# --- shared helpers ---------------------------------------------------------


def _resolve_records(args, subset: str) -> list[Record]:
    """Load a subset either from a cached JSONL or from the raw source."""
    cache = Path(args.cache_dir) / f"{subset}.jsonl" if args.cache_dir else None
    if cache and cache.exists() and not args.no_cache:
        return list(read_jsonl(cache))

    paths = load_paths(args.config, args.data_root)
    src = getattr(args, "path", None) or paths.get(subset)
    records, report = load_subset(subset, src)
    if subset in MATU_SUBSETS:
        label_path = getattr(args, "labels", None) or paths.get(f"{subset}_labels")
        if Path(label_path).exists():
            join_labels(records, label_path, strict=not args.lenient_labels)
        else:
            print(f"warning: no accuracy dict at {label_path}; labels left null", file=sys.stderr)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(records, cache)
    _ = report
    return records


def _scores_by_key(path: str | Path) -> dict[str, list]:
    from .judge.harness import group_by_record, load_scores

    return group_by_record(load_scores(path))


def _make_scorer(args):
    from .judge.backends import HFPrefixScorer, MockPrefixScorer, VerbalizedPrefixScorer

    if args.backend == "mock":
        return MockPrefixScorer(seed=args.seed)
    if args.backend == "hf":
        scorer = HFPrefixScorer(args.model, device=args.device)
        if args.readout == "verbalized":
            from .judge.backends import HFGenerator

            return VerbalizedPrefixScorer(HFGenerator(args.model, device=args.device))
        return scorer
    raise SystemExit(f"unknown backend {args.backend!r}")


def _make_generator(spec: str, api_key: str | None):
    """``mock`` | ``hf:<model_id>`` | ``openai:<model>``"""
    from .judge.backends import HFGenerator, MockGenerator, OpenAIGenerator

    if spec == "mock":
        return MockGenerator()
    kind, _, name = spec.partition(":")
    if kind == "hf":
        return HFGenerator(name)
    if kind == "openai":
        if not api_key:
            raise SystemExit(
                "openai generator needs --api-key (spec §6: credentials via CLI flags)"
            )
        return OpenAIGenerator(name or "gpt-4o", api_key=api_key)
    raise SystemExit(f"unknown generator spec {spec!r}")


def _emit(obj, out: str | None) -> None:
    text = json.dumps(obj, indent=2, default=str)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    print(text)


# --- commands ---------------------------------------------------------------


def cmd_paths(args) -> int:
    paths = load_paths(args.config, args.data_root)
    status = paths.status()
    _emit({"root": str(paths.root), "entries": status}, args.out)
    missing = [k for k, v in status.items() if not v["exists"]]
    if missing:
        print(f"\n{len(missing)} missing: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


def cmd_load(args) -> int:
    subsets = [args.subset] if args.subset else list(SUBSETS)
    out: dict = {}
    loaded: dict[str, list[Record]] = {}
    for subset in subsets:
        records = _resolve_records(args, subset)
        loaded[subset] = records
        out[subset] = corpus_stats(records)
        if args.assert_counts:
            problems = check_subset(records, subset, strict=False)
            if subset in MATU_SUBSETS:
                problems += check_matu_cell(records, subset, strict=False)
            out[subset]["expectation_violations"] = problems
        if args.out_jsonl:
            p = Path(args.out_jsonl)
            p.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(records, p if len(subsets) == 1 else p.with_name(f"{subset}.jsonl"))

    if args.assert_counts and {"alg", "hc"} <= set(loaded):
        out["whowhen_steps"] = check_whowhen_steps(loaded["alg"], loaded["hc"], strict=False) or "ok"

    _emit(out, args.out)
    violations = sum(
        len(v.get("expectation_violations", [])) for v in out.values() if isinstance(v, dict)
    )
    if args.assert_counts and violations:
        print(f"\n{violations} expectation violations", file=sys.stderr)
        return 1
    return 0


def cmd_typecheck(args) -> int:
    from .typing_.validate import audit_sample, validate_against_known

    out: dict = {}
    for subset, source in (("autogen_mmlu", "native"), ("hc", "parsed")):
        records = _resolve_records(args, subset)
        report = validate_against_known(records, reference_source=source, subset=subset)
        out[subset] = report.to_dict()
        print(report.render())
        print()
        if args.audit_out:
            Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.audit_out).write_text(
                json.dumps(audit_sample(records, n=100, seed=args.seed), indent=2),
                encoding="utf-8",
            )
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if all(v["passes_gate"] for v in out.values()) else 1


def cmd_smoke(args) -> int:
    """Spec §7.3: 50 steps per subset, sanity-check score distributions per type."""
    import numpy as np

    from .judge.harness import cost_summary, judge_record

    scorer = _make_scorer(args)
    out: dict = {}
    for subset in (args.subset,) if args.subset else SUBSETS:
        records = _resolve_records(args, subset)
        budget = args.n_steps
        results = []
        for rec in records:
            if budget <= 0:
                break
            ts = judge_record(rec, scorer, readout=args.readout, policy=args.policy)
            results.append(ts)
            budget -= len(ts.scores)
        by_type: dict[str, list[float]] = {}
        for ts in results:
            for s in ts.scores:
                by_type.setdefault(s.type_norm, []).append(s.p_raw)
        out[subset] = {
            "cost": cost_summary(results),
            "p_by_type": {
                t: {
                    "n": len(v),
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                }
                for t, v in sorted(by_type.items())
            },
            "n_augmented": sum(s.augmented for ts in results for s in ts.scores),
        }
    _emit(out, args.out)
    return 0


def cmd_judge(args) -> int:
    from .judge.harness import cost_summary, judge_corpus

    scorer = _make_scorer(args)
    records = _resolve_records(args, args.subset)
    if args.limit:
        records = records[: args.limit]
    out_path = args.out_scores or f"runs/scores_{args.subset}.jsonl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    def progress(i, n, ts):
        if i % 25 == 0 or i == n:
            print(f"  {i}/{n} trajectories ({len(ts.scores)} steps)", file=sys.stderr)

    results = judge_corpus(
        records,
        scorer,
        readout=args.readout,
        policy=args.policy,
        out_path=out_path,
        progress=progress,
    )
    _emit({"scores": out_path, "cost": cost_summary(results)}, args.out)
    return 0


def cmd_exp0(args) -> int:
    from .experiments import exp0_calibration_transfer as exp0

    fit_records = _resolve_records(args, "autogen_mmlu")
    test_records = _resolve_records(args, "camel_math")
    res, cal = exp0.run(
        fit_records,
        _scores_by_key(args.fit_scores),
        test_records,
        _scores_by_key(args.test_scores),
        method=args.method,
        out_dir=args.out_dir,
    )
    print(res.render())
    return 0 if res.transfers else 2  # 2 = falsified; the fallback path is now in force


def cmd_trajectory(args) -> int:
    from .calibration import TypedCalibrator
    from .experiments import exp_trajectory

    cal = TypedCalibrator.load(args.calibrator) if args.calibrator else None
    cells = {}
    for subset, scores_path in zip(args.subsets, args.scores):
        records = _resolve_records(args, subset)
        grouped = _scores_by_key(scores_path)
        if cal:
            for v in grouped.values():
                cal.apply_to_scores(v)
        cells[subset] = (records, grouped)
    results = exp_trajectory.run(
        cells, use_calibrated=bool(cal), out_dir=args.out_dir, n_boot=args.n_boot
    )
    for r in results.values():
        print(r.render())
        print()
    return 0


def cmd_attribution(args) -> int:
    from .calibration import TypedCalibrator
    from .experiments import exp_attribution

    cal = TypedCalibrator.load(args.calibrator) if args.calibrator else None
    threshold = args.threshold
    if threshold is None:
        if not args.threshold_file:
            raise SystemExit(
                "attribution needs a threshold chosen on the calibration corpus "
                "(--threshold or --threshold-file from exp0); choosing it on W&W "
                "would leak"
            )
        threshold = json.loads(Path(args.threshold_file).read_text())["threshold"]

    subsets = {}
    for subset, scores_path in zip(args.subsets, args.scores):
        records = _resolve_records(args, subset)
        grouped = _scores_by_key(scores_path)
        if cal:
            for v in grouped.values():
                cal.apply_to_scores(v)
        subsets[subset] = (records, grouped)

    results = exp_attribution.run(
        subsets,
        threshold=threshold,
        use_calibrated=bool(cal),
        out_dir=args.out_dir,
        n_boot=args.n_boot,
    )
    for r in results.values():
        print(r.render())
        print()
    return 0


def cmd_baselines(args) -> int:
    from .experiments import exp_baselines

    generators = {spec: _make_generator(spec, args.api_key) for spec in args.generators}
    subsets = {s: _resolve_records(args, s) for s in args.subsets}
    results = exp_baselines.run(
        subsets,
        generators,
        methods=args.methods,
        out_dir=args.out_dir,
        limit=args.limit,
        n_boot=args.n_boot,
    )
    print(exp_baselines.render(results))
    return 0


def cmd_audit(args) -> int:
    from .experiments import exp_label_audit

    judges = [_make_generator(spec, args.api_key) for spec in args.judges]
    records = _resolve_records(args, args.subset)
    report = exp_label_audit.run(
        records, judges, subset=args.subset, n=args.n, seed=args.seed, out_dir=args.out_dir
    )
    print(report.render())
    return 0


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="masuq", description=__doc__)
    p.add_argument("--version", action="version", version=f"masuq {__version__}")
    p.add_argument("--config", help="JSON file with data paths")
    p.add_argument("--data-root", help="data root (overrides config and $MASUQ_DATA_ROOT)")
    p.add_argument("--cache-dir", default="runs/cache", help="where loaded JSONL is cached")
    p.add_argument("--no-cache", action="store_true", help="ignore cached JSONL")
    p.add_argument("--lenient-labels", action="store_true", help="do not hard-fail on label join")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the command's JSON summary here")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("paths", help="show resolved data paths")
    sp.set_defaults(fn=cmd_paths)

    sp = sub.add_parser("load", help="run loaders, print stats, assert counts")
    sp.add_argument("--subset", choices=SUBSETS)
    sp.add_argument("--path", help="explicit source path for --subset")
    sp.add_argument("--labels", help="explicit accuracy_dict path (MATU)")
    sp.add_argument("--assert", dest="assert_counts", action="store_true")
    sp.add_argument("--out-jsonl", help="write unified records here")
    sp.set_defaults(fn=cmd_load)

    sp = sub.add_parser("typecheck", help="classifier vs native/parsed types (spec §2)")
    sp.add_argument("--audit-out", help="write the 100-step manual audit sample here")
    sp.set_defaults(fn=cmd_typecheck)

    def judge_args(sp):
        sp.add_argument("--backend", default="mock", choices=("mock", "hf"))
        sp.add_argument("--model", default="", help="HF model id when --backend hf")
        sp.add_argument("--device")
        sp.add_argument("--readout", default="ptrue", choices=("ptrue", "verbalized"))
        sp.add_argument(
            "--policy", default="type_conditioned_v1", choices=("type_conditioned_v1", "prefix_only")
        )

    sp = sub.add_parser("smoke", help="50-step smoke test per subset (spec §7.3)")
    judge_args(sp)
    sp.add_argument("--subset", choices=SUBSETS)
    sp.add_argument("--n-steps", type=int, default=50)
    sp.set_defaults(fn=cmd_smoke)

    sp = sub.add_parser("judge", help="score a subset")
    judge_args(sp)
    sp.add_argument("--subset", choices=SUBSETS, required=True)
    sp.add_argument("--limit", type=int)
    sp.add_argument("--out-scores")
    sp.set_defaults(fn=cmd_judge)

    sp = sub.add_parser("exp0", help="falsifier: does frozen typed calibration transfer?")
    sp.add_argument("--fit-scores", required=True, help="scores JSONL for autogen_mmlu")
    sp.add_argument("--test-scores", required=True, help="scores JSONL for camel_math")
    sp.add_argument("--method", default="percentile", choices=("percentile", "platt", "isotonic"))
    sp.add_argument("--out-dir", default="runs/exp0")
    sp.set_defaults(fn=cmd_exp0)

    sp = sub.add_parser("trajectory", help="trajectory track (MATU)")
    sp.add_argument("--subsets", nargs="+", default=list(MATU_SUBSETS))
    sp.add_argument("--scores", nargs="+", required=True, help="one scores JSONL per subset")
    sp.add_argument("--calibrator", help="frozen calibrator JSON from exp0")
    sp.add_argument("--n-boot", type=int, default=2000)
    sp.add_argument("--out-dir", default="runs/trajectory")
    sp.set_defaults(fn=cmd_trajectory)

    sp = sub.add_parser("attribution", help="attribution track (W&W)")
    sp.add_argument("--subsets", nargs="+", default=["alg", "hc"])
    sp.add_argument("--scores", nargs="+", required=True)
    sp.add_argument("--calibrator")
    sp.add_argument("--threshold", type=float)
    sp.add_argument("--threshold-file", help="threshold.json written by exp0")
    sp.add_argument("--n-boot", type=int, default=2000)
    sp.add_argument("--out-dir", default="runs/attribution")
    sp.set_defaults(fn=cmd_attribution)

    sp = sub.add_parser("baselines", help="W&W baseline reproduction (spec §6)")
    sp.add_argument("--subsets", nargs="+", default=["alg", "hc"])
    sp.add_argument(
        "--generators",
        nargs="+",
        default=["mock"],
        help="mock | hf:<model> | openai:<model> — pass both arms (spec §6)",
    )
    sp.add_argument("--methods", nargs="+", default=["all_at_once", "step_by_step", "binary_search"])
    sp.add_argument("--api-key", help="passed explicitly; env fallback is documented, not used")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--n-boot", type=int, default=2000)
    sp.add_argument("--out-dir", default="runs/baselines")
    sp.set_defaults(fn=cmd_baselines)

    sp = sub.add_parser("audit", help="MATU label spot-audit (spec §7.6)")
    sp.add_argument("--subset", choices=MATU_SUBSETS, default="autogen_mmlu")
    sp.add_argument("--judges", nargs=3, default=["mock", "mock", "mock"])
    sp.add_argument("--api-key")
    sp.add_argument("--n", type=int, default=100)
    sp.add_argument("--out-dir", default="runs/audit")
    sp.set_defaults(fn=cmd_audit)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
