"""Shared argparse wiring and the two operations every experiment performs.

Kept deliberately small: each ``eN_*.py`` is a thin main that varies one axis
and calls into here. Anything that is not shared by at least two experiments
lives in the experiment file, not this one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import paths as paths_mod
from ..attribute.rules import METHODS, PRIMARY, attribute
from ..calib.fit import FrozenCalibration
from ..calib.apply import apply_to, held_aside
from ..eval.scorers import gold_map, score_all
from ..judge.client import build_client
from ..judge.score import StepScore, by_file, load_scores, score_corpus
from ..loaders.whowhen_ag import load as load_alg
from ..loaders.whowhen_hc import load as load_hc
from ..manifest import Manifest, start
from ..record import Record, read_jsonl, write_jsonl

SUBSETS = ("alg", "hc")


def add_common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--data-root", help="root containing whowhen/{Algorithm-Generated,Hand-Crafted}")
    p.add_argument("--config", help="JSON file with explicit subset paths")
    p.add_argument("--cache-dir", default="runs/cache", help="where loaded records are cached")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=list(SUBSETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out-dir", default="runs/out")
    p.add_argument(
        "--anomaly-policy",
        default="fail",
        choices=("fail", "flag", "drop"),
        help="what to do with the 5 released files that violate Part C §1's own "
        "asserts: fail (spec-literal, will not load), flag (keep + dual-report), "
        "drop (exclude, breaks the 126/58 count assert)",
    )
    p.add_argument(
        "--no-verify-specs",
        action="store_true",
        help="skip the frozen-artifact hash check (use only while iterating on prompts)",
    )
    return p


def add_judge_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--judge", default="mock", help="mock | hf:<model_id>")
    p.add_argument("--device")
    p.add_argument("--readout", default="ptrue", choices=("ptrue", "verbalized", "binary"))
    p.add_argument("--policy", default="typed", choices=("typed", "plain", "hindsight"))
    p.add_argument("--with-gt", action="store_true", help="append the reference answer (both settings run for primary tables)")
    p.add_argument("--no-types", action="store_true", help="typing-off arm of E4")
    return p


def open_manifest(name: str, args: argparse.Namespace) -> Manifest:
    return start(name, args, verify_specs=not getattr(args, "no_verify_specs", False))


# --- records ----------------------------------------------------------------


def load_records(args: argparse.Namespace) -> dict[str, list[Record]]:
    """Load each requested subset, using the JSONL cache when present."""
    resolved = paths_mod.resolve(args.config, args.data_root)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    out: dict[str, list[Record]] = {}
    for subset in args.subsets:
        cache = cache_dir / f"{subset}.jsonl" if cache_dir else None
        if cache and cache.exists() and not args.no_cache:
            out[subset] = list(read_jsonl(cache))
            continue
        loader = load_alg if subset == "alg" else load_hc
        records, _ = loader(
            resolved.get(subset), anomaly_policy=getattr(args, "anomaly_policy", "fail")
        )
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(records, cache)
        out[subset] = records
    return out


def flatten(records_by_subset: dict[str, list[Record]]) -> list[Record]:
    return [r for subset in sorted(records_by_subset) for r in records_by_subset[subset]]


# --- scoring ----------------------------------------------------------------


def score(
    records: Sequence[Record],
    args: argparse.Namespace,
    *,
    readout: str | None = None,
    policy: str | None = None,
    with_gt: bool | None = None,
    use_types: bool | None = None,
    judge: str | None = None,
    out_path: str | Path | None = None,
) -> dict[str, list[StepScore]]:
    """Score a corpus, defaulting every axis to the CLI args."""
    client = build_client(judge or args.judge, device=getattr(args, "device", None), seed=args.seed)
    results = score_corpus(
        records,
        client,
        kind=readout if readout is not None else args.readout,
        policy=policy if policy is not None else args.policy,
        with_gt=with_gt if with_gt is not None else args.with_gt,
        use_types=(use_types if use_types is not None else not args.no_types),
        out_path=out_path,
    )
    return by_file([s for ts in results for s in ts.scores])


def read_scores(path: str | Path, cal: FrozenCalibration | None = None) -> dict[str, list[StepScore]]:
    grouped = by_file(load_scores(path))
    if cal:
        for rows in grouped.values():
            apply_to(rows, cal)
    return grouped


#: The axes a score row records about how it was produced. Grouping on these is
#: what makes every ablation the same code with different inputs.
CONFIG_AXES = ("subset", "judge", "readout", "policy", "with_gt", "use_types")


def config_of(row: StepScore) -> tuple:
    return tuple(getattr(row, a) for a in CONFIG_AXES)


def config_label(cfg: tuple) -> str:
    return " · ".join(f"{a}={v}" for a, v in zip(CONFIG_AXES, cfg))


def read_configs(
    paths: Sequence[str | Path], cal: FrozenCalibration | None = None
) -> dict[tuple, dict[str, list[StepScore]]]:
    """Read many score files and group rows by the config that produced them.

    Score rows carry their own provenance, so an ablation is just "score under
    two settings, then read both files" — no bookkeeping in the experiment.
    """
    out: dict[tuple, dict[str, list[StepScore]]] = {}
    for path in paths:
        rows = load_scores(path)
        if cal:
            apply_to(rows, cal)
        for row in rows:
            out.setdefault(config_of(row), {}).setdefault(row.key, []).append(row)
    for grouped in out.values():
        for rows in grouped.values():
            rows.sort(key=lambda s: s.step_idx)
    return out


def varied_axes(configs: Iterable[tuple]) -> list[str]:
    """Which axes actually differ across the given configs."""
    configs = list(configs)
    return [
        axis
        for i, axis in enumerate(CONFIG_AXES)
        if len({c[i] for c in configs}) > 1
    ]


# --- attribution + scoring table --------------------------------------------


def attribution_table(
    records: Sequence[Record],
    scores_by_file: dict[str, list[StepScore]],
    *,
    threshold: float,
    methods: Iterable[str] = tuple(METHODS),
    held: Iterable[str] = (),
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Returns ``(scores_by_method, predictions_by_method)``."""
    gold = gold_map(records)
    usable = {k: v for k, v in scores_by_file.items() if k in gold and v}
    table: dict[str, dict] = {}
    preds: dict[str, dict] = {}
    for method in methods:
        p = attribute(usable, threshold=threshold, method=method)
        preds[method] = p
        table[method] = score_all(
            p, gold, records, held_aside=held, n_boot=n_boot, seed=seed
        )
    return table, preds


def held_aside_keys(records: Sequence[Record], seed: int) -> set[str]:
    try:
        return held_aside(records, seed=seed)
    except ValueError:
        return set()  # corpus too small (fixtures); dual reporting simply collapses


def primary_row(table: dict[str, dict], slice_name: str = "all") -> dict:
    return table.get(PRIMARY, {}).get(f"exact/{slice_name}", {})


# --- output -----------------------------------------------------------------


def run_config_tables(
    args: argparse.Namespace,
    name: str,
    *,
    expect_axis: str | None = None,
    methods: Iterable[str] = tuple(METHODS),
) -> int:
    """The body every attribution experiment shares.

    Read score files, group them by the config that produced them, and emit one
    attribution table per config — all four rules, both scorers, every
    pre-registered slice. ``expect_axis`` names the axis the experiment claims to
    vary; if the inputs do not actually vary it, the run stops rather than
    reporting a one-row "ablation".
    """
    from ..attribute.rules import disagreement, render_disagreement, role_strata, type_strata
    from ..eval.scorers import render

    manifest = open_manifest(name, args)
    records = flatten(load_records(args))
    by_key = {r.key: r for r in records}

    cal = FrozenCalibration.load(args.calibration) if getattr(args, "calibration", None) else None
    if cal:
        manifest.calibration_hash = cal.content_hash()
    threshold = resolve_threshold(args, cal)

    configs = read_configs(args.scores, cal)
    if not configs:
        raise SystemExit(f"no score rows found in {args.scores}")
    varied = varied_axes(configs)
    if expect_axis and expect_axis not in varied:
        raise SystemExit(
            f"{name} varies '{expect_axis}', but every score file has the same "
            f"{expect_axis}. Score under both settings first; a one-row ablation "
            f"is not an ablation. (varied axes here: {varied or 'none'})"
        )

    held = held_aside_keys(records, args.seed)
    results: dict[str, Any] = {
        "threshold": threshold,
        "calibrated": bool(cal),
        "calibration_hash": manifest.calibration_hash,
        "held_aside": sorted(held),
        "varied_axes": varied,
        "configs": {},
    }
    blocks: list[str] = []

    for cfg, scores in sorted(configs.items(), key=lambda kv: str(kv[0])):
        subset = cfg[0]
        subset_records = [r for r in records if r.subset == subset]
        table, preds = attribution_table(
            subset_records,
            scores,
            threshold=threshold,
            methods=methods,
            held=held,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        dis_type = disagreement(
            preds[PRIMARY], preds["agent_first"], strata=type_strata(scores, preds[PRIMARY])
        )
        dis_role = (
            disagreement(
                preds[PRIMARY], preds["agent_first"], strata=role_strata(preds[PRIMARY])
            )
            if subset == "hc"
            else []
        )
        label = config_label(cfg)
        results["configs"][label] = {
            "scores": table,
            "n_files": len(scores),
            "disagreement_by_type": [r.to_dict() for r in dis_type],
            "disagreement_by_role": [r.to_dict() for r in dis_role],
            "predictions": {
                m: {k: a.to_dict() for k, a in p.items()} for m, p in preds.items()
            },
        }
        block = [render(table, f"— {label}"), "", render_disagreement(dis_type, "(by step type)")]
        if dis_role:
            block += ["", render_disagreement(dis_role, "(orchestrator vs worker)")]
        blocks.append("\n".join(block))

    _ = by_key
    emit(manifest, results, f"# {name}\n\n" + "\n\n---\n\n".join(blocks), args.out_dir)
    return 0


def resolve_threshold(args: argparse.Namespace, cal: FrozenCalibration | None) -> float:
    """The first-crossing threshold, which must come from the calibration corpus.

    Picking it on Who&When would leak: it is the one free parameter of the
    primary rule, and the corpus it is tuned on is the corpus being scored.
    """
    if getattr(args, "threshold", None) is not None:
        return float(args.threshold)
    if getattr(args, "threshold_file", None):
        return float(json.loads(Path(args.threshold_file).read_text())["threshold"])
    if cal is not None:
        return cal.threshold
    raise SystemExit(
        "no threshold: pass --calibration (E0's frozen map carries one), "
        "--threshold-file, or an explicit --threshold. Choosing it on Who&When "
        "would leak."
    )


def add_attribution_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--scores", nargs="+", required=True, help="step-score JSONL(s)")
    p.add_argument("--calibration", help="frozen calibration JSON from E0")
    p.add_argument("--threshold", type=float)
    p.add_argument("--threshold-file", help="threshold.json written by E0")
    return p


def emit(manifest: Manifest, results: dict[str, Any], markdown: str, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest.results = results
    manifest.write(out)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out / "results.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return out
