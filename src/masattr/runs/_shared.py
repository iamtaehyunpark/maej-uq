"""Shared argparse wiring and the two operations every experiment performs.

Kept deliberately small: each ``eN_*.py`` is a thin main that varies one axis
and calls into here. Anything that is not shared by at least two experiments
lives in the experiment file, not this one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .. import paths as paths_mod
from ..attribute.rules import METHODS, PRIMARY, ablation_methods, attribute
from ..eval.scorers import gold_map, score_all
from ..normalize.apply import apply_folds, thresholds_for
from ..normalize.fit import load_folds
from ..judge.client import build_client
from ..judge.score import PREFIX_BUDGET_CHARS, StepScore, by_file, load_scores, score_corpus
from ..loaders.whowhen_ag import load as load_alg
from ..loaders.whowhen_hc import load as load_hc
from .. import specs
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
        default="flag",
        choices=("fail", "flag", "drop"),
        help="what to do with the 5 released files that violate Part C §1's per-step "
        "asserts: flag (default — keep, flag, dual-report; counts hold), fail "
        "(refuse to load), drop (exclude, breaks the 126/58 count assert)",
    )
    p.add_argument(
        "--no-verify-specs",
        action="store_true",
        help="skip the frozen-artifact hash check (use only while iterating on prompts)",
    )
    return p


def add_judge_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument(
        "--judge",
        default="mock",
        help="mock | hf:<model_id> | a role from specs/judge.json "
        "(judge_primary, judge_secondary)",
    )
    p.add_argument("--device")
    p.add_argument("--readout", default="ptrue", choices=("ptrue", "verbalized", "binary"))
    p.add_argument("--policy", default="typed", choices=("typed", "plain", "hindsight"))
    p.add_argument("--with-gt", action="store_true", help="append the reference answer (both settings run for primary tables)")
    p.add_argument("--no-types", action="store_true", help="typing-off arm of E4")
    p.add_argument(
        "--no-subtask-pointer",
        action="store_true",
        help="E5 arm: withhold the assigned-subtask pointer from terse execute steps",
    )
    p.add_argument(
        "--no-peer-corroboration",
        action="store_true",
        help="E5 arm: withhold same-turn peer steps (direction decision §10(c))",
    )
    p.add_argument(
        "--prefix-window",
        type=int,
        help="E5 prefix-slice arm: judge step t against only the last N steps",
    )
    p.add_argument(
        "--prefix-budget-chars",
        type=int,
        default=PREFIX_BUDGET_CHARS,
        help="pre-registered truncation budget; over it, old execute detail is demoted",
    )
    return p


def resolve_model(value: str) -> str:
    """Turn a role name from ``specs/judge.json`` into a client spec.

    Anything that is not a declared role passes through unchanged, so ``mock``
    and explicit ``hf:`` ids still work. A role must be ``confirmed`` before it
    resolves — a reported number should not rest on a draft identity.
    """
    if not value:
        return value
    alias = {"primary": "judge_primary", "secondary": "judge_secondary"}.get(value, value)
    if alias in specs.ROLES:
        specs.require_role(alias)
        return specs.client_spec(alias)
    return value


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
    client = build_client(
        resolve_model(judge or args.judge),
        device=getattr(args, "device", None),
        seed=args.seed,
    )
    results = score_corpus(
        records,
        client,
        kind=readout if readout is not None else args.readout,
        policy=policy if policy is not None else args.policy,
        with_gt=with_gt if with_gt is not None else args.with_gt,
        use_types=(use_types if use_types is not None else not args.no_types),
        subtask_pointer=not getattr(args, "no_subtask_pointer", False),
        peer_corroboration=not getattr(args, "no_peer_corroboration", False),
        prefix_window=getattr(args, "prefix_window", None),
        budget_chars=getattr(args, "prefix_budget_chars", PREFIX_BUDGET_CHARS),
        out_path=out_path,
    )
    return by_file([s for ts in results for s in ts.scores])


def read_scores(path: str | Path) -> dict[str, list[StepScore]]:
    return by_file(load_scores(path))


#: The axes a score row records about how it was produced. Grouping on these is
#: what makes every ablation the same code with different inputs.
CONFIG_AXES = (
    "subset",
    "judge",
    "readout",
    "policy",
    "with_gt",
    "use_types",
    "subtask_pointer",
    "peer_corroboration",
    "prefix_window",
)


def config_of(row: StepScore) -> tuple:
    return tuple(getattr(row, a) for a in CONFIG_AXES)


def config_label(cfg: tuple) -> str:
    return " · ".join(f"{a}={v}" for a, v in zip(CONFIG_AXES, cfg))


def read_configs(
    paths: Sequence[str | Path], folds: Mapping[str, Any] | None = None, *, typed: bool = True
) -> dict[tuple, dict[str, list[StepScore]]]:
    """Read many score files and group rows by the config that produced them.

    Score rows carry their own provenance, so an ablation is just "score under
    two settings, then read both files" — no bookkeeping in the experiment.
    """
    out: dict[tuple, dict[str, list[StepScore]]] = {}
    for path in paths:
        for row in load_scores(path):
            out.setdefault(config_of(row), {}).setdefault(row.key, []).append(row)
    for grouped in out.values():
        for rows in grouped.values():
            rows.sort(key=lambda s: s.step_idx)
        if folds:
            apply_folds(grouped, folds, typed=typed, strict=False)
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
    threshold: float = 0.0,
    per_file: Mapping[str, Any] | None = None,
    methods: Iterable[str] | None = None,
    rule_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    held: Iterable[str] = (),
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Returns ``(scores_by_method, predictions_by_method)``."""
    gold = gold_map(records)
    usable = {k: v for k, v in scores_by_file.items() if k in gold and v}
    table: dict[str, dict] = {}
    preds: dict[str, dict] = {}
    for method in list(methods or [PRIMARY, *ablation_methods()]):
        p = attribute(
            usable,
            threshold=threshold,
            method=method,
            per_file=per_file,
            **dict((rule_kwargs or {}).get(method, {})),
        )
        preds[method] = p
        table[method] = score_all(
            p, gold, records, held_aside=held, n_boot=n_boot, seed=seed
        )
    return table, preds


def held_aside_keys(records: Sequence[Record], seed: int) -> set[str]:
    """No held-aside slice under leave-one-out normalization.

    Every file is already normalized under statistics fit without it, so there
    is no subset that saw the fitting data and needs excluding. Kept as a hook
    so the slice machinery stays uniform.
    """
    return set()


def primary_row(table: dict[str, dict], primary: str = PRIMARY, slice_name: str = "all") -> dict:
    return table.get(primary, {}).get(f"exact/{slice_name}", {})


# --- output -----------------------------------------------------------------


def run_config_tables(
    args: argparse.Namespace,
    name: str,
    *,
    expect_axis: str | Sequence[str] | None = None,
    methods: Iterable[str] | None = None,
) -> int:
    """The body every attribution experiment shares.

    Read score files, group them by the config that produced them, and emit one
    attribution table per config — all four rules, both scorers, every
    pre-registered slice. ``expect_axis`` names the axis the experiment claims to
    vary; if the inputs do not actually vary it, the run stops rather than
    reporting a one-row "ablation".
    """
    from ..attribute.rules import (
        disagreement,
        position_table,
        render_disagreement,
        render_positions,
        role_strata,
        type_strata,
    )
    from ..eval.scorers import render

    manifest = open_manifest(name, args)
    records = flatten(load_records(args))
    by_key = {r.key: r for r in records}

    typed_norm = not getattr(args, "pooled_normalization", False)
    folds = load_folds(args.folds) if getattr(args, "folds", None) else None
    per_file = (
        thresholds_for(folds, typed=not getattr(args, "global_threshold", False))
        if folds
        else {}
    )
    threshold = float(args.threshold) if getattr(args, "threshold", None) is not None else 0.0
    if not folds and getattr(args, "threshold", None) is None:
        raise SystemExit(
            "no normalization: pass --folds (E0 writes them) or an explicit "
            "--threshold. Scoring raw judge output against a threshold picked "
            "here would apply statistics fit on the files being scored."
        )
    primary, registered = registered_rule(args)
    manifest.rule_provenance = specs.rule_provenance()
    manifest.record_anomalies(records)
    manifest.note(
        f"primary rule {primary!r}, fixed by specs/rule_directive.md "
        f"(hash {manifest.rule_provenance})"
    )

    configs = read_configs(args.scores, folds, typed=typed_norm)
    if not configs:
        raise SystemExit(f"no score rows found in {args.scores}")
    varied = varied_axes(configs)
    wanted = (expect_axis,) if isinstance(expect_axis, str) else tuple(expect_axis or ())
    if wanted and not (set(wanted) & set(varied)):
        listed = wanted[0] if len(wanted) == 1 else " / ".join(wanted)
        raise SystemExit(
            f"{name} varies {listed}, but every score file has the same value. "
            "Score under both settings first; a one-row ablation is not an "
            f"ablation. (varied axes here: {varied or 'none'})"
        )

    methods = list(methods) if methods else [PRIMARY, *ablation_methods()]
    lengths = {r.key: r.n_steps for r in records}
    held = held_aside_keys(records, args.seed)
    results: dict[str, Any] = {
        "primary_rule": primary,
        "rule_provenance": manifest.rule_provenance,
        "registered_criteria": registered,
        "typed_normalization": typed_norm,
        "typed_thresholds": not getattr(args, "global_threshold", False),
        "normalized": bool(folds),
        "n_folds": len(folds or {}),
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
            per_file=per_file,
            methods=methods,
            rule_kwargs={PRIMARY: changepoint_kwargs(registered)},
            held=held,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        dis_type = disagreement(
            preds[primary], preds["agent_first"], strata=type_strata(scores, preds[primary])
        )
        dis_role = (
            disagreement(
                preds[primary], preds["agent_first"], strata=role_strata(preds[primary])
            )
            if subset == "hc"
            else []
        )
        positions = {
            m: position_table(p, gold_map(subset_records), lengths)
            for m, p in preds.items()
        }
        label = config_label(cfg)
        results["configs"][label] = {
            "positions": positions,
            "scores": table,
            "n_files": len(scores),
            "disagreement_by_type": [r.to_dict() for r in dis_type],
            "disagreement_by_role": [r.to_dict() for r in dis_role],
            "predictions": {
                m: {k: a.to_dict() for k, a in p.items()} for m, p in preds.items()
            },
        }
        block = [
            render(table, f"— {label}"),
            "",
            render_positions(positions),
            "",
            render_disagreement(dis_type, "(by step type)"),
        ]
        if dis_role:
            block += ["", render_disagreement(dis_role, "(orchestrator vs worker)")]
        blocks.append("\n".join(block))

    _ = by_key
    emit(manifest, results, f"# {name}\n\n" + "\n\n---\n\n".join(blocks), args.out_dir)
    return 0


def registered_rule(args: argparse.Namespace) -> tuple[str, dict]:
    """The primary rule and the registered parameters it runs under.

    The rule is fixed by ``specs/rule_directive.md`` — no experiment's outcome
    selects it. What still has to be registered is the changepoint fallback
    condition, because it is part of the rule's definition: choosing when the
    rule declines to find a regime, after seeing how often it does, would make
    the rule outcome-dependent.
    """
    registered = specs.criteria()
    if not getattr(args, "allow_draft_criteria", False):
        specs.require_status(
            "criteria",
            registered,
            "registered",
            "The changepoint fallback condition is part of the primary rule.",
        )
    return PRIMARY, registered


def changepoint_kwargs(registered: Mapping[str, Any]) -> dict[str, Any]:
    """Registered parameters for the primary rule."""
    from ..attribute.rules import (
        CHANGEPOINT_BOUNDARY_FALLBACK,
        CHANGEPOINT_MIN_CONTRAST,
        CHANGEPOINT_MIN_SEG,
    )

    return {
        "min_seg": int(registered.get("changepoint_min_seg", CHANGEPOINT_MIN_SEG)),
        "min_contrast": float(
            registered.get("changepoint_min_contrast", CHANGEPOINT_MIN_CONTRAST)
        ),
        "boundary_fallback": bool(
            registered.get("changepoint_boundary_fallback", CHANGEPOINT_BOUNDARY_FALLBACK)
        ),
    }


def add_attribution_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--scores", nargs="+", required=True, help="step-score JSONL(s)")
    p.add_argument("--folds", help="fold statistics JSON written by E0")
    p.add_argument(
        "--allow-draft-criteria",
        action="store_true",
        help="run against unregistered criteria (exploration only; never a result)",
    )
    p.add_argument("--threshold", type=float, help="override the fitted threshold")
    p.add_argument(
        "--pooled-normalization",
        action="store_true",
        help="E4 arm: one set of statistics for every type instead of per-type",
    )
    p.add_argument(
        "--global-threshold",
        action="store_true",
        help="E4 arm: one crossing threshold for every type",
    )
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
