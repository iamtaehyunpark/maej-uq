"""E7 — surrogate-intrinsic baseline (Part C §6).

A proxy LM's per-step mean logprob and mean token entropy over ``content``, fed
through the *same* attribution rules as the judge field. Frozen logs do not
carry the generating model's own distributions, so this is the closest thing to
an intrinsic signal that exists here — and it is expected to be weak. Its job is
to show the judge field is not merely tracking fluency.
"""

from __future__ import annotations

import argparse

from ..baselines.surrogate import MockProxyLM, ProxyLM, entropy_scores, score_records
from ..eval.scorers import render
from ._shared import (
    add_common,
    attribution_table,
    emit,
    flatten,
    held_aside_keys,
    load_records,
    open_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    p = add_common(argparse.ArgumentParser(prog="masattr e7", description=__doc__))
    p.add_argument(
        "--proxy-lm",
        default="proxy_lm",
        help="mock | <hf model id> | proxy_lm (from specs/judge.json)",
    )
    p.add_argument("--device")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="the surrogate is uncalibrated, so first-crossing needs an explicit "
        "threshold; argmin is the rule to read here",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = open_manifest("e7_surrogate", args)
    records = flatten(load_records(args))

    spec = args.proxy_lm
    if spec == "proxy_lm":
        from .. import specs as spec_files

        spec_files.require_role("proxy_lm")
        spec = spec_files.role_id("proxy_lm")
    lm = MockProxyLM() if spec == "mock" else ProxyLM(spec, device=args.device)
    rows, logprob_scores = score_records(records, lm)
    ent_scores = entropy_scores(rows, records)

    held = held_aside_keys(records, args.seed)
    results: dict = {"proxy_lm": getattr(lm, "model_id", args.proxy_lm), "arms": {}}
    blocks = []
    for arm, scores in (("mean_logprob", logprob_scores), ("mean_entropy", ent_scores)):
        for subset in sorted({r.subset for r in records}):
            subset_records = [r for r in records if r.subset == subset]
            sub_scores = {k: v for k, v in scores.items() if k.startswith(f"{subset}/")}
            table, _ = attribution_table(
                subset_records,
                sub_scores,
                threshold=args.threshold,
                held=held,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            label = f"{arm} · subset={subset}"
            results["arms"][label] = table
            blocks.append(render(table, f"— surrogate {label}"))

    md = "\n".join(
        [
            "# E7 — surrogate-intrinsic baseline",
            "",
            "> Uncalibrated by construction: there is nothing principled to calibrate a "
            "proxy LM's logprob against here. Read the argmin row; first-crossing "
            "depends on a threshold this signal does not have.",
            "",
            "\n\n---\n\n".join(blocks),
        ]
    )
    emit(manifest, results, md, args.out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
