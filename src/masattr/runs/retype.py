"""Hierarchical typing: gate the plan/delegate splitter on HC, then apply it to AG.

Measured on the release, the rules split coordination / execute / final at
0.9935 but plan vs delegate at 0.4162 — below the 0.6934 majority-class
baseline. Per Part C §2 that is the trigger to escalate, and per the direction
decision the escalation is *hierarchical*: rules keep the coarse split, an LLM
takes only the plan/delegate sub-split inside coordination.

This runs in two halves, in order:

1. **Gate.** Run the splitter on HC coordination steps, where plan/delegate is
   parsed and therefore known. It must clear 0.90 *and* beat the majority-class
   baseline — a splitter that always says "plan" scores 0.69 and has learned
   nothing.
2. **Apply.** Only if the gate passes, re-split AG's classified coordination
   steps and write the retyped records to the cache the judge reads.

The splitter must be family-disjoint from the judge (Part C §Validity): typing
conditions the judge's evidence policy, so one family for both closes a loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..record import write_jsonl
from ..typing.refine import build_splitter, refine_records, validate_splitter
from ._shared import add_common, load_records, open_manifest


def build_parser() -> argparse.ArgumentParser:
    p = add_common(argparse.ArgumentParser(prog="masattr retype", description=__doc__))
    p.add_argument("--splitter", default="mock", help="mock | hf:<model_id>")
    p.add_argument("--judge", default="", help="the primary judge this must be disjoint from")
    p.add_argument("--sensitivity-judge", default="", help="the second judge family, likewise")
    p.add_argument("--device")
    p.add_argument("--out-cache", help="directory to write retyped records into")
    p.add_argument(
        "--force",
        action="store_true",
        help="apply the splitter to AG even if the HC gate fails (exploration only)",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = open_manifest("retype", args)
    manifest.record_models(
        type_classifier=args.splitter,
        judge=args.judge,
        sensitivity_judge=args.sensitivity_judge,
    )

    records = load_records(args)
    if "hc" not in records:
        raise SystemExit(
            "retype gates the splitter on hc, the only subset where plan/delegate "
            "is parsed and therefore known — include it in --subsets"
        )

    splitter = build_splitter(
        args.splitter,
        judge_model=args.judge,
        sensitivity_judge=args.sensitivity_judge,
        device=args.device,
        seed=args.seed,
    )
    report = validate_splitter(records["hc"], splitter)
    print(report.render())
    print()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retype_gate.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    (out / "retype_gate.md").write_text(report.render() + "\n", encoding="utf-8")
    manifest.results = {"gate": report.to_dict()}

    if args.splitter == "mock":
        manifest.note("splitter=mock: a stand-in, never a reported number")

    if not report.passes and not args.force:
        manifest.write(out)
        print(
            "\nGATE FAILED: the splitter does not beat both the 0.90 gate and the "
            f"{report.majority_baseline:.4f} majority-class baseline on HC. AG types "
            "were not rewritten — licensing AG typing on a splitter that has not "
            "shown it can split is the circularity the gate exists to prevent."
        )
        return 1

    applied = 0
    if "alg" in records:
        if args.splitter == "mock" and not args.force:
            print(
                "\nRefusing to write AG types from the mock splitter; pass a real "
                "--splitter, or --force for a dry run."
            )
        else:
            retyped, applied = refine_records(records["alg"], splitter)
            cache = Path(args.out_cache or args.cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            write_jsonl(retyped, cache / "alg.jsonl")
            print(f"\nre-split {applied} AG coordination steps → {cache / 'alg.jsonl'}")

    manifest.results["applied_to_alg_steps"] = applied
    manifest.write(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
