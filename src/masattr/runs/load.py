"""Load Who&When and assert the pre-registered counts (spec v3 Part C §1)."""

from __future__ import annotations

import argparse
import json

from ..loaders._common import EXPECTED_TOTAL_STEPS, check_expectations, check_total_steps
from ..record import stats
from ._shared import add_common, load_records


def build_parser() -> argparse.ArgumentParser:
    p = add_common(argparse.ArgumentParser(prog="masattr load", description=__doc__))
    p.add_argument("--assert", dest="do_assert", action="store_true", help="check the pre-registered counts")
    p.add_argument("--out-jsonl", help="write unified records here (one file per subset)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args)

    out: dict = {}
    problems: list[str] = []
    for subset, recs in records.items():
        out[subset] = stats(recs)
        if args.do_assert:
            found = check_expectations(recs, subset, strict=False)
            out[subset]["violations"] = found
            problems += found
        if args.out_jsonl:
            from pathlib import Path

            from ..record import write_jsonl

            p = Path(args.out_jsonl)
            p.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(recs, p if len(records) == 1 else p.with_name(f"{subset}.jsonl"))

    if args.do_assert and {"alg", "hc"} <= set(records):
        found = check_total_steps(records["alg"], records["hc"], strict=False)
        out["total_steps"] = {
            "expected": EXPECTED_TOTAL_STEPS,
            "got": sum(r.n_steps for rs in records.values() for r in rs),
            "violations": found,
        }
        problems += found

    print(json.dumps(out, indent=2))
    if problems:
        print("\n" + "\n".join(f"VIOLATION: {p}" for p in problems))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
