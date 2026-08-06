"""Type-rule validation gate (Part C §2).

Runs the AG rules against HC's parsed types. ≥90% agreement is required before
the rules may be used on AG at all, so this exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..typing.validate import audit_sample, validate
from ._shared import add_common, load_records


def build_parser() -> argparse.ArgumentParser:
    p = add_common(argparse.ArgumentParser(prog="masattr typecheck", description=__doc__))
    p.add_argument("--audit-out", help="write the deterministic 100-step manual-audit sample here")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args)
    if "hc" not in records:
        raise SystemExit("typecheck needs the hc subset: it is the only corpus with parsed types")

    report = validate(records["hc"], subset="hc")
    print(report.render())
    print()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "typecheck.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (out / "typecheck.md").write_text(report.render() + "\n", encoding="utf-8")

    if args.audit_out:
        pool = [r for recs in records.values() for r in recs]
        Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_out).write_text(
            json.dumps(audit_sample(pool, n=100, seed=args.seed), indent=2), encoding="utf-8"
        )
        print(f"wrote 100-step audit sample to {args.audit_out}")

    if not report.passes:
        print(
            "\nGATE FAILED: rules do not reproduce HC's parsed types at 90%. "
            "Per Part C §2 this is the trigger to escalate to an LLM classifier — "
            "after a 100-step manual audit, not before."
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
