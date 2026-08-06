"""E5 — evidence ablation, including the hindsight ceiling figure.

Three policies: ``plain`` (prefix only), ``typed`` (prefix plus the
within-trajectory rescue for near-empty execute steps), and ``hindsight`` (the
whole trajectory as context for every step).

Hindsight is not a method — it is the ceiling. It tells you how much of the gap
to perfect attribution is the judge's reasoning versus the information the
prefix-conditional setting withholds by construction.
"""

from __future__ import annotations

import argparse

from ._shared import add_attribution_args, add_common, run_config_tables


def build_parser() -> argparse.ArgumentParser:
    return add_attribution_args(
        add_common(argparse.ArgumentParser(prog="masattr e5", description=__doc__))
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run_config_tables(args, "e5_evidence", expect_axis="policy")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
