"""E2 — readout ablation: P(True) logit vs verbalized number vs binary verdict.

All three readouts run under an *identical* prompt scaffold, so the only
difference between the rows is where the number comes from. That is what makes
this an ablation rather than three methods.
"""

from __future__ import annotations

import argparse

from ._shared import add_attribution_args, add_common, run_config_tables


def build_parser() -> argparse.ArgumentParser:
    return add_attribution_args(
        add_common(argparse.ArgumentParser(prog="masattr e2", description=__doc__))
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run_config_tables(args, "e2_readout", expect_axis="readout")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
