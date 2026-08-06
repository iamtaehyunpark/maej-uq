"""E6 — judge-family sensitivity.

The same field computed by a second judge family. If the attribution
numbers track the judge rather than the method, the contribution is a judge
choice, not an estimator.
"""

from __future__ import annotations

import argparse

from ._shared import add_attribution_args, add_common, run_config_tables


def build_parser() -> argparse.ArgumentParser:
    return add_attribution_args(
        add_common(argparse.ArgumentParser(prog="masattr e6", description=__doc__))
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run_config_tables(args, "e6_judges", expect_axis="judge")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
