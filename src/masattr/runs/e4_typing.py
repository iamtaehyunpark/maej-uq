"""E4 — typing on/off: does the act-type layer earn its place?

The typing-off arm strips ``type_norm`` from the rendered evidence and the
rubric, leaving the judge with agent identity and content alone. If the tables
do not move, the typing layer is decoration and the paper should say so.
"""

from __future__ import annotations

import argparse

from ._shared import add_attribution_args, add_common, run_config_tables


def build_parser() -> argparse.ArgumentParser:
    return add_attribution_args(
        add_common(argparse.ArgumentParser(prog="masattr e4", description=__doc__))
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run_config_tables(args, "e4_typing", expect_axis="use_types")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
