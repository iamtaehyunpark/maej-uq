"""E1 — primary attribution vs the three Who&When baselines (Part C §7).

Emits the primary table for every scoring config found in the input files:
all four rules, exact and substring scorers, and every pre-registered slice
(all / excl-flagged / excl-held-aside / excl-both), with bootstrap CIs over
files.

Both GT settings and both judges belong in this table, so pass the score files
for all of them at once — they group themselves by the provenance each row
carries. Baseline predictions come from ``masattr baselines`` and are compared
against these numbers; they are a separate run because they call a different
model family.
"""

from __future__ import annotations

import argparse

from ._shared import add_attribution_args, add_common, run_config_tables


def build_parser() -> argparse.ArgumentParser:
    return add_attribution_args(
        add_common(argparse.ArgumentParser(prog="masattr e1", description=__doc__))
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run_config_tables(args, "e1_primary")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
