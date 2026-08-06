"""E3 — attribution-rule ablation.

first_crossing (primary) vs argmin vs changepoint vs agent_first, over the same
field. Every table in this harness already carries all four rules, so E3 adds
nothing to the scoring; what it adds is the head-to-head summary and the
step-first vs agent-first disagreement, stratified by step type and — on HC —
by orchestrator vs worker.

The disagreement is the interesting number, not the win. Where the two readings
diverge is where the multi-agent structure is doing work that a flat "which step
scored lowest" cannot see.
"""

from __future__ import annotations

import argparse

from ._shared import add_attribution_args, add_common, run_config_tables


def build_parser() -> argparse.ArgumentParser:
    return add_attribution_args(
        add_common(argparse.ArgumentParser(prog="masattr e3", description=__doc__))
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run_config_tables(args, "e3_rules")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
