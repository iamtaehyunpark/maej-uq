"""E3 — attribution-rule ablation.

The primary rule, ``changepoint_single``, against every demoted alternative over
the same field: ``argmin``; ``first_crossing`` on the leave-one-out threshold;
``changepoint`` (the same split chosen by an unnormalised mean gap, which
ablates the contrast statistic); ``agent_first``; and ``relative_crossing`` at
k ∈ {1.5, 2, 2.5}.

The k sweep is here and only here. ``RELATIVE_K`` is deliberately unregistered:
a rule demoted to an ablation row should show its sensitivity to its own
hyperparameter rather than rest on one value chosen off-camera.

Every table in this harness carries all of these rows already, so E3 adds
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
