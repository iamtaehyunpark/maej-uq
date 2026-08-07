"""MAS-Attribution harness.

Failure attribution over frozen multi-agent trajectories: load Who&When into one
record type, normalise step act-types, judge every step prefix-conditionally,
normalise the scores per type by leave-one-file-out CV, then localise the
decisive mistake.

Implements ``docs/mas_attr_harness_spec_v3.md``. Module docstrings cite the
section they implement.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .record import Record, Step, read_jsonl, stats, write_jsonl

__all__ = ["Record", "Step", "__version__", "read_jsonl", "stats", "write_jsonl"]
