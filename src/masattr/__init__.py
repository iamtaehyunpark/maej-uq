"""MAS-Attribution harness.

Failure attribution over frozen multi-agent trajectories: load Who&When into one
record type, normalise step act-types, judge every step prefix-conditionally,
calibrate per type with maps fit once on paper 1's single-agent corpus, then
localise the decisive mistake by first crossing.

Implements ``docs/mas_attr_harness_spec_v2.md``. Module docstrings cite the
section they implement.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .record import Record, Step, read_jsonl, stats, write_jsonl

__all__ = ["Record", "Step", "__version__", "read_jsonl", "stats", "write_jsonl"]
