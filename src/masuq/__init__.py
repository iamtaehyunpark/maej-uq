"""MAS-UQ pilot.

Uncertainty quantification over frozen multi-agent-system trajectories: load
four log formats into one schema, normalise step types, judge every step
prefix-conditionally, calibrate once per type, then read the result two ways —
as trajectory-level uncertainty and as failure attribution.

Implements ``mas_uq_pilot_spec_v1.md``; module docstrings cite the section they
implement.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .schema import Record, Step, corpus_stats, read_jsonl, write_jsonl

__all__ = [
    "Record",
    "Step",
    "__version__",
    "corpus_stats",
    "read_jsonl",
    "write_jsonl",
]
