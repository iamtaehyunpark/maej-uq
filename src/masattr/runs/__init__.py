"""One module per experiment, argparse only (spec v2 Part B rule 3)."""

from . import (
    baselines,
    e0_field,
    e1_primary,
    e2_readout,
    e3_rules,
    e4_typing,
    e5_evidence,
    e6_judges,
    e7_surrogate,
    e9_uniformity,
    judge,
    load,
    retype,
    smoke,
    typecheck,
)

#: Experiment manifest order (spec v3 Part C §7). E8 (success-control) is gated behind an
#: explicit owner decision and is deliberately absent — see Part D.
ORDER = (
    "load",
    "typecheck",
    "retype",
    "smoke",
    "judge",
    "e0",
    "e1",
    "e2",
    "e3",
    "e4",
    "e5",
    "e6",
    "e7",
    "e9",
)

COMMANDS = {
    "load": load,
    "typecheck": typecheck,
    "retype": retype,
    "smoke": smoke,
    "judge": judge,
    "baselines": baselines,
    "e0": e0_field,
    "e1": e1_primary,
    "e2": e2_readout,
    "e3": e3_rules,
    "e4": e4_typing,
    "e5": e5_evidence,
    "e6": e6_judges,
    "e7": e7_surrogate,
    "e9": e9_uniformity,
}

__all__ = ["COMMANDS", "ORDER"]
