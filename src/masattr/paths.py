"""Data location. No path is hard-coded anywhere else in the package.

The release ships one parquet per subset; a directory of per-trajectory JSON is
also accepted. Expected layout under the data root, in resolution order::

    <root>/who_and_when/Algorithm-Generated.parquet   # subset "alg", 126 rows
    <root>/who_and_when/Hand-Crafted.parquet          # subset "hc",   58 rows
    <root>/whowhen/Algorithm-Generated/*.json         # legacy per-file layout
    <root>/whowhen/Hand-Crafted/*.json

Resolve with ``--data-root``, or ``$MASATTR_DATA_ROOT``, or a JSON config
mapping subset names to explicit paths.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_ROOT = "MASATTR_DATA_ROOT"

DEFAULT_RELATIVE = {
    "alg": "who_and_when/Algorithm-Generated.parquet",
    "hc": "who_and_when/Hand-Crafted.parquet",
}

#: Alternate locations and layouts, tried in order when the default is absent.
ALTERNATES = {
    "alg": (
        "who_and_when/Algorithm-Generated",
        "whowhen/Algorithm-Generated.parquet",
        "whowhen/Algorithm-Generated",
        "Who&When/Algorithm-Generated.parquet",
        "Who&When/Algorithm-Generated",
        "Algorithm-Generated.parquet",
        "Algorithm-Generated",
    ),
    "hc": (
        "who_and_when/Hand-Crafted",
        "whowhen/Hand-Crafted.parquet",
        "whowhen/Hand-Crafted",
        "Who&When/Hand-Crafted.parquet",
        "Who&When/Hand-Crafted",
        "Hand-Crafted.parquet",
        "Hand-Crafted",
    ),
}


@dataclass
class Paths:
    root: Path
    overrides: dict[str, str] = field(default_factory=dict)

    def get(self, subset: str) -> Path:
        if subset in self.overrides:
            return Path(self.overrides[subset]).expanduser()
        p = self.root / DEFAULT_RELATIVE[subset]
        if p.exists():
            return p
        for alt in ALTERNATES.get(subset, ()):
            q = self.root / alt
            if q.exists():
                return q
        return p  # let the loader raise with the canonical path in its message

    def status(self) -> dict[str, dict]:
        return {k: {"path": str(self.get(k)), "exists": self.get(k).exists()} for k in DEFAULT_RELATIVE}


def resolve(config: str | Path | None = None, root: str | Path | None = None) -> Paths:
    overrides: dict[str, str] = {}
    cfg_root = None
    if config:
        blob = json.loads(Path(config).read_text(encoding="utf-8"))
        cfg_root = blob.get("root")
        overrides = {k: v for k, v in blob.items() if k != "root"}
    return Paths(
        root=Path(root or cfg_root or os.environ.get(ENV_ROOT) or "data").expanduser(),
        overrides=overrides,
    )
