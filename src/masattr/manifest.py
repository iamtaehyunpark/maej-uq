"""Run manifest (spec v2 Part B rule 6, Part E).

Every experiment writes one. It records the commit, the frozen artifact hashes,
the calibration hash, and the full argument set — so ``(commit, manifest)`` is
enough to re-run, and any number can be traced to the prompts and maps that
produced it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import specs


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            dirty = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


@dataclass
class Manifest:
    experiment: str
    args: dict[str, Any] = field(default_factory=dict)
    commit: str = field(default_factory=git_commit)
    spec_hashes: dict[str, str] = field(default_factory=specs.hashes)
    calibration_hash: str = ""
    #: {role: model family}, so the Part C §Validity disjointness constraints are
    #: auditable from the manifest rather than promised in prose.
    model_families: dict[str, str] = field(default_factory=dict)
    #: File ids of the released records that violate Part C §1's per-step asserts.
    anomalous_files: dict[str, list[str]] = field(default_factory=dict)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    results: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def record_models(self, **roles: str) -> None:
        """Log each role's model family and check the pairs that must be disjoint."""
        from .models import check_disjoint, families

        self.model_families = {**self.model_families, **families(**roles)}
        for a, b in (("type_classifier", "judge"), ("judge", "labeling_judge")):
            if roles.get(a) and roles.get(b):
                problem = check_disjoint(a, roles[a], b, roles[b], strict=False)
                if problem:
                    self.note(f"VALIDITY: {problem}")

    def record_anomalies(self, records) -> None:
        """Log the file ids carrying record-level anomalies, per subset."""
        out: dict[str, list[str]] = {}
        for r in records:
            if r.is_anomalous:
                out.setdefault(r.subset, []).append(r.file_id)
        self.anomalous_files = {k: sorted(v) for k, v in out.items()}

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "manifest.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return path


def start(experiment: str, args: Any, *, verify_specs: bool = True) -> Manifest:
    """Open a manifest and check the frozen artifacts have not drifted."""
    if verify_specs:
        specs.verify(strict=True)
    payload = vars(args) if hasattr(args, "__dict__") else dict(args)
    return Manifest(experiment=experiment, args={k: str(v) for k, v in payload.items()})
