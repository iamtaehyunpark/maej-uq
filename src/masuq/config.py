"""Dataset paths and run configuration.

Nothing in this package hard-codes a location. Point it at your data either with
a small JSON config or with ``MASUQ_DATA_ROOT``; the CLI accepts explicit paths
that override both.

Expected layout under the data root (matches the upstream repos' own layout)::

    <root>/matu/quick_start/results/conversation_logs_Math_qwen2.5.json
    <root>/matu/quick_start/results/conversation_logs_MMLU_Autogen_qwen2.5.json
    <root>/matu/quick_start/results/accuracy_dict_Math_qwen2.5.pkl
    <root>/matu/quick_start/results/accuracy_dict_MMLU_Autogen_qwen2.5.pkl
    <root>/whowhen/Algorithm-Generated/*.json
    <root>/whowhen/Hand-Crafted/*.json
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ENV_ROOT = "MASUQ_DATA_ROOT"

DEFAULT_RELATIVE = {
    "camel_math": "matu/quick_start/results/conversation_logs_Math_qwen2.5.json",
    "autogen_mmlu": "matu/quick_start/results/conversation_logs_MMLU_Autogen_qwen2.5.json",
    "camel_math_labels": "matu/quick_start/results/accuracy_dict_Math_qwen2.5.pkl",
    "autogen_mmlu_labels": "matu/quick_start/results/accuracy_dict_MMLU_Autogen_qwen2.5.pkl",
    "alg": "whowhen/Algorithm-Generated",
    "hc": "whowhen/Hand-Crafted",
}

#: Candidate filenames tried when the default is absent — upstream dumps have
#: been renamed more than once.
ALTERNATES: dict[str, tuple[str, ...]] = {
    "camel_math": (
        "matu/results/conversation_logs_Math_qwen2.5.json",
        "conversation_logs_Math_qwen2.5.json",
    ),
    "autogen_mmlu": (
        "matu/results/conversation_logs_MMLU_Autogen_qwen2.5.json",
        "conversation_logs_MMLU_Autogen_qwen2.5.json",
    ),
    "camel_math_labels": (
        "matu/results/accuracy_dict_Math_qwen2.5.pkl",
        "accuracy_dict_Math_qwen2.5.pkl",
    ),
    "autogen_mmlu_labels": (
        "matu/results/accuracy_dict_MMLU_Autogen_qwen2.5.pkl",
        "accuracy_dict_MMLU_Autogen_qwen2.5.pkl",
    ),
    "alg": ("Who&When/Algorithm-Generated", "Algorithm-Generated"),
    "hc": ("Who&When/Hand-Crafted", "Hand-Crafted"),
}


@dataclass
class Paths:
    root: Path
    overrides: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> Path:
        if key in self.overrides:
            return Path(self.overrides[key]).expanduser()
        p = self.root / DEFAULT_RELATIVE[key]
        if p.exists():
            return p
        for alt in ALTERNATES.get(key, ()):
            q = self.root / alt
            if q.exists():
                return q
        return p  # let the loader raise with the canonical path in the message

    def exists(self, key: str) -> bool:
        return self.get(key).exists()

    def status(self) -> dict[str, dict]:
        return {
            k: {"path": str(self.get(k)), "exists": self.get(k).exists()}
            for k in DEFAULT_RELATIVE
        }


def load_paths(config_file: str | Path | None = None, root: str | Path | None = None) -> Paths:
    """Resolve the data root from (in order) an explicit arg, a config file, the env."""
    overrides: dict[str, str] = {}
    cfg_root: str | None = None
    if config_file:
        blob = json.loads(Path(config_file).read_text(encoding="utf-8"))
        cfg_root = blob.get("root")
        overrides = {k: v for k, v in blob.items() if k != "root"}
    resolved = root or cfg_root or os.environ.get(ENV_ROOT) or "data"
    return Paths(root=Path(resolved).expanduser(), overrides=overrides)


@dataclass
class RunConfig:
    """Everything that has to be identical between a fit run and an apply run."""

    judge_model: str = "mock"
    judge_backend: str = "mock"  # mock | hf
    readout: str = "ptrue"  # ptrue | verbalized
    evidence_policy: str = "type_conditioned_v1"
    calibration_method: str = "percentile"
    aggregator: str = "noisy_or"
    attribution_method: str = "first_crossing"
    threshold: float | None = None  # None ⇒ take it from the frozen calibrator
    seed: int = 0
    n_boot: int = 2000

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
