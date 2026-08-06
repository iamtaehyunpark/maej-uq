"""Classifier validation against known types (spec §2).

The classifier is only licensed on the untyped subsets if it reproduces types we
already know. So we run it on MATU-AutoGen (``native`` types) and W&W-HC
(``parsed`` types) and report a confusion matrix plus overall agreement. Gate:
≥90% agreement.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..schema import Record, TYPE_NORMS
from .classifier import classify_trajectory, coverage

AGREEMENT_GATE = 0.90
COVERAGE_GATE = 0.90


@dataclass(slots=True)
class ValidationReport:
    subset: str
    reference_source: str
    n_steps: int
    agreement: float
    coverage: float
    confusion: dict[str, dict[str, int]]
    per_rule: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def passes(self) -> bool:
        return self.agreement >= AGREEMENT_GATE

    def to_dict(self) -> dict:
        return {
            "subset": self.subset,
            "reference_source": self.reference_source,
            "n_steps": self.n_steps,
            "agreement": self.agreement,
            "coverage": self.coverage,
            "passes_gate": self.passes,
            "confusion": self.confusion,
            "per_rule": self.per_rule,
        }

    def render(self) -> str:
        """Markdown table — this is a reportable artifact, not a debug print."""
        labels = [t for t in TYPE_NORMS]
        head = "| ref \\ pred | " + " | ".join(labels) + " |"
        sep = "|" + "---|" * (len(labels) + 1)
        rows = []
        for ref in labels:
            cells = [str(self.confusion.get(ref, {}).get(pred, 0)) for pred in labels]
            rows.append(f"| **{ref}** | " + " | ".join(cells) + " |")
        verdict = "PASS" if self.passes else "FAIL"
        return "\n".join(
            [
                f"### Classifier vs {self.reference_source} types — {self.subset}",
                f"n_steps={self.n_steps}  agreement={self.agreement:.3f}  "
                f"coverage={self.coverage:.3f}  gate={AGREEMENT_GATE:.2f} → **{verdict}**",
                "",
                head,
                sep,
                *rows,
            ]
        )


def validate_against_known(
    records: Sequence[Record],
    *,
    reference_source: str,
    subset: str | None = None,
) -> ValidationReport:
    """Compare classifier output to steps whose ``type_source == reference_source``."""
    confusion: dict[str, dict[str, int]] = {t: {u: 0 for u in TYPE_NORMS} for t in TYPE_NORMS}
    per_rule: dict[str, Counter[str]] = {}
    agree = 0
    total = 0
    all_verdicts = []
    for rec in records:
        verdicts = classify_trajectory(rec.steps)
        all_verdicts.extend(verdicts)
        for step, v in zip(rec.steps, verdicts):
            if step.type_source != reference_source:
                continue
            total += 1
            confusion[step.type_norm][v.type_norm] += 1
            bucket = per_rule.setdefault(v.rule, Counter())
            bucket["n"] += 1
            if step.type_norm == v.type_norm:
                agree += 1
                bucket["correct"] += 1
    return ValidationReport(
        subset=subset or (records[0].subset if records else "?"),
        reference_source=reference_source,
        n_steps=total,
        agreement=(agree / total) if total else 0.0,
        coverage=coverage(all_verdicts),
        confusion={k: dict(v) for k, v in confusion.items()},
        per_rule={k: dict(v) for k, v in per_rule.items()},
    )


def audit_sample(records: Iterable[Record], n: int = 100, seed: int = 0) -> list[dict]:
    """Deterministic sample of steps for the 100-step manual audit (spec §2)."""
    import random

    rng = random.Random(seed)
    pool = []
    for rec in records:
        verdicts = classify_trajectory(rec.steps)
        for step, v in zip(rec.steps, verdicts):
            pool.append(
                {
                    "key": rec.key,
                    "idx": step.idx,
                    "agent": step.agent,
                    "role_raw": step.role_raw,
                    "content": step.content[:1200],
                    "predicted": v.type_norm,
                    "rule": v.rule,
                    "reference": step.type_norm if step.type_source != "classified" else None,
                    "manual_label": None,
                }
            )
    rng.shuffle(pool)
    return pool[:n]
