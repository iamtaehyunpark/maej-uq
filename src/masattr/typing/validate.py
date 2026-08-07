"""Rule validation against HC's parsed types (spec v3 Part C §2).

The gate runs **before any use of the rules on AG**: HC types are parsed from
the compound role, so they are known, and the rules must reproduce them at ≥90%
agreement. The confusion matrix is a reportable table, not a debug print — it is
the typing layer's credibility.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..record import TYPE_NORMS, Record
from .normalize import classify_steps, coverage

AGREEMENT_GATE = 0.90


@dataclass(slots=True)
class ValidationReport:
    subset: str
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
            "n_steps": self.n_steps,
            "agreement": self.agreement,
            "coverage": self.coverage,
            "gate": AGREEMENT_GATE,
            "passes_gate": self.passes,
            "confusion": self.confusion,
            "per_rule": self.per_rule,
        }

    def render(self) -> str:
        head = "| parsed \\ rule | " + " | ".join(TYPE_NORMS) + " |"
        sep = "|" + "---|" * (len(TYPE_NORMS) + 1)
        rows = [
            f"| **{ref}** | "
            + " | ".join(str(self.confusion.get(ref, {}).get(p, 0)) for p in TYPE_NORMS)
            + " |"
            for ref in TYPE_NORMS
        ]
        return "\n".join(
            [
                f"### Type rules vs parsed types — {self.subset}",
                f"n_steps={self.n_steps}  agreement={self.agreement:.3f}  "
                f"coverage={self.coverage:.3f}  gate={AGREEMENT_GATE:.2f} → "
                f"**{'PASS' if self.passes else 'FAIL'}**",
                "",
                head,
                sep,
                *rows,
            ]
        )


def validate(records: Sequence[Record], *, subset: str = "hc") -> ValidationReport:
    confusion = {t: {u: 0 for u in TYPE_NORMS} for t in TYPE_NORMS}
    per_rule: dict[str, Counter[str]] = {}
    agree = total = 0
    all_verdicts = []
    for rec in records:
        verdicts = classify_steps(rec.steps)
        all_verdicts.extend(verdicts)
        for step, v in zip(rec.steps, verdicts):
            if step.type_source != "parsed":
                continue
            total += 1
            confusion[step.type_norm][v.type_norm] += 1
            bucket = per_rule.setdefault(v.rule, Counter())
            bucket["n"] += 1
            if step.type_norm == v.type_norm:
                agree += 1
                bucket["correct"] += 1
    return ValidationReport(
        subset=subset,
        n_steps=total,
        agreement=(agree / total) if total else 0.0,
        coverage=coverage(all_verdicts),
        confusion=confusion,
        per_rule={k: dict(v) for k, v in per_rule.items()},
    )


def audit_sample(records: Iterable[Record], n: int = 100, seed: int = 0) -> list[dict]:
    """Deterministic sample for the 100-step manual audit that gates escalation
    to an LLM classifier."""
    pool = []
    for rec in records:
        for step, v in zip(rec.steps, classify_steps(rec.steps)):
            pool.append(
                {
                    "key": rec.key,
                    "idx": step.idx,
                    "agent": step.agent,
                    "role_raw": step.role_raw,
                    "content": step.content[:1200],
                    "rule_type": v.type_norm,
                    "rule": v.rule,
                    "parsed_type": step.type_norm if step.type_source == "parsed" else None,
                    "manual_label": None,
                }
            )
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]
