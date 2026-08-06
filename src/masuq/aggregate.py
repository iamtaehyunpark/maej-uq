"""Trajectory-level uncertainty from per-step scores (spec §5).

Primary: **noisy-OR** over per-step failure probabilities,
``U = 1 − Π_t (1 − q_t)`` where ``q_t = 1 − p_t``. It encodes the assumption
that a trajectory fails if *any* step fails, and that step failures are
conditionally independent given the prefix. Neither is true — steps in a
trajectory are strongly dependent, and the product saturates toward 1 as ``T``
grows — so the length-normalised and max variants are reported alongside it as
the ablation, not as afterthoughts. On W&W-HC (up to 130 steps) plain noisy-OR
is near-saturated by construction; that is a property of the estimator, and it
is why the trajectory track is validated on MATU, where ``T`` is short.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

EPS = 1e-9


def noisy_or(p: Sequence[float]) -> float:
    """``1 − Π(1 − p_t)`` computed in log space over failure probabilities."""
    if len(p) == 0:
        return float("nan")
    log_survive = 0.0
    for pt in p:
        q = 1.0 - min(max(float(pt), EPS), 1.0 - EPS)  # q = P(step t is fine)
        log_survive += math.log(max(q, EPS))
    return 1.0 - math.exp(log_survive)


def noisy_or_uncertainty(p_correct: Sequence[float]) -> float:
    """Trajectory uncertainty from per-step *correctness* probabilities.

    ``p_t`` from the judge is P(step is correct); the failure probability is
    ``1 − p_t``, so ``U = 1 − Π p_t``.
    """
    if len(p_correct) == 0:
        return float("nan")
    log_ok = 0.0
    for pt in p_correct:
        log_ok += math.log(max(min(float(pt), 1.0 - EPS), EPS))
    return 1.0 - math.exp(log_ok)


def length_normalized(p_correct: Sequence[float]) -> float:
    """Geometric-mean form: removes the length saturation of plain noisy-OR."""
    if len(p_correct) == 0:
        return float("nan")
    log_ok = sum(math.log(max(min(float(pt), 1.0 - EPS), EPS)) for pt in p_correct)
    return 1.0 - math.exp(log_ok / len(p_correct))


def max_failure(p_correct: Sequence[float]) -> float:
    """Weakest-link: the single least-confident step."""
    if len(p_correct) == 0:
        return float("nan")
    return 1.0 - float(min(p_correct))


def mean_failure(p_correct: Sequence[float]) -> float:
    if len(p_correct) == 0:
        return float("nan")
    return 1.0 - float(np.mean(list(p_correct)))


def last_step(p_correct: Sequence[float]) -> float:
    """Final-step confidence alone — the cheapest possible baseline."""
    if len(p_correct) == 0:
        return float("nan")
    return 1.0 - float(p_correct[-1])


AGGREGATORS: dict[str, Callable[[Sequence[float]], float]] = {
    "noisy_or": noisy_or_uncertainty,
    "length_normalized": length_normalized,
    "max_failure": max_failure,
    "mean_failure": mean_failure,
    "last_step": last_step,
}

#: The pre-registered primary (spec §5).
PRIMARY = "noisy_or"


@dataclass(slots=True)
class TrajectoryU:
    key: str
    n_steps: int
    values: dict[str, float]
    label_correct: bool | None = None

    @property
    def primary(self) -> float:
        return self.values[PRIMARY]

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "n_steps": self.n_steps,
            "label_correct": self.label_correct,
            **{f"U_{k}": v for k, v in self.values.items()},
        }


def aggregate_trajectory(
    key: str,
    p_correct: Sequence[float],
    *,
    label_correct: bool | None = None,
    types: Sequence[str] | None = None,
) -> TrajectoryU:
    """Compute every aggregator for one trajectory.

    When ``types`` is given, an extra ``noisy_or_no_final`` variant excludes the
    ``final`` step: the final answer's own score is nearly a direct readout of
    task success, so keeping it in makes the aggregate look strong for a reason
    that has nothing to do with multi-agent structure.
    """
    values = {name: fn(p_correct) for name, fn in AGGREGATORS.items()}
    if types is not None and len(types) == len(p_correct):
        kept = [p for p, t in zip(p_correct, types) if t != "final"]
        values["noisy_or_no_final"] = (
            noisy_or_uncertainty(kept) if kept else float("nan")
        )
    return TrajectoryU(
        key=key, n_steps=len(p_correct), values=values, label_correct=label_correct
    )


def aggregate_corpus(
    grouped: dict[str, list],
    labels: dict[str, bool | None] | None = None,
    *,
    use_calibrated: bool = True,
) -> list[TrajectoryU]:
    """Aggregate ``{record_key: [StepScore, ...]}`` into per-trajectory U values."""
    out: list[TrajectoryU] = []
    for key, scores in grouped.items():
        scores = sorted(scores, key=lambda s: s.step_idx)
        p = [
            (s.p_cal if (use_calibrated and s.p_cal is not None) else s.p_raw) for s in scores
        ]
        out.append(
            aggregate_trajectory(
                key,
                p,
                label_correct=(labels or {}).get(key),
                types=[s.type_norm for s in scores],
            )
        )
    return out
