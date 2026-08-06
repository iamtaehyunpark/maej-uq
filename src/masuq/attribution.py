"""Failure attribution from per-step scores (spec §5).

Primary rule: **first crossing** — the earliest step whose calibrated ``p_t``
falls below the frozen threshold. It encodes the causal reading of a failed
trajectory: the decisive error is the first one, and everything after it is
downstream contamination. ``argmin`` (lowest-scoring step anywhere) and a
changepoint rule are the ablations, and the agent-first two-stage rule is the
structural alternative — aggregate per agent, pick the worst agent, then
localise within that agent's steps.

The step-first vs agent-first *disagreement* is itself a reportable quantity,
stratified by type and by orchestrator/worker (HC only), because where the two
readings diverge is where the multi-agent structure is doing the work.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(slots=True)
class Attribution:
    key: str
    method: str
    step: int | None
    agent: str | None
    score: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_pair(self) -> tuple[str | None, int | None]:
        return self.agent, self.step

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "method": self.method,
            "pred_step": self.step,
            "pred_agent": self.agent,
            "score": self.score,
            **self.detail,
        }


def _p(scores: Sequence, use_calibrated: bool) -> list[float]:
    return [
        (s.p_cal if (use_calibrated and s.p_cal is not None) else s.p_raw) for s in scores
    ]


def first_crossing(
    scores: Sequence, threshold: float, *, use_calibrated: bool = True
) -> Attribution:
    """Earliest step below ``threshold``. Falls back to argmin when none crosses."""
    if not scores:
        return Attribution("", "first_crossing", None, None)
    p = _p(scores, use_calibrated)
    for s, pt in zip(scores, p):
        if pt < threshold:
            return Attribution(
                s.key, "first_crossing", s.step_idx, s.agent, pt, {"crossed": True}
            )
    i = int(np.argmin(p))
    return Attribution(
        scores[i].key,
        "first_crossing",
        scores[i].step_idx,
        scores[i].agent,
        p[i],
        {"crossed": False, "fallback": "argmin"},
    )


def argmin_step(scores: Sequence, threshold: float = 0.0, *, use_calibrated: bool = True):
    """Lowest-scoring step anywhere in the trajectory."""
    if not scores:
        return Attribution("", "argmin", None, None)
    p = _p(scores, use_calibrated)
    i = int(np.argmin(p))
    return Attribution(scores[i].key, "argmin", scores[i].step_idx, scores[i].agent, p[i])


def changepoint(scores: Sequence, threshold: float = 0.0, *, use_calibrated: bool = True):
    """Largest drop in the running mean — where the trajectory's quality breaks.

    Scores the split point ``k`` by ``mean(p[:k]) − mean(p[k:])`` and returns the
    first step of the worse segment. Unlike argmin this is robust to a single
    noisy assessment.
    """
    if not scores:
        return Attribution("", "changepoint", None, None)
    p = np.asarray(_p(scores, use_calibrated))
    n = p.size
    if n < 2:
        return Attribution(scores[0].key, "changepoint", scores[0].step_idx, scores[0].agent, float(p[0]))
    best_k, best_gap = 1, -np.inf
    for k in range(1, n):
        gap = float(p[:k].mean() - p[k:].mean())
        if gap > best_gap:
            best_k, best_gap = k, gap
    s = scores[best_k]
    return Attribution(
        s.key, "changepoint", s.step_idx, s.agent, float(p[best_k]), {"gap": best_gap}
    )


def agent_first(
    scores: Sequence,
    threshold: float = 0.0,
    *,
    use_calibrated: bool = True,
    agent_stat: str = "mean",
) -> Attribution:
    """Two-stage: worst agent by aggregate score, then that agent's worst step.

    This asks a different question from the step-first rules — "who is failing?"
    rather than "where did it break?" — and the two answers disagree in exactly
    the cases where a weak agent contributes many mediocre steps but no single
    catastrophic one.
    """
    if not scores:
        return Attribution("", "agent_first", None, None)
    p = _p(scores, use_calibrated)
    by_agent: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(scores):
        by_agent[s.agent].append(i)

    def stat(idxs: list[int]) -> float:
        vals = [p[i] for i in idxs]
        if agent_stat == "min":
            return min(vals)
        if agent_stat == "noisy_or":
            from .aggregate import noisy_or_uncertainty

            return 1.0 - noisy_or_uncertainty(vals)
        return float(np.mean(vals))

    worst_agent = min(by_agent, key=lambda a: stat(by_agent[a]))
    idxs = by_agent[worst_agent]
    j = min(idxs, key=lambda i: p[i])
    s = scores[j]
    return Attribution(
        s.key,
        "agent_first",
        s.step_idx,
        s.agent,
        p[j],
        {
            "agent_stat": agent_stat,
            "agent_score": stat(idxs),
            "n_agent_steps": len(idxs),
            "n_agents": len(by_agent),
        },
    )


METHODS: dict[str, Callable[..., Attribution]] = {
    "first_crossing": first_crossing,
    "argmin": argmin_step,
    "changepoint": changepoint,
    "agent_first": agent_first,
}

PRIMARY = "first_crossing"


def attribute(
    grouped: dict[str, list],
    *,
    threshold: float,
    method: str = PRIMARY,
    use_calibrated: bool = True,
) -> dict[str, Attribution]:
    fn = METHODS[method]
    out: dict[str, Attribution] = {}
    for key, scores in grouped.items():
        scores = sorted(scores, key=lambda s: s.step_idx)
        a = fn(scores, threshold, use_calibrated=use_calibrated)
        a.key = key
        out[key] = a
    return out


# --- disagreement analysis (spec §5) ----------------------------------------


@dataclass(slots=True)
class DisagreementRow:
    stratum: str
    n: int
    n_disagree_step: int
    n_disagree_agent: int

    @property
    def step_rate(self) -> float:
        return self.n_disagree_step / self.n if self.n else float("nan")

    @property
    def agent_rate(self) -> float:
        return self.n_disagree_agent / self.n if self.n else float("nan")


def disagreement(
    a: dict[str, Attribution],
    b: dict[str, Attribution],
    *,
    strata: dict[str, str] | None = None,
) -> list[DisagreementRow]:
    """Compare two attribution methods, optionally stratified.

    ``strata`` maps record key → stratum label (e.g. the predicted step's type,
    or ``orchestrator``/``worker`` for HC).
    """
    buckets: dict[str, list[tuple[Attribution, Attribution]]] = defaultdict(list)
    for key in a.keys() & b.keys():
        label = (strata or {}).get(key, "all")
        buckets[label].append((a[key], b[key]))
        if strata:
            buckets["all"].append((a[key], b[key]))

    rows = []
    for label, pairs in sorted(buckets.items()):
        rows.append(
            DisagreementRow(
                stratum=label,
                n=len(pairs),
                n_disagree_step=sum(1 for x, y in pairs if x.step != y.step),
                n_disagree_agent=sum(1 for x, y in pairs if x.agent != y.agent),
            )
        )
    return rows


def orchestrator_strata(grouped: dict[str, list], attributions: dict[str, Attribution]) -> dict[str, str]:
    """Label each record by whether the attributed step belongs to the orchestrator."""
    from .typing_.classifier import is_orchestrator

    out: dict[str, str] = {}
    for key, att in attributions.items():
        out[key] = "orchestrator" if (att.agent and is_orchestrator(att.agent)) else "worker"
    return out


def type_strata(grouped: dict[str, list], attributions: dict[str, Attribution]) -> dict[str, str]:
    """Label each record by the normalised type of the attributed step."""
    out: dict[str, str] = {}
    for key, att in attributions.items():
        scores = {s.step_idx: s for s in grouped.get(key, [])}
        s = scores.get(att.step) if att.step is not None else None
        out[key] = s.type_norm if s else "unknown"
    return out


def render_disagreement(rows: Sequence[DisagreementRow], title: str = "") -> str:
    lines = [
        f"### Step-first vs agent-first disagreement {title}".rstrip(),
        "",
        "| stratum | n | step disagree | agent disagree |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.stratum} | {r.n} | {r.n_disagree_step} ({r.step_rate:.1%}) | "
            f"{r.n_disagree_agent} ({r.agent_rate:.1%}) |"
        )
    return "\n".join(lines)
