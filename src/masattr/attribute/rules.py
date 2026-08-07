"""Attribution rules (spec v2 Part C §5).

Primary: **first crossing** on the type-normalised ``p_t`` — the earliest step
whose calibrated score falls below the frozen threshold; the fault agent is the
owner of that step. This encodes the causal reading of a failed trajectory: the
decisive error is the first one, and everything after it is contamination.

Ablations: ``argmin``; ``changepoint`` (binary segmentation, one frozen
hyperparameter); ``agent_first`` (per-agent max ``p`` selects the agent, then
first-crossing inside that agent's steps).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..judge.score import StepScore
from ..typing.normalize import is_orchestrator

#: Frozen changepoint hyperparameter: minimum segment length, in steps. One
#: knob, fixed here, so the ablation cannot be tuned into a win.
CHANGEPOINT_MIN_SEG = 2


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
            "pred_agent": self.agent,
            "pred_step": self.step,
            "score": self.score,
            **self.detail,
        }


Threshold = float | Mapping[str, float]


def _p(scores: Sequence[StepScore]) -> list[float]:
    return [s.p for s in scores]


def _thr(threshold: Threshold, type_norm: str) -> float:
    """Resolve the crossing threshold for one step.

    §5 says a step crosses *its* calibrated threshold, so a mapping of per-type
    thresholds is the typed arm and a bare float is the global-threshold arm.
    A type absent from the mapping falls back to its ``""`` entry.
    """
    if isinstance(threshold, Mapping):
        return float(threshold.get(type_norm, threshold.get("", 0.5)))
    return float(threshold)


def _at(scores: Sequence[StepScore], i: int, method: str, extra: dict | None = None) -> Attribution:
    s = scores[i]
    return Attribution(s.key, method, s.step_idx, s.agent, s.p, extra or {})


def first_crossing(scores: Sequence[StepScore], threshold: Threshold) -> Attribution:
    """Earliest step below ``threshold``; argmin if the trajectory never crosses.

    The fallback matters: every Who&When trajectory failed, so a no-crossing
    trajectory means the judge missed the failure, not that there wasn't one.
    Declining to answer would silently drop those files from the denominator.
    """
    if not scores:
        return Attribution("", "first_crossing", None, None)
    for i, s in enumerate(scores):
        if s.p < _thr(threshold, s.type_norm):
            return _at(scores, i, "first_crossing", {"crossed": True})
    i = int(np.argmin(_p(scores)))
    return _at(scores, i, "first_crossing", {"crossed": False, "fallback": "argmin"})


def argmin(scores: Sequence[StepScore], threshold: Threshold = 0.0) -> Attribution:
    if not scores:
        return Attribution("", "argmin", None, None)
    return _at(scores, int(np.argmin(_p(scores))), "argmin")


def changepoint(scores: Sequence[StepScore], threshold: Threshold = 0.0) -> Attribution:
    """Binary segmentation: the split maximising ``mean(before) − mean(after)``.

    Returns the first step of the worse segment. Unlike argmin this is robust to
    one noisy assessment, which is the point of having it as an ablation.
    """
    if not scores:
        return Attribution("", "changepoint", None, None)
    p = np.asarray(_p(scores))
    n = p.size
    if n < 2 * CHANGEPOINT_MIN_SEG:
        return _at(scores, int(np.argmin(p)), "changepoint", {"degenerate": True})
    best_k, best_gap = CHANGEPOINT_MIN_SEG, -np.inf
    for k in range(CHANGEPOINT_MIN_SEG, n - CHANGEPOINT_MIN_SEG + 1):
        gap = float(p[:k].mean() - p[k:].mean())
        if gap > best_gap:
            best_k, best_gap = k, gap
    return _at(scores, best_k, "changepoint", {"gap": best_gap, "min_seg": CHANGEPOINT_MIN_SEG})


def agent_first(scores: Sequence[StepScore], threshold: Threshold) -> Attribution:
    """Two-stage: select the agent whose *best* step is still worst, then run
    first-crossing inside that agent's steps.

    A different question from the step-first rules — "who is failing?" rather
    than "where did it break?" — and the two diverge when an agent contributes
    many mediocre steps but no single catastrophic one.
    """
    if not scores:
        return Attribution("", "agent_first", None, None)
    by_agent: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(scores):
        by_agent[s.agent].append(i)
    p = _p(scores)
    selector = {a: max(p[i] for i in idxs) for a, idxs in by_agent.items()}
    worst = min(selector, key=lambda a: selector[a])
    idxs = by_agent[worst]
    chosen = next(
        (i for i in idxs if p[i] < _thr(threshold, scores[i].type_norm)),
        min(idxs, key=lambda i: p[i]),
    )
    return _at(
        scores,
        chosen,
        "agent_first",
        {
            "selected_agent": worst,
            "agent_selector_max_p": selector[worst],
            "n_agent_steps": len(idxs),
            "n_agents": len(by_agent),
        },
    )


METHODS: dict[str, Callable[..., Attribution]] = {
    "first_crossing": first_crossing,
    "argmin": argmin,
    "changepoint": changepoint,
    "agent_first": agent_first,
}
PRIMARY = "first_crossing"


def attribute(
    scores_by_file: dict[str, list[StepScore]], *, threshold: Threshold, method: str = PRIMARY
) -> dict[str, Attribution]:
    fn = METHODS[method]
    out = {}
    for key, rows in scores_by_file.items():
        rows = sorted(rows, key=lambda s: s.step_idx)
        a = fn(rows, threshold)
        a.key = key
        out[key] = a
    return out


# --- disagreement analysis (Part C §5) --------------------------------------


@dataclass(slots=True)
class DisagreementRow:
    stratum: str
    n: int
    n_step: int
    n_agent: int

    @property
    def step_rate(self) -> float:
        return self.n_step / self.n if self.n else float("nan")

    @property
    def agent_rate(self) -> float:
        return self.n_agent / self.n if self.n else float("nan")

    def to_dict(self) -> dict:
        return {
            "stratum": self.stratum,
            "n": self.n,
            "step_disagree": self.step_rate,
            "agent_disagree": self.agent_rate,
        }


def type_strata(
    scores_by_file: dict[str, list[StepScore]], preds: dict[str, Attribution]
) -> dict[str, str]:
    out = {}
    for key, att in preds.items():
        rows = {s.step_idx: s for s in scores_by_file.get(key, [])}
        s = rows.get(att.step) if att.step is not None else None
        out[key] = s.type_norm if s else "unknown"
    return out


def role_strata(preds: dict[str, Attribution]) -> dict[str, str]:
    return {
        k: ("orchestrator" if (a.agent and is_orchestrator(a.agent)) else "worker")
        for k, a in preds.items()
    }


def disagreement(
    a: dict[str, Attribution], b: dict[str, Attribution], *, strata: dict[str, str] | None = None
) -> list[DisagreementRow]:
    buckets: dict[str, list[tuple[Attribution, Attribution]]] = defaultdict(list)
    for key in a.keys() & b.keys():
        buckets[(strata or {}).get(key, "all")].append((a[key], b[key]))
        if strata:
            buckets["all"].append((a[key], b[key]))
    return [
        DisagreementRow(
            stratum=label,
            n=len(pairs),
            n_step=sum(1 for x, y in pairs if x.step != y.step),
            n_agent=sum(1 for x, y in pairs if x.agent != y.agent),
        )
        for label, pairs in sorted(buckets.items())
    ]


def render_disagreement(rows: Sequence[DisagreementRow], title: str = "") -> str:
    lines = [
        f"### Step-first vs agent-first disagreement {title}".rstrip(),
        "",
        "| stratum | n | step disagree | agent disagree |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.stratum} | {r.n} | {r.n_step} ({r.step_rate:.1%}) | "
            f"{r.n_agent} ({r.agent_rate:.1%}) |"
        )
    return "\n".join(lines)


# --- normalized position (the early-skew receipt, §4.4/§4.5) ----------------


def normalized_position(step: int | None, n_steps: int) -> float | None:
    """Where in the trajectory a step sits, on [0, 1]."""
    if step is None or n_steps <= 1:
        return None
    return step / (n_steps - 1)


def position_table(
    preds: Mapping[str, Attribution],
    gold: Mapping[str, tuple[str, int]],
    lengths: Mapping[str, int],
) -> dict[str, Any]:
    """Predicted vs gold normalized position, per rule.

    §4.4 reports the labels' early skew (normalized median ≈ 0.29–0.33) and §4.5
    argues that skew punishes argmin's downstream bias. That argument is only
    visible if the predictions' own position distribution is reported next to the
    labels'.
    """
    import statistics

    pred_pos, gold_pos, deltas = [], [], []
    for key, att in preds.items():
        n = lengths.get(key, 0)
        gp = normalized_position(gold.get(key, (None, None))[1], n)
        pp = normalized_position(att.step, n)
        if gp is None or pp is None:
            continue
        gold_pos.append(gp)
        pred_pos.append(pp)
        deltas.append(pp - gp)

    def summary(xs: list[float]) -> dict[str, float | None]:
        if not xs:
            return {"n": 0, "mean": None, "median": None}
        return {
            "n": len(xs),
            "mean": round(statistics.fmean(xs), 4),
            "median": round(statistics.median(xs), 4),
        }

    return {
        "gold": summary(gold_pos),
        "predicted": summary(pred_pos),
        "delta_pred_minus_gold": summary(deltas),
        "fraction_predicted_after_gold": (
            round(sum(1 for d in deltas if d > 0) / len(deltas), 4) if deltas else None
        ),
    }


def render_positions(rows: Mapping[str, dict], title: str = "") -> str:
    lines = [
        f"### Normalized position of the attributed step {title}".rstrip(),
        "",
        "| method | n | gold median | pred median | mean delta | predicted late |",
        "|---|---|---|---|---|---|",
    ]
    def num(value: float | None, spec: str = ".3f") -> str:
        return "—" if value is None else format(value, spec)

    for method, r in rows.items():
        g, pred, d = r["gold"], r["predicted"], r["delta_pred_minus_gold"]
        late = r["fraction_predicted_after_gold"]
        lines.append(
            f"| {method} | {g['n']} | {num(g['median'])} | {num(pred['median'])} | "
            f"{num(d['mean'], '+.3f')} | {num(late, '.1%')} |"
        )
    lines += [
        "",
        "> A rule biased toward downstream damage shows up here as a positive mean "
        "delta and a high 'predicted late' fraction, against labels whose own "
        "median sits early.",
    ]
    return "\n".join(lines)
