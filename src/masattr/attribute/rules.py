"""Attribution rules.

Primary: **``changepoint_single``** — a two-regime mean-shift split of the
per-step score sequence. The decisive step is the first step of regime 2, and
the fault agent is its owner. The rule is fixed by
``specs/rule_directive.md``, not chosen by any experiment's outcome, and that
directive's hash is logged on every run.

Why this shape. A failed trajectory is not a sequence of independent bad steps;
it is a run that was going acceptably and then was not. A mean-shift split reads
exactly that — where the level of the field changes — and its answer is the
*boundary*, which is what "decisive step" means. It needs no threshold carried
in from other files, so nothing about it depends on a quantity estimated across
the corpus.

The split is chosen by a **contrast statistic**, not a raw mean difference:
a two-sample statistic scaled by pooled spread and segment sizes, so a split
isolating two steps at the end cannot outrank a genuine regime change simply by
having a large raw gap. When the best split is at a boundary, or its contrast is
below the registered minimum, the trajectory has no regime structure to find and
the rule **falls back to argmin**. Both conditions live in
``specs/criteria.json`` and must be registered before any attribution number is
computed.

Ablation rows (E3): ``argmin``; ``relative_crossing`` at k ∈ {1.5, 2, 2.5};
``first_crossing`` on the leave-one-out threshold; ``changepoint`` (the same
split chosen by an unnormalised mean gap, which ablates the contrast statistic);
``agent_first`` (per-agent selection, then localisation within that agent).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..judge.score import StepScore
from ..typing.normalize import is_orchestrator

#: Frozen changepoint hyperparameter: minimum segment length, in steps. One
#: knob, fixed here, so the ablation cannot be tuned into a win.
CHANGEPOINT_MIN_SEG = 2

#: Default k for ``relative_crossing``: how many of the trajectory's own standard
#: deviations below its own mean a step must fall. Deliberately **not**
#: registered — the rule is an E3 ablation row, and E3 sweeps the sensitivity set
#: below rather than resting on one value.
RELATIVE_K = 2.0

#: The k values E3 reports for ``relative_crossing``.
RELATIVE_K_SWEEP = (1.5, 2.0, 2.5)

#: Fallback defaults, used when the registered criteria are unavailable. The
#: registered file is what governs a reported run; these keep the rule callable
#: in isolation (tests, notebooks) without silently inventing a different rule.
CHANGEPOINT_MIN_CONTRAST = 1.0
CHANGEPOINT_BOUNDARY_FALLBACK = True


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

    Used by the demoted threshold-dependent rows. A mapping of per-type
    thresholds is the typed arm and a bare float is the global-threshold arm;
    a type absent from the mapping falls back to its ``""`` entry.
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


def changepoint_single(
    scores: Sequence[StepScore],
    threshold: Threshold = 0.0,
    *,
    min_seg: int = CHANGEPOINT_MIN_SEG,
    min_contrast: float = CHANGEPOINT_MIN_CONTRAST,
    boundary_fallback: bool = CHANGEPOINT_BOUNDARY_FALLBACK,
) -> Attribution:
    """Primary rule: two-regime mean-shift split; decisive = first step of regime 2.

    The contrast at split ``k`` is the two-sample statistic

        (mean(p[:k]) − mean(p[k:])) / (s_pooled · sqrt(1/k + 1/(n−k)))

    which rewards a large drop supported by enough steps on both sides, rather
    than any large raw gap. ``threshold`` is accepted and ignored: this rule
    takes nothing from outside the trajectory.

    Falls back to argmin when the trajectory is too short to have two regimes,
    when the best split sits at a boundary (isolating an endpoint rather than
    finding a regime), or when the best contrast is below the registered
    minimum. The fallback is recorded on the result, so a table can always say
    how often the primary rule actually found a regime.
    """
    if not scores:
        return Attribution("", "changepoint_single", None, None)
    p = np.asarray(_p(scores), dtype=float)
    n = p.size
    if n < 2 * min_seg:
        return _at(scores, int(np.argmin(p)), "changepoint_single",
                   {"fallback": "argmin", "reason": "too_short", "n": int(n)})

    scale = float(p.std(ddof=0))
    if scale <= 1e-12:
        # A trajectory with no variation at all has no regimes to find.
        return _at(scores, int(np.argmin(p)), "changepoint_single",
                   {"fallback": "argmin", "reason": "no_variation"})

    lo, hi = min_seg, n - min_seg
    best_k, best_c = lo, -np.inf
    for k in range(lo, hi + 1):
        before, after = p[:k], p[k:]
        var = (before.var(ddof=0) * k + after.var(ddof=0) * (n - k)) / n
        # Floor the pooled spread against the trajectory's own scale. A *perfect*
        # split has zero within-segment variance, which is the strongest possible
        # evidence of two regimes — without the floor it divides by zero and the
        # one split that should win is the one that gets skipped.
        sd = max(math.sqrt(var), 1e-6 * scale)
        c = float(
            (before.mean() - after.mean()) / (sd * math.sqrt(1.0 / k + 1.0 / (n - k)))
        )
        if c > best_c:
            best_k, best_c = k, c
    at_boundary = best_k in (lo, hi)
    if boundary_fallback and at_boundary:
        return _at(scores, int(np.argmin(p)), "changepoint_single",
                   {"fallback": "argmin", "reason": "boundary", "split": int(best_k),
                    "contrast": best_c})
    if best_c < min_contrast:
        return _at(scores, int(np.argmin(p)), "changepoint_single",
                   {"fallback": "argmin", "reason": "low_contrast", "split": int(best_k),
                    "contrast": best_c, "min_contrast": min_contrast})
    return _at(scores, best_k, "changepoint_single",
               {"contrast": best_c, "min_seg": min_seg, "split": int(best_k)})


def relative_crossing(
    scores: Sequence[StepScore], threshold: Threshold = 0.0, *, k: float = RELATIVE_K
) -> Attribution:
    """Earliest step falling ``k`` sd below the trajectory's own mean.

    An E3 ablation row. ``k`` is not registered: E3 sweeps
    ``RELATIVE_K_SWEEP`` and reports the sensitivity rather than resting on one
    value.

    Falls back to argmin when nothing stands out, for the same reason
    first_crossing does: every trajectory here failed, so declining to answer
    would drop files from the denominator rather than score a miss.
    """
    if not scores:
        return Attribution("", "relative_crossing", None, None)
    p = np.asarray(_p(scores), dtype=float)
    sd = float(p.std(ddof=0))
    if sd <= 0:
        return _at(scores, int(np.argmin(p)), "relative_crossing", {"degenerate": True})
    cut = float(p.mean()) - k * sd
    for i, v in enumerate(p):
        if v < cut:
            return _at(scores, i, "relative_crossing", {"crossed": True, "cut": cut, "k": k})
    i = int(np.argmin(p))
    return _at(scores, i, "relative_crossing", {"crossed": False, "fallback": "argmin", "cut": cut})


METHODS: dict[str, Callable[..., Attribution]] = {
    "changepoint_single": changepoint_single,
    "first_crossing": first_crossing,
    "argmin": argmin,
    "changepoint": changepoint,
    "agent_first": agent_first,
    "relative_crossing": relative_crossing,
}

#: Rules needing a threshold estimated across files.
THRESHOLD_DEPENDENT = ("first_crossing", "agent_first")

#: Rules that read only the trajectory in front of them.
THRESHOLD_FREE = ("changepoint_single", "argmin", "changepoint", "relative_crossing")

#: Fixed by ``specs/rule_directive.md``, not by any experiment's outcome.
PRIMARY = "changepoint_single"

#: Withdrawn from the reported ablation. ``agent_first`` selects the agent whose
#: *best* step is still worst, which systematically elects whoever contributed
#: fewest steps — one confident step is enough to clear an agent. Measured on the
#: corrected P(True) field it is the weakest of eight agent selectors on
#: algorithm-generated logs (0.294 agent accuracy against 0.484 for the best),
#: and loses to at least one other rule in every cell but one: hand-crafted step
#: accuracy with the answer hidden, 0.172 against 0.121, a margin of about three
#: logs on the noisiest subset. It stays implemented and callable by name; it is
#: no longer reported.
WITHDRAWN = ("agent_first",)


#: E3's rows: everything that is not the primary rule, with the relative-crossing
#: sensitivity sweep expanded.
def ablation_methods() -> list[str]:
    rows = [
        m for m in METHODS
        if m not in (PRIMARY, "relative_crossing") and m not in WITHDRAWN
    ]
    return rows + [f"relative_crossing@{k}" for k in RELATIVE_K_SWEEP]


def resolve_method(name: str) -> Callable[..., Attribution]:
    """Look up a rule, including parameterised names like ``relative_crossing@1.5``."""
    base, sep, arg = name.partition("@")
    fn = METHODS.get(base)
    if fn is None:
        raise KeyError(f"unknown attribution rule {name!r}; known: {sorted(METHODS)}")
    if not sep:
        return fn
    if base != "relative_crossing":
        raise KeyError(f"rule {base!r} takes no parameter, got {name!r}")
    k = float(arg)
    return lambda scores, threshold=0.0: relative_crossing(scores, threshold, k=k)


def attribute(
    scores_by_file: dict[str, list[StepScore]],
    *,
    threshold: Threshold = 0.0,
    method: str = PRIMARY,
    per_file: Mapping[str, Threshold] | None = None,
    **rule_kwargs: Any,
) -> dict[str, Attribution]:
    """Run one rule over every file.

    ``per_file`` carries each file's own threshold — under leave-one-out
    normalization the threshold is a property of the fold, not of the corpus, so
    a single global value would apply statistics fit *with* the file to the file
    itself.
    """
    fn = resolve_method(method)
    out = {}
    for key, rows in scores_by_file.items():
        rows = sorted(rows, key=lambda s: s.step_idx)
        thr = (per_file or {}).get(key, threshold)
        a = fn(rows, thr, **rule_kwargs)
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
