"""Metrics: AUROC, AUARC, reliability/ECE, and bootstrap CIs (spec §8).

Pre-registered primaries:

* trajectory track — AUROC + AUARC of trajectory ``U`` against per-run correctness
* attribution track — exact-match agent-accuracy and step-accuracy on W&W

All Who&When numbers are reported with bootstrap CIs over *files* (n=126/58),
because the file is the sampling unit; bootstrapping over steps would understate
the interval by treating one trajectory's steps as independent draws.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# --- ranking metrics --------------------------------------------------------


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve, with midpoint ranks for ties.

    Convention: ``labels[i] is True`` means the positive class, and higher
    ``scores[i]`` should indicate the positive class. For the trajectory track we
    pass ``U`` against *incorrectness*, so a good uncertainty measure scores high
    on failures.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if s.size != y.size:
        raise ValueError(f"length mismatch: {s.size} scores vs {y.size} labels")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def accuracy_rejection_curve(
    uncertainty: Sequence[float], correct: Sequence[bool], n_points: int = 101
) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy as a function of rejection rate, rejecting most-uncertain first."""
    u = np.asarray(uncertainty, dtype=float)
    c = np.asarray(correct, dtype=bool)
    n = u.size
    if n == 0:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(-u, kind="mergesort")  # most uncertain first
    c_sorted = c[order]
    fracs = np.linspace(0.0, 1.0, n_points)
    accs = np.empty_like(fracs)
    for i, f in enumerate(fracs):
        n_keep = int(round(n * (1.0 - f)))
        accs[i] = c_sorted[n - n_keep :].mean() if n_keep > 0 else 1.0
    return fracs, accs


def auarc(uncertainty: Sequence[float], correct: Sequence[bool], n_points: int = 101) -> float:
    """Area under the accuracy-rejection curve."""
    fracs, accs = accuracy_rejection_curve(uncertainty, correct, n_points)
    if fracs.size == 0:
        return float("nan")
    return float(np.trapezoid(accs, fracs)) if hasattr(np, "trapezoid") else float(
        np.trapz(accs, fracs)
    )


def auarc_normalized(uncertainty: Sequence[float], correct: Sequence[bool]) -> float:
    """AUARC rescaled between the random-rejection and oracle curves.

    Raw AUARC is dominated by base accuracy, which makes cross-cell comparison
    misleading; the normalised form answers "how much of the achievable
    rejection gain did this signal capture?"
    """
    c = np.asarray(correct, dtype=bool)
    if c.size == 0 or c.all() or (~c).all():
        return float("nan")
    got = auarc(uncertainty, c)
    base = float(c.mean())
    oracle = auarc(np.where(c, 0.0, 1.0), c)
    if math.isclose(oracle, base):
        return float("nan")
    return (got - base) / (oracle - base)


# --- calibration quality ----------------------------------------------------


@dataclass(slots=True)
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    mean_p: float
    frac_pos: float


@dataclass(slots=True)
class ReliabilityDiagram:
    bins: list[ReliabilityBin]
    ece: float
    mce: float
    brier: float
    n: int

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "bins": [
                {"lo": b.lo, "hi": b.hi, "n": b.n, "mean_p": b.mean_p, "frac_pos": b.frac_pos}
                for b in self.bins
            ],
        }

    def render(self, title: str = "") -> str:
        lines = [f"### Reliability {title}".rstrip(), "", "| bin | n | mean p | observed | gap |", "|---|---|---|---|---|"]
        for b in self.bins:
            if b.n == 0:
                continue
            lines.append(
                f"| [{b.lo:.2f},{b.hi:.2f}) | {b.n} | {b.mean_p:.3f} | "
                f"{b.frac_pos:.3f} | {b.frac_pos - b.mean_p:+.3f} |"
            )
        lines.append("")
        lines.append(f"ECE={self.ece:.4f}  MCE={self.mce:.4f}  Brier={self.brier:.4f}  n={self.n}")
        return "\n".join(lines)


def reliability(
    probs: Sequence[float], labels: Sequence[bool], n_bins: int = 10
) -> ReliabilityDiagram:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if p.size == 0:
        return ReliabilityDiagram([], float("nan"), float("nan"), float("nan"), 0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p < hi) if i < n_bins - 1 else (p <= hi))
        n = int(mask.sum())
        if n == 0:
            bins.append(ReliabilityBin(lo, hi, 0, float("nan"), float("nan")))
            continue
        mean_p = float(p[mask].mean())
        frac = float(y[mask].mean())
        gap = abs(frac - mean_p)
        ece += (n / p.size) * gap
        mce = max(mce, gap)
        bins.append(ReliabilityBin(lo, hi, n, mean_p, frac))
    brier = float(((p - y.astype(float)) ** 2).mean())
    return ReliabilityDiagram(bins, ece, mce, brier, int(p.size))


# --- uncertainty on the estimates ------------------------------------------


@dataclass(slots=True)
class CI:
    point: float
    lo: float
    hi: float
    level: float = 0.95
    n_boot: int = 0

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"

    def to_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "level": self.level}


def bootstrap_ci(
    units: Sequence,
    statistic: Callable[[Sequence], float],
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap over ``units`` — one unit per *file* for W&W."""
    point = statistic(units)
    if len(units) < 2:
        return CI(point, float("nan"), float("nan"), level, 0)
    rng = random.Random(seed)
    n = len(units)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        v = statistic(sample)
        if not (isinstance(v, float) and math.isnan(v)):
            stats.append(v)
    if not stats:
        return CI(point, float("nan"), float("nan"), level, 0)
    stats.sort()
    a = (1.0 - level) / 2.0
    lo = stats[max(int(a * len(stats)) - 1, 0)]
    hi = stats[min(int((1 - a) * len(stats)), len(stats) - 1)]
    return CI(point, lo, hi, level, len(stats))


# --- Who&When attribution scorers (spec §6) ---------------------------------


def exact_agent_match(pred: str | None, gold: str | None) -> bool:
    """Primary scorer: case- and whitespace-insensitive exact match."""
    if pred is None or gold is None:
        return False
    from .loaders.whowhen import collapse_orchestrator

    return collapse_orchestrator(pred) == collapse_orchestrator(gold)


def substring_agent_match(pred: str | None, gold: str | None) -> bool:
    """Comparability scorer: reproduces the published-number regime."""
    if pred is None or gold is None:
        return False
    return pred.strip().lower() in gold.strip().lower() or gold.strip().lower() in pred.strip().lower()


def exact_step_match(pred: int | None, gold: int | None) -> bool:
    """Primary scorer: integer equality after int-cast."""
    if pred is None or gold is None:
        return False
    return int(pred) == int(gold)


def substring_step_match(pred: int | None, gold: int | None) -> bool:
    """Comparability scorer: the published substring test.

    Note the artifact this reproduces — ``"1" in "12"`` is a match — which is
    exactly why the exact-match scorer is primary and this one is a footnote.
    """
    if pred is None or gold is None:
        return False
    return str(pred) in str(gold) or str(gold) in str(pred)


SCORERS = {
    "exact": (exact_agent_match, exact_step_match),
    "substring": (substring_agent_match, substring_step_match),
}


@dataclass(slots=True)
class AttributionScore:
    scorer: str
    n: int
    agent_acc: float
    step_acc: float
    both_acc: float
    agent_ci: CI | None = None
    step_ci: CI | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scorer": self.scorer,
            "n": self.n,
            "agent_acc": self.agent_acc,
            "step_acc": self.step_acc,
            "both_acc": self.both_acc,
            "agent_ci": self.agent_ci.to_dict() if self.agent_ci else None,
            "step_ci": self.step_ci.to_dict() if self.step_ci else None,
            **self.extra,
        }


def score_attribution(
    pairs: Sequence[tuple[tuple[str | None, int | None], tuple[str | None, int | None]]],
    *,
    scorer: str = "exact",
    n_boot: int = 2000,
    seed: int = 0,
) -> AttributionScore:
    """Score ``[(pred_agent, pred_step), (gold_agent, gold_step)]`` pairs.

    One entry per *file*, so the bootstrap resamples files (spec §8).
    """
    agent_fn, step_fn = SCORERS[scorer]
    units = [
        (agent_fn(p[0], g[0]), step_fn(p[1], g[1]))
        for p, g in pairs
    ]
    if not units:
        return AttributionScore(scorer, 0, float("nan"), float("nan"), float("nan"))
    agent_acc = sum(a for a, _ in units) / len(units)
    step_acc = sum(s for _, s in units) / len(units)
    both = sum(a and s for a, s in units) / len(units)
    return AttributionScore(
        scorer=scorer,
        n=len(units),
        agent_acc=agent_acc,
        step_acc=step_acc,
        both_acc=both,
        agent_ci=bootstrap_ci(
            units, lambda u: sum(a for a, _ in u) / len(u), n_boot=n_boot, seed=seed
        ),
        step_ci=bootstrap_ci(
            units, lambda u: sum(s for _, s in u) / len(u), n_boot=n_boot, seed=seed + 1
        ),
    )
