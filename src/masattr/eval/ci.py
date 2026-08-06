"""Bootstrap CIs and reliability diagrams (ported from paper 1).

Every Who&When number is reported with a bootstrap CI **over files** — the file
is the sampling unit (n=126/58), and resampling steps would treat one
trajectory's steps as independent draws and understate the interval.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


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
    point = statistic(units)
    if len(units) < 2:
        return CI(point, float("nan"), float("nan"), level, 0)
    rng = random.Random(seed)
    n = len(units)
    stats = []
    for _ in range(n_boot):
        v = statistic([units[rng.randrange(n)] for _ in range(n)])
        if not (isinstance(v, float) and math.isnan(v)):
            stats.append(v)
    if not stats:
        return CI(point, float("nan"), float("nan"), level, 0)
    stats.sort()
    a = (1 - level) / 2
    return CI(
        point,
        stats[max(int(a * len(stats)) - 1, 0)],
        stats[min(int((1 - a) * len(stats)), len(stats) - 1)],
        level,
        len(stats),
    )


@dataclass(slots=True)
class Reliability:
    bins: list[dict]
    ece: float
    mce: float
    brier: float
    n: int

    def to_dict(self) -> dict:
        return {"n": self.n, "ece": self.ece, "mce": self.mce, "brier": self.brier, "bins": self.bins}

    def render(self, title: str = "") -> str:
        lines = [
            f"### Reliability {title}".rstrip(),
            "",
            "| bin | n | mean p | observed | gap |",
            "|---|---|---|---|---|",
        ]
        for b in self.bins:
            if b["n"] == 0:
                continue
            lines.append(
                f"| [{b['lo']:.2f},{b['hi']:.2f}) | {b['n']} | {b['mean_p']:.3f} | "
                f"{b['frac_pos']:.3f} | {b['frac_pos'] - b['mean_p']:+.3f} |"
            )
        lines += ["", f"ECE={self.ece:.4f}  MCE={self.mce:.4f}  Brier={self.brier:.4f}  n={self.n}"]
        return "\n".join(lines)


def reliability(probs: Sequence[float], labels: Sequence[bool], n_bins: int = 10) -> Reliability:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if p.size == 0:
        return Reliability([], float("nan"), float("nan"), float("nan"), 0)
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece, mce = [], 0.0, 0.0
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (p >= lo) & ((p < hi) if i < n_bins - 1 else (p <= hi))
        n = int(mask.sum())
        if n == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0, "mean_p": float("nan"), "frac_pos": float("nan")})
            continue
        mean_p, frac = float(p[mask].mean()), float(y[mask].mean())
        gap = abs(frac - mean_p)
        ece += (n / p.size) * gap
        mce = max(mce, gap)
        bins.append({"lo": lo, "hi": hi, "n": n, "mean_p": mean_p, "frac_pos": frac})
    return Reliability(bins, ece, mce, float(((p - y.astype(float)) ** 2).mean()), int(p.size))


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Midpoint-rank AUROC. Used for diagnostics only — the primary metrics here
    are exact-match accuracies, not ranking."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int(y.size - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size)
    ss = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
