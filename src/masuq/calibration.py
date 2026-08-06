"""Per-type calibration, fit once and frozen (spec §4).

The coherence claim of the pilot is that *one* typed calibration map serves both
tracks. So the map is fit exactly once — on MATU-AutoGen, the only subset with
both native types and per-run labels — then frozen and applied unchanged to
MATU-CAMEL (trajectory track) and Who&When (attribution track). Experiment 0
tests whether that transfer holds before any attribution number is looked at.

**Stated assumption.** The fit target is per-*run* correctness, but the unit
being calibrated is a per-*step* score. v1 propagates the run label to each of
its steps: a step in a correct run is treated as correct. This is a weak and
biased label (a correct run can contain a bad step), and it is the reason the
v1 default is the rank-preserving percentile map rather than Platt — a monotone
map cannot be misled about ordering by label noise, only about level.

Three methods:

* ``percentile`` (v1 default) — per-type empirical percentile → observed
  positive rate at that percentile. Monotone, non-parametric, no optimiser.
* ``platt`` — per-type logistic on the logit of the raw score.
* ``isotonic`` — per-type PAV fit; most flexible, most prone to overfit at
  small per-type n.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .schema import TYPE_NORMS

METHODS = ("percentile", "platt", "isotonic")

#: Types with fewer than this many fitting points fall back to the pooled map.
MIN_PER_TYPE_N = 200


class CalibrationError(RuntimeError):
    pass


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# --- per-type maps ----------------------------------------------------------


@dataclass(slots=True)
class PercentileMap:
    """Monotone step function: sorted raw scores → smoothed positive rate."""

    knots_x: list[float]
    knots_y: list[float]

    def apply(self, p: float) -> float:
        if not self.knots_x:
            return p
        return float(np.interp(p, self.knots_x, self.knots_y))

    def to_dict(self) -> dict:
        return {"kind": "percentile", "knots_x": self.knots_x, "knots_y": self.knots_y}


@dataclass(slots=True)
class PlattMap:
    a: float
    b: float

    def apply(self, p: float) -> float:
        return _sigmoid(self.a * _logit(p) + self.b)

    def to_dict(self) -> dict:
        return {"kind": "platt", "a": self.a, "b": self.b}


@dataclass(slots=True)
class IsotonicMap:
    thresholds: list[float]
    values: list[float]

    def apply(self, p: float) -> float:
        if not self.thresholds:
            return p
        return float(np.interp(p, self.thresholds, self.values))

    def to_dict(self) -> dict:
        return {"kind": "isotonic", "thresholds": self.thresholds, "values": self.values}


def _map_from_dict(d: Mapping[str, Any]):
    kind = d["kind"]
    if kind == "percentile":
        return PercentileMap(list(d["knots_x"]), list(d["knots_y"]))
    if kind == "platt":
        return PlattMap(float(d["a"]), float(d["b"]))
    if kind == "isotonic":
        return IsotonicMap(list(d["thresholds"]), list(d["values"]))
    raise CalibrationError(f"unknown map kind {kind!r}")


# --- fitting ----------------------------------------------------------------


def fit_percentile(p: np.ndarray, y: np.ndarray, n_bins: int = 20) -> PercentileMap:
    """Equal-count bins in score space → observed positive rate, made monotone."""
    order = np.argsort(p, kind="mergesort")
    p_s, y_s = p[order], y[order].astype(float)
    n = p_s.size
    bins = min(n_bins, max(n // 25, 2))
    edges = np.linspace(0, n, bins + 1).astype(int)
    xs, ys = [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        xs.append(float(p_s[lo:hi].mean()))
        ys.append(float(y_s[lo:hi].mean()))
    # Enforce monotonicity via PAV so the map cannot invert the judge's ordering.
    ys = _pav(np.asarray(ys), np.ones(len(ys))).tolist()
    return PercentileMap(xs, ys)


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: least-squares isotonic regression."""
    y = y.astype(float).copy()
    w = w.astype(float).copy()
    n = y.size
    level_y: list[float] = []
    level_w: list[float] = []
    level_n: list[int] = []
    for i in range(n):
        level_y.append(y[i])
        level_w.append(w[i])
        level_n.append(1)
        while len(level_y) > 1 and level_y[-2] > level_y[-1]:
            wy = level_w[-2] * level_y[-2] + level_w[-1] * level_y[-1]
            ww = level_w[-2] + level_w[-1]
            nn = level_n[-2] + level_n[-1]
            level_y[-2:] = [wy / ww]
            level_w[-2:] = [ww]
            level_n[-2:] = [nn]
    out = np.empty(n)
    k = 0
    for val, cnt in zip(level_y, level_n):
        out[k : k + cnt] = val
        k += cnt
    return out


def fit_platt(p: np.ndarray, y: np.ndarray, *, iters: int = 200, lr: float = 0.1) -> PlattMap:
    """Logistic on logit(score), fit by plain gradient descent (no scipy needed)."""
    x = np.array([_logit(float(v)) for v in p])
    t = y.astype(float)
    a, b = 1.0, 0.0
    n = max(x.size, 1)
    for _ in range(iters):
        z = a * x + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = pred - t
        a -= lr * float((err * x).mean())
        b -= lr * float(err.mean())
    return PlattMap(float(a), float(b))


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicMap:
    order = np.argsort(p, kind="mergesort")
    xs = p[order].astype(float)
    ys = _pav(y[order].astype(float), np.ones(y.size))
    # Deduplicate x for interp stability.
    keep_x, keep_y = [], []
    for xv, yv in zip(xs, ys):
        if keep_x and math.isclose(xv, keep_x[-1]):
            keep_y[-1] = float(yv)
        else:
            keep_x.append(float(xv))
            keep_y.append(float(yv))
    return IsotonicMap(keep_x, keep_y)


_FITTERS = {"percentile": fit_percentile, "platt": fit_platt, "isotonic": fit_isotonic}


# --- the frozen calibrator --------------------------------------------------


@dataclass
class TypedCalibrator:
    """Per-type calibration maps plus a pooled fallback, fit once then frozen."""

    method: str = "percentile"
    maps: dict[str, Any] = field(default_factory=dict)
    pooled: Any | None = None
    frozen: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- fit ---------------------------------------------------------------

    def fit(
        self,
        scores: Sequence[float],
        types: Sequence[str],
        labels: Sequence[bool],
        *,
        fit_on: str = "",
    ) -> "TypedCalibrator":
        if self.frozen:
            raise CalibrationError(
                "this calibrator is frozen; spec §4 fits once on MATU-AutoGen and "
                "applies the result unchanged to every other subset"
            )
        if self.method not in METHODS:
            raise CalibrationError(f"unknown method {self.method!r}; known: {METHODS}")
        p = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=bool)
        ty = np.asarray(list(types))
        if not (p.size == y.size == ty.size):
            raise CalibrationError(f"length mismatch {p.size}/{ty.size}/{y.size}")
        if p.size == 0:
            raise CalibrationError("no fitting data")

        fitter = _FITTERS[self.method]
        self.pooled = fitter(p, y)
        self.maps = {}
        per_type_n: dict[str, int] = {}
        for t in TYPE_NORMS:
            mask = ty == t
            n = int(mask.sum())
            per_type_n[t] = n
            if n >= MIN_PER_TYPE_N and 0 < int(y[mask].sum()) < n:
                self.maps[t] = fitter(p[mask], y[mask])
        self.provenance = {
            "method": self.method,
            "fit_on": fit_on,
            "n": int(p.size),
            "base_rate": float(y.mean()),
            "per_type_n": per_type_n,
            "types_with_own_map": sorted(self.maps),
            "min_per_type_n": MIN_PER_TYPE_N,
            "label_target": "per-run correctness propagated to each step",
        }
        return self

    def freeze(self) -> "TypedCalibrator":
        self.frozen = True
        return self

    # -- apply -------------------------------------------------------------

    def transform_one(self, p: float, type_norm: str) -> float:
        m = self.maps.get(type_norm) or self.pooled
        if m is None:
            raise CalibrationError("calibrator has not been fit")
        return float(min(max(m.apply(float(p)), 0.0), 1.0))

    def transform(self, scores: Sequence[float], types: Sequence[str]) -> list[float]:
        return [self.transform_one(p, t) for p, t in zip(scores, types)]

    def apply_to_scores(self, step_scores: Iterable) -> None:
        """Write ``p_cal`` onto :class:`~masuq.judge.harness.StepScore` objects."""
        for s in step_scores:
            s.p_cal = self.transform_one(s.p_raw, s.type_norm)

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "frozen": self.frozen,
            "pooled": self.pooled.to_dict() if self.pooled else None,
            "maps": {k: v.to_dict() for k, v in self.maps.items()},
            "provenance": self.provenance,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TypedCalibrator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        c = cls(method=d["method"], frozen=bool(d.get("frozen", True)))
        c.pooled = _map_from_dict(d["pooled"]) if d.get("pooled") else None
        c.maps = {k: _map_from_dict(v) for k, v in d.get("maps", {}).items()}
        c.provenance = d.get("provenance", {})
        return c


# --- threshold selection ----------------------------------------------------


def choose_threshold(
    probs: Sequence[float],
    labels: Sequence[bool],
    *,
    objective: str = "f1",
) -> float:
    """Pick the attribution threshold on the *calibration* corpus, not on W&W.

    Attribution's first-crossing rule needs a threshold, and taking it from the
    corpus being scored would be leakage. It is therefore chosen here, on the
    same MATU-AutoGen fit set, and frozen alongside the maps (spec §4/§5).
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if p.size == 0:
        return 0.5
    cands = np.unique(np.round(p, 3))
    best, best_v = 0.5, -1.0
    for c in cands:
        pred_bad = p < c  # below threshold ⇒ predicted incorrect
        tp = int((pred_bad & ~y).sum())
        fp = int((pred_bad & y).sum())
        fn = int((~pred_bad & ~y).sum())
        if objective == "f1":
            v = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        elif objective == "youden":
            tn = int((~pred_bad & y).sum())
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            v = tpr - fpr
        else:
            raise CalibrationError(f"unknown objective {objective!r}")
        if v > best_v:
            best, best_v = float(c), v
    return best
