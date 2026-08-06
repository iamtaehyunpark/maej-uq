"""Per-type calibration, fit once on paper 1's corpus (spec v2 Part C §4).

The uniformity claim is that *one* typed calibration, fit on a single-agent
step-labeled corpus, transfers to multi-agent trajectories. So the maps are fit
here and nowhere else, then frozen to ``calib/frozen/`` and applied unchanged to
Who&When. Exp-0 tests the transfer before any attribution number is seen.

**Input contract.** Fitting needs paper 1's ~30k step-labeled corpus *scored by
this judge* — a JSONL of ``{"p_raw": float, "type": str, "correct": bool}``
rows, where ``type`` is paper 1's action/thought type. Those types are mapped
into the function-type space by the table in ``specs/paper1_type_map.json``,
which is frozen before Exp-0 and hashed into the run manifest. The mapping is
the one place single-agent and multi-agent vocabularies meet, so it is data, not
code.

Methods: ``percentile`` (default — monotone, non-parametric), ``platt``,
``isotonic``. The default is rank-preserving because a monotone map cannot be
misled about ordering by label noise, only about level.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..record import TYPE_NORMS

METHODS = ("percentile", "platt", "isotonic")

#: Types with fewer than this many fitting rows fall back to the pooled map.
MIN_PER_TYPE_N = 200

#: Default paper-1 → function-type mapping. Overridden by
#: ``specs/paper1_type_map.json`` when present; that file is the frozen artifact.
DEFAULT_TYPE_MAP: dict[str, str] = {
    "thought": "plan",
    "think": "plan",
    "reason": "plan",
    "plan": "plan",
    "action": "execute",
    "act": "execute",
    "tool": "execute",
    "observation": "execute",
    "answer": "final",
    "final": "final",
    "finish": "final",
}


class CalibrationError(RuntimeError):
    pass


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: least-squares isotonic regression."""
    ys, ws, ns = [], [], []
    for i in range(y.size):
        ys.append(float(y[i]))
        ws.append(float(w[i]))
        ns.append(1)
        while len(ys) > 1 and ys[-2] > ys[-1]:
            wy = ws[-2] * ys[-2] + ws[-1] * ys[-1]
            ww = ws[-2] + ws[-1]
            nn = ns[-2] + ns[-1]
            ys[-2:], ws[-2:], ns[-2:] = [wy / ww], [ww], [nn]
    out = np.empty(y.size)
    k = 0
    for val, cnt in zip(ys, ns):
        out[k : k + cnt] = val
        k += cnt
    return out


# --- map kinds --------------------------------------------------------------


@dataclass(slots=True)
class Map:
    kind: str
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    a: float = 1.0
    b: float = 0.0

    def apply(self, p: float) -> float:
        if self.kind == "platt":
            return _sigmoid(self.a * _logit(p) + self.b)
        if not self.xs:
            return p
        return float(np.interp(p, self.xs, self.ys))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "xs": self.xs, "ys": self.ys, "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Map":
        return cls(
            kind=d["kind"],
            xs=list(d.get("xs", [])),
            ys=list(d.get("ys", [])),
            a=float(d.get("a", 1.0)),
            b=float(d.get("b", 0.0)),
        )


def fit_percentile(p: np.ndarray, y: np.ndarray, n_bins: int = 20) -> Map:
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order].astype(float)
    bins = min(n_bins, max(ps.size // 25, 2))
    edges = np.linspace(0, ps.size, bins + 1).astype(int)
    xs, vals = [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if hi > lo:
            xs.append(float(ps[lo:hi].mean()))
            vals.append(float(ys[lo:hi].mean()))
    vals = _pav(np.asarray(vals), np.ones(len(vals))).tolist()
    return Map("percentile", xs=xs, ys=vals)


def fit_platt(p: np.ndarray, y: np.ndarray, iters: int = 300, lr: float = 0.1) -> Map:
    x = np.array([_logit(float(v)) for v in p])
    t = y.astype(float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        pred = 1 / (1 + np.exp(-np.clip(a * x + b, -30, 30)))
        err = pred - t
        a -= lr * float((err * x).mean())
        b -= lr * float(err.mean())
    return Map("platt", a=float(a), b=float(b))


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> Map:
    order = np.argsort(p, kind="mergesort")
    xs = p[order].astype(float)
    ys = _pav(y[order].astype(float), np.ones(y.size))
    kx, ky = [], []
    for xv, yv in zip(xs, ys):
        if kx and math.isclose(xv, kx[-1]):
            ky[-1] = float(yv)
        else:
            kx.append(float(xv))
            ky.append(float(yv))
    return Map("isotonic", xs=kx, ys=ky)


_FITTERS = {"percentile": fit_percentile, "platt": fit_platt, "isotonic": fit_isotonic}


# --- the frozen artifact ----------------------------------------------------


@dataclass
class FrozenCalibration:
    method: str = "percentile"
    maps: dict[str, Map] = field(default_factory=dict)
    pooled: Map | None = None
    threshold: float = 0.5
    provenance: dict[str, Any] = field(default_factory=dict)

    def apply_one(self, p: float, type_norm: str) -> float:
        m = self.maps.get(type_norm) or self.pooled
        if m is None:
            raise CalibrationError("calibration has not been fit")
        return float(min(max(m.apply(float(p)), 0.0), 1.0))

    def apply_many(self, ps: Sequence[float], types: Sequence[str]) -> list[float]:
        return [self.apply_one(p, t) for p, t in zip(ps, types)]

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "pooled": self.pooled.to_dict() if self.pooled else None,
            "maps": {k: v.to_dict() for k, v in self.maps.items()},
            "provenance": self.provenance,
        }

    def content_hash(self) -> str:
        payload = json.dumps(
            {k: v for k, v in self.to_dict().items() if k != "provenance"}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = self.to_dict()
        blob["content_hash"] = self.content_hash()
        path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FrozenCalibration":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        c = cls(method=d["method"], threshold=float(d.get("threshold", 0.5)))
        c.pooled = Map.from_dict(d["pooled"]) if d.get("pooled") else None
        c.maps = {k: Map.from_dict(v) for k, v in d.get("maps", {}).items()}
        c.provenance = d.get("provenance", {})
        stored = d.get("content_hash")
        if stored and stored != c.content_hash():
            raise CalibrationError(
                f"{path}: content hash mismatch (stored {stored}, computed "
                f"{c.content_hash()}) — the frozen map has been edited in place"
            )
        return c


# --- paper-1 corpus ---------------------------------------------------------


def load_type_map(path: str | Path | None = None) -> dict[str, str]:
    if path and Path(path).exists():
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return {str(k).lower(): str(v) for k, v in blob.items()}
    return dict(DEFAULT_TYPE_MAP)


def load_paper1_scores(
    path: str | Path, type_map: Mapping[str, str]
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, int]]:
    """Read the scored paper-1 corpus and map its types into function types.

    Unmapped source types become ``unknown`` and are counted rather than
    dropped — a silently discarded slice of the fitting corpus would move every
    map without leaving a trace.
    """
    ps, types, ys = [], [], []
    unmapped: dict[str, int] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            raw = str(row.get("type", "")).strip().lower()
            mapped = type_map.get(raw)
            if mapped is None:
                unmapped[raw] = unmapped.get(raw, 0) + 1
                mapped = "unknown"
            if mapped not in TYPE_NORMS:
                raise CalibrationError(f"type map sends {raw!r} to unknown function type {mapped!r}")
            ps.append(float(row["p_raw"]))
            types.append(mapped)
            ys.append(bool(row["correct"]))
    if not ps:
        raise CalibrationError(f"{path}: no rows")
    return np.asarray(ps), types, np.asarray(ys, dtype=bool), unmapped


def choose_threshold(probs: Sequence[float], labels: Sequence[bool], objective: str = "f1") -> float:
    """Pick the first-crossing threshold on the *fitting* corpus.

    Choosing it on Who&When would leak: the threshold is the one free parameter
    of the primary attribution rule.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if p.size == 0:
        return 0.5
    best, best_v = 0.5, -1.0
    for c in np.unique(np.round(p, 3)):
        bad = p < c
        tp = int((bad & ~y).sum())
        fp = int((bad & y).sum())
        fn = int((~bad & ~y).sum())
        if objective == "f1":
            v = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        else:
            tn = int((~bad & y).sum())
            v = (tp / (tp + fn) if tp + fn else 0.0) - (fp / (fp + tn) if fp + tn else 0.0)
        if v > best_v:
            best, best_v = float(c), v
    return best


def fit(
    ps: np.ndarray,
    types: Sequence[str],
    ys: np.ndarray,
    *,
    method: str = "percentile",
    fit_on: str = "paper1",
    type_map_hash: str = "",
    extra: Mapping[str, Any] | None = None,
) -> FrozenCalibration:
    if method not in METHODS:
        raise CalibrationError(f"unknown method {method!r}; known: {METHODS}")
    fitter = _FITTERS[method]
    cal = FrozenCalibration(method=method)
    cal.pooled = fitter(ps, ys)

    per_type_n: dict[str, int] = {}
    ty = np.asarray(list(types))
    for t in TYPE_NORMS:
        mask = ty == t
        n = int(mask.sum())
        per_type_n[t] = n
        if n >= MIN_PER_TYPE_N and 0 < int(ys[mask].sum()) < n:
            cal.maps[t] = fitter(ps[mask], ys[mask])

    cal.threshold = choose_threshold(cal.apply_many(ps.tolist(), list(types)), ys)
    cal.provenance = {
        "method": method,
        "fit_on": fit_on,
        "n": int(ps.size),
        "base_rate": float(ys.mean()),
        "per_type_n": per_type_n,
        "types_with_own_map": sorted(cal.maps),
        "min_per_type_n": MIN_PER_TYPE_N,
        "type_map_hash": type_map_hash,
        **dict(extra or {}),
    }
    return cal
