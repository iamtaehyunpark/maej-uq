"""Per-type calibration, fit once on paper 1's corpus (spec v2 Part C §4).

The uniformity claim is that *one* typed calibration, fit on a single-agent
step-labeled corpus, transfers to multi-agent trajectories. So the maps are fit
here and nowhere else, then frozen to ``calib/frozen/`` and applied unchanged to
Who&When. Exp-0 tests the transfer before any attribution number is seen.

**Input contract.** Fitting needs paper 1's ~30k step-labeled corpus *scored by
this judge*, one JSONL row per judged step::

    {"step_id": str, "arm": str, "model": str, "benchmark": "alfworld|hotpotqa",
     "step_kind": "thought|action",
     "tau": {"info": bool, "world_mod": bool, "reversible": bool, "cost": str},
     "p_raw": float, "judge_model": str, "label_correct": bool}

``p_raw`` must be the **raw single-probe** P(True) under the protocol arm
matching ours — prefix-conditional, logit readout. Not a noisy-OR or otherwise
combined score: calibrating a combined value makes every downstream map inherit
the combiner, which is exactly the dependency this design is trying not to have.

Rows are filtered to ``judge_model`` matching the transferring judge, so the
maps describe the judge that will actually be applied to Who&When.

``step_kind`` and ``tau`` are mapped into the function-type space by
``specs/paper1_type_map.json`` — the one place single-agent and multi-agent
vocabularies meet, so it is data, hashed into every manifest, not code. That
table must be marked ``"status": "frozen"`` before E0 runs.

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

#: Fallback rule table, used only when ``specs/paper1_type_map.json`` is absent.
#: It is deliberately the spec's sketch and is marked draft, so a run without the
#: frozen artifact cannot quietly proceed.
DEFAULT_TYPE_MAP: dict[str, Any] = {
    "status": "draft",
    "rules": [
        {"step_kind": "thought", "type": "plan"},
        {"step_kind": "action", "tau": {"info": True}, "type": "execute"},
        {"step_kind": "action", "tau": {"world_mod": True}, "type": "execute"},
        {"step_kind": "action", "type": "execute"},
    ],
    "default": "unknown",
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
    #: Per-type crossing thresholds. §5 says "crosses **its** calibrated
    #: threshold", and E4's "type-normalized vs global threshold" arm is
    #: unrunnable without these. ``threshold`` remains the global fallback.
    thresholds: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def apply_one(self, p: float, type_norm: str) -> float:
        m = self.maps.get(type_norm) or self.pooled
        if m is None:
            raise CalibrationError("calibration has not been fit")
        return float(min(max(m.apply(float(p)), 0.0), 1.0))

    def apply_many(self, ps: Sequence[float], types: Sequence[str]) -> list[float]:
        return [self.apply_one(p, t) for p, t in zip(ps, types)]

    def threshold_for(self, type_norm: str, *, per_type: bool = True) -> float:
        """Crossing threshold for a step of this type.

        ``per_type=False`` is E4's global-threshold arm — same maps, one
        threshold — so the two arms differ in exactly one thing.
        """
        if not per_type:
            return self.threshold
        return self.thresholds.get(type_norm, self.threshold)

    def pooled_only(self) -> "FrozenCalibration":
        """This calibration with typing switched off: pooled map, one threshold.

        The calibration half of E4. Keeping it a derived view rather than a
        second fit means the two arms cannot differ by anything else.
        """
        return FrozenCalibration(
            method=self.method,
            maps={},
            pooled=self.pooled,
            threshold=self.threshold,
            thresholds={},
            provenance={**self.provenance, "typing": "off (pooled map, global threshold)"},
        )

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "thresholds": self.thresholds,
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
        c = cls(
            method=d["method"],
            threshold=float(d.get("threshold", 0.5)),
            thresholds={k: float(v) for k, v in d.get("thresholds", {}).items()},
        )
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


def load_type_map(path: str | Path | None = None) -> dict[str, Any]:
    """Load the paper-1 τ → function-type rule table."""
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(DEFAULT_TYPE_MAP)


def map_step(step_kind: str, tau: Mapping[str, Any] | None, table: Mapping[str, Any]) -> str:
    """Apply the rule table to one paper-1 step.

    Rules are ordered; the first whose ``step_kind`` matches (or is absent) and
    whose every listed ``tau`` key equals the row's value wins.
    """
    kind = (step_kind or "").strip().lower()
    tau = tau or {}
    for rule in table.get("rules", []):
        want_kind = rule.get("step_kind")
        if want_kind is not None and str(want_kind).lower() != kind:
            continue
        if any(tau.get(k) != v for k, v in (rule.get("tau") or {}).items()):
            continue
        return str(rule["type"])
    return str(table.get("default", "unknown"))


def load_paper1_scores(
    path: str | Path,
    table: Mapping[str, Any],
    *,
    judge_model: str | None = None,
    require_frozen: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, Any]]:
    """Read paper 1's scored corpus and map its steps into function types.

    Returns ``(p_raw, types, label_correct, report)``. The report carries the
    counts that would otherwise vanish: rows filtered out by judge model, rows
    the table sent to ``unknown``, and the (step_kind, tau) combinations
    responsible — a silently discarded slice of the fitting corpus moves every
    map without leaving a trace.
    """
    if require_frozen and str(table.get("status", "draft")).lower() != "frozen":
        raise CalibrationError(
            "the paper-1 type map is still marked "
            f"{table.get('status', 'draft')!r}. Part C §4 requires the mapping table "
            "frozen before E0 — set \"status\": \"frozen\" in "
            "specs/paper1_type_map.json once it has been reviewed against the corpus."
        )

    ps: list[float] = []
    types: list[str] = []
    ys: list[bool] = []
    unmapped: dict[str, int] = {}
    seen_judges: dict[str, int] = {}
    kept_benchmarks: dict[str, int] = {}
    n_rows = n_filtered = 0

    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            n_rows += 1
            jm = str(row.get("judge_model", ""))
            seen_judges[jm] = seen_judges.get(jm, 0) + 1
            if judge_model and jm != judge_model:
                n_filtered += 1
                continue
            for required in ("p_raw", "label_correct", "step_kind"):
                if required not in row:
                    raise CalibrationError(
                        f"{path}:{lineno}: missing {required!r}; expected the E0 "
                        "handoff schema (step_id, arm, model, benchmark, step_kind, "
                        "tau, p_raw, judge_model, label_correct)"
                    )
            mapped = map_step(row["step_kind"], row.get("tau"), table)
            if mapped not in TYPE_NORMS:
                raise CalibrationError(
                    f"type map produced {mapped!r}, not one of {TYPE_NORMS}"
                )
            if mapped == "unknown":
                key = f"{row['step_kind']}|{sorted((row.get('tau') or {}).items())}"
                unmapped[key] = unmapped.get(key, 0) + 1
            ps.append(float(row["p_raw"]))
            types.append(mapped)
            ys.append(bool(row["label_correct"]))
            b = str(row.get("benchmark", "?"))
            kept_benchmarks[b] = kept_benchmarks.get(b, 0) + 1

    if not ps:
        raise CalibrationError(
            f"{path}: no rows left"
            + (
                f" after filtering to judge_model == {judge_model!r}; "
                f"saw {sorted(seen_judges)}"
                if judge_model
                else ""
            )
        )
    report = {
        "n_rows_read": n_rows,
        "n_rows_used": len(ps),
        "n_filtered_by_judge_model": n_filtered,
        "judge_model_filter": judge_model,
        "judge_models_present": seen_judges,
        "benchmarks_used": kept_benchmarks,
        "unmapped_to_unknown": unmapped,
        "type_map_status": table.get("status"),
    }
    return np.asarray(ps), types, np.asarray(ys, dtype=bool), report


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

    calibrated = cal.apply_many(ps.tolist(), list(types))
    cal.threshold = choose_threshold(calibrated, ys)
    # Per-type thresholds where there is enough of that type to pick one; types
    # without their own map do not get their own threshold either.
    ty = np.asarray(list(types))
    for t in cal.maps:
        mask = ty == t
        if int(mask.sum()) >= MIN_PER_TYPE_N:
            cal.thresholds[t] = choose_threshold(
                [c for c, keep in zip(calibrated, mask) if keep], ys[mask]
            )
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
