"""Per-type score normalization, fit by leave-one-file-out CV (spec v2.1 §2).

The judge emits a raw score ``s_t`` per step. Those scores are not comparable
across act types — a plan and a tool call are graded against different rubrics,
and their score distributions differ in level and spread — so a single crossing
threshold over raw scores would silently rank types against each other rather
than steps within a type.

Normalization fixes the level and spread per type: ``z = (s − μ_type) / σ_type``.
The statistics are estimated by **leave-one-file-out CV inside each subset** —
each file is z-scored under statistics fit on every *other* file, so no file
contributes to its own normalization. Fold order is the sorted file key, which
makes the whole procedure deterministic without needing a seed.

Two arms come out of the same fit, which is what makes E4 an ablation rather
than two pipelines:

* **typed** — per-type statistics and per-type thresholds.
* **pooled** — one set of statistics and one threshold for every type.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..record import TYPE_NORMS, Record

#: Types with fewer than this many training steps fall back to the pooled
#: statistics: a mean and sd estimated from a handful of steps would add noise,
#: not comparability.
MIN_PER_TYPE_N = 50

#: Floor on σ. A type whose training scores are constant would otherwise divide
#: by zero; it is also the degenerate-field case E0 checks for.
MIN_SD = 1e-3


class NormalizationError(RuntimeError):
    pass


@dataclass(slots=True)
class TypeStats:
    mean: float
    sd: float
    n: int

    def z(self, s: float) -> float:
        return (float(s) - self.mean) / max(self.sd, MIN_SD)


@dataclass
class FoldStats:
    """Normalization statistics for one CV fold, plus its crossing thresholds."""

    held_out: str
    subset: str
    per_type: dict[str, TypeStats] = field(default_factory=dict)
    pooled: TypeStats | None = None
    threshold: float = 0.0
    thresholds: dict[str, float] = field(default_factory=dict)
    n_train_files: int = 0
    n_train_steps: int = 0

    def stats_for(self, type_norm: str, *, typed: bool = True) -> TypeStats:
        if typed:
            st = self.per_type.get(type_norm)
            if st is not None:
                return st
        if self.pooled is None:
            raise NormalizationError(f"{self.held_out}: fold has no pooled statistics")
        return self.pooled

    def z(self, s: float, type_norm: str, *, typed: bool = True) -> float:
        return self.stats_for(type_norm, typed=typed).z(s)

    def threshold_for(self, type_norm: str, *, typed: bool = True) -> float:
        if typed and type_norm in self.thresholds:
            return self.thresholds[type_norm]
        return self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "held_out": self.held_out,
            "subset": self.subset,
            "per_type": {k: asdict(v) for k, v in self.per_type.items()},
            "pooled": asdict(self.pooled) if self.pooled else None,
            "threshold": self.threshold,
            "thresholds": self.thresholds,
            "n_train_files": self.n_train_files,
            "n_train_steps": self.n_train_steps,
        }


# --- derived step labels ----------------------------------------------------

LABEL_POLICIES = ("prefix", "point")


def step_labels(record: Record, policy: str = "prefix") -> dict[int, bool]:
    """Derived per-step correctness for one trajectory.

    Who&When annotates one decisive mistake per trajectory, not every step, so
    the labels a threshold is fit against have to be derived. Two policies, both
    pre-registered:

    * ``prefix`` (default) — steps ``0..mistake_step`` only, ``correct = idx <
      mistake_step``. Everything after the decisive mistake is downstream
      contamination with no ground truth of its own, so it is excluded rather
      than guessed.
    * ``point`` — every step, ``correct = idx != mistake_step``. Keeps the whole
      trajectory but asserts the tail is fine, which it usually is not.
    """
    if policy not in LABEL_POLICIES:
        raise NormalizationError(f"unknown label policy {policy!r}; known: {LABEL_POLICIES}")
    m = record.label_mistake_step
    if policy == "prefix":
        return {i: (i < m) for i in range(min(m + 1, record.n_steps))}
    return {i: (i != m) for i in range(record.n_steps)}


# --- fitting ----------------------------------------------------------------


def _stats(values: Sequence[float]) -> TypeStats:
    arr = np.asarray(values, dtype=float)
    return TypeStats(mean=float(arr.mean()), sd=float(arr.std(ddof=0)), n=int(arr.size))


def choose_threshold(z: Sequence[float], correct: Sequence[bool], objective: str = "f1") -> float:
    """Pick the crossing threshold on the training folds.

    Scanning the observed z values rather than a fixed grid keeps the threshold
    on the same scale as the data it will be applied to.
    """
    zs = np.asarray(z, dtype=float)
    y = np.asarray(correct, dtype=bool)
    if zs.size == 0 or y.all() or (~y).all():
        return 0.0
    best, best_v = 0.0, -1.0
    for c in np.unique(np.round(zs, 3)):
        bad = zs < c
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


def _rows(
    records: Sequence[Record],
    scores_by_file: Mapping[str, list],
    keys: Sequence[str],
    policy: str,
) -> tuple[list[float], list[str], list[bool]]:
    by_key = {r.key: r for r in records}
    s: list[float] = []
    t: list[str] = []
    y: list[bool] = []
    for key in keys:
        rec = by_key.get(key)
        if rec is None:
            continue
        labels = step_labels(rec, policy)
        for row in scores_by_file.get(key, []):
            if row.step_idx in labels:
                s.append(row.p_raw)
                t.append(row.type_norm)
                y.append(labels[row.step_idx])
    return s, t, y


def fit_folds(
    records: Sequence[Record],
    scores_by_file: Mapping[str, list],
    *,
    policy: str = "prefix",
    subset: str | None = None,
) -> dict[str, FoldStats]:
    """One :class:`FoldStats` per file, fit on every *other* file in its subset.

    Subsets are normalized independently: AG and HC are different frameworks
    with different score distributions, and pooling them would import one
    corpus's level into the other's z-scores.
    """
    keys = sorted(r.key for r in records if subset is None or r.subset == subset)
    if len(keys) < 2:
        raise NormalizationError(
            f"leave-one-out needs at least 2 files, got {len(keys)}"
            + (f" for subset {subset!r}" if subset else "")
        )

    out: dict[str, FoldStats] = {}
    for held in keys:
        train = [k for k in keys if k != held]
        s, t, y = _rows(records, scores_by_file, train, policy)
        if not s:
            raise NormalizationError(f"{held}: no labelled training steps in its fold")

        fold = FoldStats(
            held_out=held,
            subset=subset or held.split("/", 1)[0],
            pooled=_stats(s),
            n_train_files=len(train),
            n_train_steps=len(s),
        )
        arr_t = np.asarray(t)
        for type_norm in TYPE_NORMS:
            mask = arr_t == type_norm
            if int(mask.sum()) >= MIN_PER_TYPE_N:
                fold.per_type[type_norm] = _stats([v for v, m in zip(s, mask) if m])

        z_typed = [fold.z(v, tt) for v, tt in zip(s, t)]
        fold.threshold = choose_threshold(z_typed, y)
        for type_norm in fold.per_type:
            mask = arr_t == type_norm
            zs = [z for z, m in zip(z_typed, mask) if m]
            ys = [v for v, m in zip(y, mask) if m]
            if len(zs) >= MIN_PER_TYPE_N:
                fold.thresholds[type_norm] = choose_threshold(zs, ys)
        out[held] = fold
    return out


def fit_all_subsets(
    records: Sequence[Record], scores_by_file: Mapping[str, list], *, policy: str = "prefix"
) -> dict[str, FoldStats]:
    folds: dict[str, FoldStats] = {}
    for subset in sorted({r.subset for r in records}):
        folds.update(fit_folds(records, scores_by_file, policy=policy, subset=subset))
    return folds


# --- persistence ------------------------------------------------------------


def save_folds(folds: Mapping[str, FoldStats], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"folds": {k: v.to_dict() for k, v in sorted(folds.items())}}
    blob["content_hash"] = content_hash(blob["folds"])
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return path


def content_hash(folds_blob: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(folds_blob, sort_keys=True).encode()).hexdigest()[:16]


def load_folds(path: str | Path) -> dict[str, FoldStats]:
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    stored = blob.get("content_hash")
    if stored and stored != content_hash(blob["folds"]):
        raise NormalizationError(
            f"{path}: content hash mismatch — the fitted statistics have been "
            "edited in place"
        )
    out: dict[str, FoldStats] = {}
    for key, d in blob["folds"].items():
        fold = FoldStats(
            held_out=d["held_out"],
            subset=d["subset"],
            per_type={k: TypeStats(**v) for k, v in d.get("per_type", {}).items()},
            pooled=TypeStats(**d["pooled"]) if d.get("pooled") else None,
            threshold=float(d.get("threshold", 0.0)),
            thresholds={k: float(v) for k, v in d.get("thresholds", {}).items()},
            n_train_files=int(d.get("n_train_files", 0)),
            n_train_steps=int(d.get("n_train_steps", 0)),
        )
        out[key] = fold
    return out


# --- stability --------------------------------------------------------------


def coefficient_of_variation(values: Sequence[float]) -> float:
    """CV on a signed, roughly zero-centred quantity.

    Thresholds live in z-space and hover near zero, so the textbook ``sd/mean``
    explodes for reasons that have nothing to do with instability. Dividing by
    the mean *absolute* value keeps the quantity meaningful, and a
    near-zero-spread threshold set still reports ~0.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    scale = float(np.abs(arr).mean())
    if scale < MIN_SD:
        return 0.0 if float(arr.std(ddof=0)) < MIN_SD else math.inf
    return float(arr.std(ddof=0) / scale)


def stability(folds: Mapping[str, FoldStats]) -> dict[str, Any]:
    """Cross-fold variation of the thresholds and of the per-type statistics."""
    by_subset: dict[str, list[FoldStats]] = {}
    for fold in folds.values():
        by_subset.setdefault(fold.subset, []).append(fold)

    out: dict[str, Any] = {}
    for subset, fs in sorted(by_subset.items()):
        thresholds = [f.threshold for f in fs]
        row: dict[str, Any] = {
            "n_folds": len(fs),
            "global_threshold": {
                "mean": float(np.mean(thresholds)),
                "sd": float(np.std(thresholds, ddof=0)),
                "cv": coefficient_of_variation(thresholds),
                "min": float(np.min(thresholds)),
                "max": float(np.max(thresholds)),
            },
            "per_type": {},
        }
        types = sorted({t for f in fs for t in f.per_type})
        for t in types:
            means = [f.per_type[t].mean for f in fs if t in f.per_type]
            sds = [f.per_type[t].sd for f in fs if t in f.per_type]
            thr = [f.thresholds[t] for f in fs if t in f.thresholds]
            row["per_type"][t] = {
                "n_folds_with_type": len(means),
                "mean_cv": coefficient_of_variation(means),
                "sd_cv": coefficient_of_variation(sds),
                "threshold_cv": coefficient_of_variation(thr) if len(thr) > 1 else None,
                "threshold_mean": float(np.mean(thr)) if thr else None,
            }
        out[subset] = row
    return out
