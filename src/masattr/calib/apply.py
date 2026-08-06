"""Apply a frozen calibration to Who&When scores (spec v2 Part C §4).

Also holds the two things Exp-0 needs: the held-aside-20 slice and the
leave-one-out fallback that comes into force if the single-agent→MAS transfer
fails.

**Step labels on Who&When — a stated modelling decision.** Who&When annotates
one decisive mistake per trajectory; it does not label every step. To draw a
reliability diagram at all, per-step correctness has to be derived, and the spec
leaves how implicit. Two policies, both pre-registered here:

* ``prefix`` (default) — consider steps ``0..mistake_step`` only, with
  ``correct = idx < mistake_step``. Everything after the decisive mistake is
  downstream contamination with no ground truth of its own, so it is excluded
  rather than guessed.
* ``point`` — consider every step, with ``correct = idx != mistake_step``. Keeps
  the whole trajectory but asserts post-mistake steps are fine, which they
  usually are not.

``prefix`` is the default because it declines to invent labels. Both are
reported where the choice could matter.
"""

from __future__ import annotations

import random
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..record import Record
from ..judge.score import StepScore
from .fit import FrozenCalibration, fit

LABEL_POLICIES = ("prefix", "point")

#: Exp-0's held-aside slice: 10 AG + 10 HC, chosen by seeded RNG.
HELD_ASIDE_PER_SUBSET = 10


def step_labels(record: Record, policy: str = "prefix") -> dict[int, bool]:
    """Derived per-step correctness for one trajectory. See the module docstring."""
    if policy not in LABEL_POLICIES:
        raise ValueError(f"unknown label policy {policy!r}; known: {LABEL_POLICIES}")
    m = record.label_mistake_step
    if policy == "prefix":
        return {i: (i < m) for i in range(min(m + 1, record.n_steps))}
    return {i: (i != m) for i in range(record.n_steps)}


def labelled_rows(
    records: Sequence[Record],
    scores_by_file: Mapping[str, list[StepScore]],
    *,
    policy: str = "prefix",
    keys: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Flatten (p_raw, type_norm, derived correctness) over the given files."""
    allow = set(keys) if keys is not None else None
    ps, types, ys = [], [], []
    for rec in records:
        if allow is not None and rec.key not in allow:
            continue
        labels = step_labels(rec, policy)
        for s in scores_by_file.get(rec.key, []):
            if s.step_idx in labels:
                ps.append(s.p_raw)
                types.append(s.type_norm)
                ys.append(labels[s.step_idx])
    return np.asarray(ps, dtype=float), types, np.asarray(ys, dtype=bool)


def held_aside(
    records: Sequence[Record], *, seed: int = 0, per_subset: int = HELD_ASIDE_PER_SUBSET
) -> set[str]:
    """The seeded 10 AG + 10 HC slice Exp-0 checks transfer on.

    These files return to the test pool for non-calibration analyses, and every
    primary number is reported with and without them (Part E).
    """
    rng = random.Random(seed)
    chosen: set[str] = set()
    for subset in ("alg", "hc"):
        pool = sorted(r.key for r in records if r.subset == subset)
        if len(pool) < per_subset:
            raise ValueError(f"{subset}: need {per_subset} files for the held-aside slice, have {len(pool)}")
        chosen.update(rng.sample(pool, per_subset))
    return chosen


def apply_to(scores: Iterable[StepScore], cal: FrozenCalibration) -> None:
    """Write ``p_cal`` onto score rows in place."""
    for s in scores:
        s.p_cal = cal.apply_one(s.p_raw, s.type_norm)


def loo_calibrate(
    records: Sequence[Record],
    scores_by_file: dict[str, list[StepScore]],
    *,
    method: str = "percentile",
    policy: str = "prefix",
) -> dict[str, FrozenCalibration]:
    """Disclosed fallback: leave-one-file-out calibration fit on Who&When itself.

    In force only if Exp-0 fails. Each file is calibrated by a map fit on every
    *other* file, so no file contributes to its own calibration — but the
    uniformity claim is weakened to in-corpus calibration, and the paper text
    has to say so.
    """
    out: dict[str, FrozenCalibration] = {}
    keys = [r.key for r in records]
    for held in keys:
        ps, types, ys = labelled_rows(
            records, scores_by_file, policy=policy, keys=[k for k in keys if k != held]
        )
        if ps.size == 0:
            raise ValueError(f"{held}: no rows left to fit the leave-one-out map")
        out[held] = fit(ps, types, ys, method=method, fit_on=f"whowhen_loo(-{held})")
    return out


def apply_loo(
    scores_by_file: dict[str, list[StepScore]], maps: Mapping[str, FrozenCalibration]
) -> None:
    for key, rows in scores_by_file.items():
        cal = maps.get(key)
        if cal is None:
            continue
        for s in rows:
            s.p_cal = cal.apply_one(s.p_raw, s.type_norm)
