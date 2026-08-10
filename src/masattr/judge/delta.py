"""Lookahead-shift (delta) score fields.

The lookahead arms lose to W0 on level: appending what happened after step ``t``
makes the judge's absolute P(True) worse, not better, so those arms were dropped
as candidate primary fields. The *shift* is a different quantity. A step that
reads as sound in isolation and collapses once its realized response is appended
has been falsified by its own consequence; a step that reads the same either way
has not. That contrast is what this module extracts::

    delta[t] = p_lookahead[t] - p_base[t]

Polarity is already correct for the rule set: falsified means the score falls,
so the decisive step is the one where ``delta`` is most negative, and the
low-is-suspicious rules (``argmin``, ``first_crossing``, ``changepoint_single``)
apply unchanged. No sign flip anywhere.

A delta field is emitted as an ordinary ``StepScore`` JSONL, so it normalizes
through the same LOO folds and attributes through the same rules as any judged
field. The only field-level provenance that changes is ``readout``, which
becomes ``delta_<arm>`` so downstream grouping keeps delta rows separate from
the arms they were built from.

**Pairing is checked, not assumed.** The two inputs must agree on subset, file,
step index, agent, and step type. A mismatch means the arms were scored over
different corpora or different typing, and differencing them would produce a
field with no meaning; that is an error, not a warning.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .score import StepScore, load_scores


def _index(rows: Sequence[StepScore]) -> dict[tuple[str, str, int], StepScore]:
    return {(r.subset, r.file_id, r.step_idx): r for r in rows}


def derive_delta(
    base: Sequence[StepScore],
    arm: Sequence[StepScore],
    *,
    require_full_overlap: bool = True,
) -> list[StepScore]:
    """``arm - base`` per step, carrying the base row's provenance.

    ``parse_ok`` is the conjunction of the two inputs: a delta built on a row
    either side failed to parse is not a measurement of anything, and flagging
    it keeps it out of the sanity statistics without dropping the step.
    """
    b_idx, a_idx = _index(base), _index(arm)
    if require_full_overlap and set(b_idx) != set(a_idx):
        only_b, only_a = len(set(b_idx) - set(a_idx)), len(set(a_idx) - set(b_idx))
        raise ValueError(
            f"delta inputs do not cover the same steps: {only_b} only in base, "
            f"{only_a} only in the lookahead arm. Differencing partial arms "
            "would silently score a different corpus per row."
        )
    arms = {r.lookahead for r in arm}
    if len(arms) != 1:
        raise ValueError(f"expected one lookahead arm in the delta input, got {sorted(arms)}")
    arm_name = arms.pop()
    if arm_name == "none":
        raise ValueError("the lookahead input is itself W0; delta against it is identically zero")

    out: list[StepScore] = []
    for key in sorted(b_idx.keys() & a_idx.keys()):
        b, a = b_idx[key], a_idx[key]
        if (b.agent, b.type_norm) != (a.agent, a.type_norm):
            raise ValueError(
                f"{key}: arms disagree on the step itself — "
                f"base ({b.agent!r}, {b.type_norm!r}) vs arm ({a.agent!r}, {a.type_norm!r})"
            )
        if b.with_gt != a.with_gt:
            raise ValueError(f"{key}: arms differ in GT setting; delta would mix regimes")
        out.append(
            replace(
                b,
                p_raw=a.p_raw - b.p_raw,
                p_norm=None,  # refit on the delta field's own distribution
                readout=f"delta_{arm_name}",
                lookahead=arm_name,
                n_lookahead=a.n_lookahead,
                parse_ok=b.parse_ok and a.parse_ok,
                prefix_tokens=a.prefix_tokens,
                readout_tokens=a.readout_tokens,
            )
        )
    return out


def derive_from_paths(base_path: str | Path, arm_path: str | Path, out_path: str | Path) -> int:
    rows = derive_delta(load_scores(base_path), load_scores(arm_path))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.to_dict()) + "\n")
    return len(rows)
