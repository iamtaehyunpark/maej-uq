"""Delta-field derivation: the pairing guards are the point of the tests.

A delta built from mismatched arms is not a weak signal, it is a meaningless
one — the failure has to be loud rather than a quietly shorter output file.
"""

from __future__ import annotations

import pytest

from masattr.judge.delta import derive_delta
from masattr.judge.score import StepScore
from masattr.runs.delta_field import _auroc, step_level


def _s(idx, p, *, look="none", agent="A", t="execute", key="hc/f", gt=False, ok=True):
    subset, _, file_id = key.partition("/")
    return StepScore(
        subset=subset,
        file_id=file_id,
        step_idx=idx,
        agent=agent,
        type_norm=t,
        type_source="parsed",
        p_raw=p,
        lookahead=look,
        with_gt=gt,
        parse_ok=ok,
    )


def test_delta_is_arm_minus_base_and_is_retagged():
    base = [_s(0, 0.9), _s(1, 0.8)]
    arm = [_s(0, 0.9, look="resp"), _s(1, 0.2, look="resp")]
    out = derive_delta(base, arm)
    assert [round(r.p_raw, 6) for r in out] == [0.0, -0.6]
    assert {r.readout for r in out} == {"delta_resp"}
    assert {r.lookahead for r in out} == {"resp"}
    # the delta gets its own normalization; the base's p_norm must not leak
    assert all(r.p_norm is None for r in out)


def test_parse_failure_propagates_from_either_side():
    base = [_s(0, 0.5, ok=False)]
    arm = [_s(0, 0.1, look="resp", ok=True)]
    assert derive_delta(base, arm)[0].parse_ok is False


def test_partial_overlap_is_an_error_not_a_short_file():
    base = [_s(0, 0.9), _s(1, 0.8)]
    arm = [_s(0, 0.4, look="resp")]
    with pytest.raises(ValueError, match="same steps"):
        derive_delta(base, arm)


def test_mismatched_step_identity_is_rejected():
    base = [_s(0, 0.9, agent="A")]
    arm = [_s(0, 0.4, look="resp", agent="B")]
    with pytest.raises(ValueError, match="disagree on the step"):
        derive_delta(base, arm)


def test_mixed_gt_regimes_are_rejected():
    base = [_s(0, 0.9, gt=False)]
    arm = [_s(0, 0.4, look="resp", gt=True)]
    with pytest.raises(ValueError, match="GT setting"):
        derive_delta(base, arm)


def test_delta_against_w0_itself_is_rejected():
    base = [_s(0, 0.9)]
    arm = [_s(0, 0.4)]
    with pytest.raises(ValueError, match="identically zero"):
        derive_delta(base, arm)


def test_step_level_auroc_rewards_a_drop_at_the_gold_step():
    # gold step 2 is the only one that lost credibility under lookahead
    rows = [_s(0, 0.0), _s(1, 0.05), _s(2, -0.7), _s(3, 0.02)]
    out = step_level(rows, {"hc/f": 2})
    assert out["within_file_auroc"] == 1.0
    assert out["mean_delta_gold"] < out["mean_delta_other"]


def test_auroc_splits_ties_in_half():
    assert _auroc([1.0], [1.0]) == 0.5
    assert _auroc([], [1.0]) is None


def test_risk_coverage_counts_only_correct_selections():
    """A curve that reads 1.000 at full coverage while most picks are wrong is
    counting rows, not hits — the bug this guards was exactly that."""
    from masattr.runs.delta_field import file_level

    class _Rec:
        def __init__(self, key, step, agent):
            self.key, self.label_mistake_step, self.label_mistake_agent = key, step, agent

    grouped, records = {}, []
    for i in range(4):
        key = f"hc/f{i}"
        # step 1 always wins argmin; only file 0 has gold at step 1
        grouped[key] = [
            _s(0, 0.0, look="resp", key=key),
            _s(1, -0.5 - i, look="resp", key=key),
        ]
        for r in grouped[key]:
            r.p_norm = r.p_raw
        records.append(_Rec(key, 1 if i == 0 else 0, "A"))

    out = file_level(grouped, records, "argmin", {})
    assert out["n_selected"] == 4 and out["n_correct"] == 1
    assert out["risk_coverage"][-1]["accuracy"] == 0.25
