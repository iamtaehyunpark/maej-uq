from __future__ import annotations

import pytest

from masattr.judge.client import MockClient
from masattr.judge.prompts import preamble, prompt_hash, readout
from masattr.judge.score import (
    cost_summary,
    pointers,
    score_corpus,
    score_record,
    turn_blocks,
)
from masattr.record import Record, Step


def _rec() -> Record:
    return Record(
        subset="hc",
        file_id="t",
        query="Which option is right?",
        ground_truth="B",
        steps=(
            Step(0, "Orchestrator", "Orchestrator (thought)", "Analyst: solve the stem.", "plan", "parsed"),
            Step(1, "Analyst", "Analyst", "The stem points to option B for these reasons.", "execute", "parsed"),
            Step(2, "Verifier", "Verifier", "B", "execute", "parsed"),
            Step(3, "Orchestrator", "Orchestrator (final answer)", "The answer is B.", "final", "parsed"),
        ),
        label_mistake_agent="Verifier",
        label_mistake_step=2,
    )


def test_scores_every_step():
    ts = score_record(_rec(), MockClient())
    assert [s.step_idx for s in ts.scores] == [0, 1, 2, 3]
    assert all(0.0 < s.p_raw < 1.0 for s in ts.scores)


def test_prefix_grows_monotonically():
    ts = score_record(_rec(), MockClient())
    toks = [s.prefix_tokens for s in ts.scores]
    assert toks == sorted(toks)


def test_near_empty_execute_is_augmented():
    ts = score_record(_rec(), MockClient(), policy="typed")
    assert ts.scores[2].augmented
    assert not ts.scores[1].augmented


def test_plain_policy_skips_augmentation():
    ts = score_record(_rec(), MockClient(), policy="plain")
    assert not any(s.augmented for s in ts.scores)


def test_pointers_never_reach_forward():
    rec = _rec()
    ptrs = pointers(rec, 2, turn_blocks(rec.steps))
    assert ptrs
    assert "The answer is B." not in "\n".join(ptrs)


def test_turn_blocks_open_on_coordination():
    rec = _rec()
    assert turn_blocks(rec.steps) == [0, 0, 0, 0]
    steps = (
        Step(0, "O", "O", "plan a", "plan", "parsed"),
        Step(1, "W", "W", "work a", "execute", "parsed"),
        Step(2, "O", "O", "plan b", "plan", "parsed"),
        Step(3, "W", "W", "work b", "execute", "parsed"),
    )
    assert turn_blocks(steps) == [0, 0, 1, 1]


def test_with_gt_setting_changes_the_preamble():
    rec = _rec()
    assert "[reference answer]" not in preamble(rec.query, rec.ground_truth)
    assert "[reference answer]" in preamble(rec.query, rec.ground_truth, with_gt=True)


def test_readouts_share_the_scaffold_and_differ_only_in_instruction():
    step = _rec().steps[1]
    a, b = readout(step, "ptrue"), readout(step, "verbalized")
    common = "Is step 1 by 'Analyst' correct and appropriate"
    assert common in a and common in b
    assert a != b


def test_binary_and_verbalized_readouts_produce_scores():
    for kind in ("verbalized", "binary"):
        ts = score_record(_rec(), MockClient(), kind=kind)
        assert all(0.0 <= s.p_raw <= 1.0 for s in ts.scores)
        assert all(s.readout == kind for s in ts.scores)


def test_hindsight_policy_uses_the_whole_trajectory():
    ts = score_record(_rec(), MockClient(), policy="hindsight")
    # The prefix is fixed and complete from the first assessment onward.
    assert len({s.prefix_tokens for s in ts.scores}) == 1
    assert ts.scores[0].prefix_tokens > 0


def test_untyped_arm_strips_types():
    ts = score_record(_rec(), MockClient(), use_types=False)
    assert {s.type_norm for s in ts.scores} == {"unknown"}
    assert all(s.use_types is False for s in ts.scores)


def test_scoring_refuses_a_client_without_prefix_sharing():
    client = MockClient()
    client.prefix_sharing = False
    with pytest.raises(RuntimeError, match="shared-prefix"):
        score_record(_rec(), client)


def test_cost_summary_reports_the_savings():
    cost = cost_summary(score_corpus([_rec()], MockClient()))
    assert cost["n_assessments"] == 4
    assert cost["quadratic_prefix_tokens_avoided"] > 0


def test_prompt_hash_is_stable():
    assert prompt_hash() == prompt_hash()
    assert len(prompt_hash()) == 16
