from __future__ import annotations

from masuq.judge import (
    MockPrefixScorer,
    MockSurrogateLM,
    build_evidence,
    cost_summary,
    judge_record,
    turn_blocks,
)
from masuq.loaders import load_subset
from masuq.schema import Record, Step


def _rec() -> Record:
    return Record(
        dataset="matu",
        subset="autogen_mmlu",
        task_id="t",
        run_id=0,
        query="Which option is right?",
        steps=[
            Step(0, "StarAgent", "assistant", "Analyst: solve the stem. Verifier: check it.", type_norm="plan", type_source="native"),
            Step(1, "Analyst", "assistant", "The stem points to option B for these reasons...", type_norm="execute", type_source="native"),
            Step(2, "Verifier", "assistant", "B", type_norm="execute", type_source="native"),
            Step(3, "StarAgent", "assistant", "The answer is B.", type_norm="final", type_source="native"),
        ],
    )


def test_evidence_is_prefix_conditional():
    rec = _rec()
    ev = build_evidence(rec, 1)
    assert "[step 0" in ev.text and "[step 1" in ev.text
    assert "[step 2" not in ev.text  # never look ahead


def test_near_empty_execute_gets_augmented():
    rec = _rec()
    ev = build_evidence(rec, 2)
    assert ev.augmented
    assert any("assigned subtask" in p for p in ev.pointers)


def test_augmentation_never_pulls_from_the_future():
    rec = _rec()
    ev = build_evidence(rec, 2)
    assert "The answer is B." not in "\n".join(ev.pointers)


def test_prefix_only_policy_skips_augmentation():
    ev = build_evidence(_rec(), 2, policy="prefix_only")
    assert not ev.augmented


def test_turn_blocks_keep_one_round_together():
    # One plan followed by worker steps and a final answer is a single round.
    assert turn_blocks(_rec().steps) == [0, 0, 0, 0]


def test_turn_blocks_open_a_new_block_per_coordination():
    steps = [
        Step(0, "Orch", "assistant", "plan a", type_norm="plan"),
        Step(1, "W", "assistant", "work a", type_norm="execute"),
        Step(2, "Orch", "assistant", "plan b", type_norm="plan"),
        Step(3, "W", "assistant", "work b", type_norm="execute"),
    ]
    assert turn_blocks(steps) == [0, 0, 1, 1]


def test_judge_record_scores_every_step():
    rec = _rec()
    ts = judge_record(rec, MockPrefixScorer())
    assert len(ts.scores) == len(rec.steps)
    assert all(0.0 < s.p_raw < 1.0 for s in ts.scores)
    assert [s.step_idx for s in ts.scores] == [0, 1, 2, 3]
    assert ts.scores[2].augmented


def test_prefix_grows_monotonically():
    rec = _rec()
    ts = judge_record(rec, MockPrefixScorer())
    toks = [s.prefix_tokens for s in ts.scores]
    assert toks == sorted(toks)


def test_cost_summary_reports_savings():
    rec = _rec()
    ts = judge_record(rec, MockPrefixScorer())
    cost = cost_summary([ts])
    assert cost["n_assessments"] == 4
    assert cost["quadratic_tokens_avoided"] > 0


def test_judge_over_loaded_corpus(paths):
    records, _ = load_subset("hc", paths.get("hc"))
    ts = judge_record(records[0], MockPrefixScorer())
    assert len(ts.scores) == len(records[0].steps)
    assert {s.type_norm for s in ts.scores} <= {"plan", "delegate", "execute", "final", "unknown"}


def test_surrogate_baseline_runs():
    scores = MockSurrogateLM().score_record(_rec())
    assert len(scores) == 4
    assert all(s.n_tokens > 0 for s in scores)
    assert all(0.0 < s.p_proxy < 1.0 for s in scores)
