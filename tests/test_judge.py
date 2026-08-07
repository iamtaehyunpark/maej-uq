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


# --- evidence toggles (E5 arms) ---------------------------------------------


def test_subtask_and_peer_pointers_toggle_independently():
    rec = _rec()
    blocks = turn_blocks(rec.steps)
    both = pointers(rec, 2, blocks)
    only_subtask = pointers(rec, 2, blocks, peers=False)
    only_peers = pointers(rec, 2, blocks, subtask=False)
    assert len(both) == len(only_subtask) + len(only_peers)
    assert all("assigned subtask" in p for p in only_subtask)
    assert all(p.startswith("peer step") for p in only_peers)


def test_peer_corroboration_off_still_scores():
    ts = score_record(_rec(), MockClient(), peer_corroboration=False)
    assert len(ts.scores) == 4
    assert all(s.peer_corroboration is False for s in ts.scores)


def test_prefix_window_limits_visible_history():
    from masattr.judge.score import PREFIX_BUDGET_CHARS, retained_render

    ts = score_record(_rec(), MockClient(), prefix_window=2)
    assert all(s.prefix_window == 2 for s in ts.scores)
    # With a window the prefix stops growing without bound.
    assert ts.scores[-1].prefix_tokens <= ts.scores[1].prefix_tokens + 200
    _ = PREFIX_BUDGET_CHARS, retained_render


# --- truncation: type-aware retention ---------------------------------------


def _long_record(n_execute: int = 40, chars: int = 4000) -> Record:
    steps = [Step(0, "Orchestrator", "Orchestrator (thought)", "plan the work", "plan", "parsed")]
    for i in range(1, n_execute + 1):
        steps.append(Step(i, "WebSurfer", "WebSurfer", "x" * chars, "execute", "parsed"))
    steps.append(
        Step(len(steps), "Orchestrator", "Orchestrator (-> WebSurfer)", "go on", "delegate", "parsed")
    )
    return Record(
        subset="hc",
        file_id="long",
        query="q",
        ground_truth="g",
        steps=tuple(steps),
        label_mistake_agent="WebSurfer",
        label_mistake_step=1,
    )


def test_retention_keeps_coordination_verbatim_and_demotes_old_execution():
    from masattr.judge.score import retained_render

    rec = _long_record()
    text, demoted = retained_render("HEAD\n", rec.steps, budget=30_000)
    assert demoted, "a 40x4000-char trajectory must exceed a 30k budget"
    # Every coordination step survives in full; only execution is demoted.
    assert "plan the work" in text and "go on" in text
    coordination = {s.idx for s in rec.steps if s.type_norm in ("plan", "delegate")}
    assert not (set(demoted) & coordination)
    # The newest execution survives, the oldest does not.
    assert rec.steps[-2].idx not in demoted
    assert 1 in demoted


def test_demoted_steps_keep_their_row():
    from masattr.judge.score import retained_render

    text, demoted = retained_render("HEAD\n", _long_record().steps, budget=30_000)
    for idx in demoted:
        assert f"[step {idx} |" in text  # the row survives; the detail does not
    assert "chars withheld" in text


def test_truncation_rebuilds_and_is_reported():
    from masattr.judge.score import cost_summary

    ts = score_record(_long_record(), MockClient(), budget_chars=30_000)
    assert ts.rebuilds >= 1
    assert any(s.n_demoted for s in ts.scores)
    cost = cost_summary([ts])
    assert cost["trajectories_truncated"] == 1
    assert cost["fraction_assessments_truncated"] > 0
    assert cost["max_demoted_steps"] > 0


def test_no_truncation_under_budget():
    ts = score_record(_rec(), MockClient())
    assert ts.rebuilds == 0
    assert all(s.n_demoted == 0 for s in ts.scores)


def test_rebuild_leaves_headroom_so_it_does_not_thrash():
    # Rebuilding to exactly the budget makes the next step breach immediately and
    # the trajectory rebuilds every step — 104 rebuilds over 14 HC trajectories
    # before RETAIN_TARGET was introduced. Rebuild count scales with step size
    # over headroom, so this uses a step of ~3% of budget, the HC ballpark.
    ts = score_record(_long_record(n_execute=60, chars=1000), MockClient(), budget_chars=30_000)
    assert ts.rebuilds < len(ts.scores) / 4


@pytest.mark.parametrize("chars", [500, 1000, 2000, 4000])
def test_prefix_never_exceeds_the_budget(chars):
    # The bound has to hold even when a single step is a large fraction of the
    # budget — the case that broke the first implementation, which charged the
    # demoted header lines only after deciding what to keep verbatim.
    ts = score_record(_long_record(n_execute=60, chars=chars), MockClient(), budget_chars=30_000)
    # MockClient reports prefix_tokens as chars/4, so the budget converts too.
    assert max(s.prefix_tokens for s in ts.scores) * 4 <= 30_000
