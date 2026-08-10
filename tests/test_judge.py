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


# --- lookahead arms (W0 / W+resp / W+own) ------------------------------------


def _mixed() -> Record:
    steps = (
        Step(0, "Orch", "o", "plan the work", "plan", "parsed"),
        Step(1, "Orch", "o", "WebSurfer, go", "delegate", "parsed"),
        Step(2, "WebSurfer", "w", "searching, no luck", "execute", "parsed"),
        Step(3, "Verifier", "v", "that looks wrong", "execute", "parsed"),
        Step(4, "Coder", "c", "third responder", "execute", "parsed"),
        Step(5, "Orch", "o", "the answer is X", "final", "parsed"),
    )
    return Record(
        subset="hc", file_id="m", query="q", ground_truth="g", steps=steps,
        label_mistake_agent="Orch", label_mistake_step=1,
    )


def test_resp_takes_contiguous_other_agent_steps_capped_at_two():
    from masattr.judge.score import RESP_CAP, lookahead_steps

    rec = _mixed()
    assert [s.idx for s in lookahead_steps(rec.steps, 1, "resp")] == [2, 3]
    assert RESP_CAP == 2
    # It stops at the actor's own reappearance, not just at the cap.
    assert [s.idx for s in lookahead_steps(rec.steps, 4, "resp")] == [5]


def test_own_adds_the_actors_next_appearance():
    from masattr.judge.score import lookahead_steps

    rec = _mixed()
    assert [s.idx for s in lookahead_steps(rec.steps, 1, "own")] == [2, 3, 5]
    # No later appearance -> nothing extra beyond the response window.
    assert [s.idx for s in lookahead_steps(rec.steps, 2, "own")] == [3, 4]


def test_w0_is_prefix_conditional():
    from masattr.judge.score import lookahead_steps

    assert lookahead_steps(_mixed().steps, 1, "none") == []


def test_lookahead_is_recorded_on_every_row():
    for mode in ("none", "resp", "own"):
        ts = score_record(_mixed(), MockClient(), lookahead=mode)
        assert all(s.lookahead == mode for s in ts.scores)
        assert any(s.n_lookahead > 0 for s in ts.scores) == (mode != "none")


def test_lookahead_does_not_enter_the_shared_prefix():
    # It differs per step, so it must ride in the readout segment; otherwise the
    # cache would be invalidated every step and cost would go quadratic.
    base = score_record(_mixed(), MockClient(), lookahead="none")
    ahead = score_record(_mixed(), MockClient(), lookahead="own")
    assert [s.prefix_tokens for s in base.scores] == [s.prefix_tokens for s in ahead.scores]


def test_near_empty_rescue_is_independent_of_the_arm():
    # The rescue is base assembly, not an arm.
    rec = _rec()  # step 2 is the bare "B"
    for mode in ("none", "resp", "own"):
        ts = score_record(rec, MockClient(), lookahead=mode)
        assert ts.scores[2].augmented


def test_deleg_widens_the_window_only_for_delegate_steps():
    from masattr.judge.score import DELEG_CAP, RESP_CAP, lookahead_steps

    # Orchestrator delegates at 0, then five other agents work, then it returns.
    steps = [Step(0, "Orch", "o", "W, go", "delegate", "parsed")]
    steps += [Step(i, f"A{i}", "a", "work", "execute", "parsed") for i in range(1, 7)]
    steps.append(Step(7, "Orch", "o", "back to me", "plan", "parsed"))

    assert DELEG_CAP > RESP_CAP
    # The delegation gets the wide window...
    assert len(lookahead_steps(steps, 0, "deleg")) == DELEG_CAP
    assert len(lookahead_steps(steps, 0, "resp")) == RESP_CAP
    # ...and a non-delegate step does not.
    assert len(lookahead_steps(steps, 1, "deleg")) == RESP_CAP


def test_deleg_window_stops_when_control_returns():
    from masattr.judge.score import lookahead_steps

    steps = [
        Step(0, "Orch", "o", "W, go", "delegate", "parsed"),
        Step(1, "W", "w", "ok", "execute", "parsed"),
        Step(2, "Orch", "o", "back already", "plan", "parsed"),
        Step(3, "W", "w", "more", "execute", "parsed"),
    ]
    # Control returns at step 2, so the window is just step 1 despite the cap.
    assert [s.idx for s in lookahead_steps(steps, 0, "deleg")] == [1]


# --- resilience on a shared box ---------------------------------------------


def test_resume_skips_completed_files_and_redoes_partial_ones(tmp_path):
    from masattr.judge.score import load_scores, score_corpus

    a, b = _rec(), _mixed()
    out = tmp_path / "scores.jsonl"
    score_corpus([a], MockClient(), out_path=out)
    assert len({r.key for r in load_scores(out)}) == 1

    # Append a half-scored trajectory, as a crash mid-file would leave.
    rows = load_scores(out)
    with out.open("a") as fh:
        import json as _json

        partial = score_corpus([b], MockClient())[0].scores[:2]
        for r in partial:
            fh.write(_json.dumps(r.to_dict()) + "\n")

    score_corpus([a, b], MockClient(), out_path=out, resume=True)
    got = load_scores(out)
    counts = {}
    for r in got:
        counts[r.key] = counts.get(r.key, 0) + 1
    # The finished file is not rescored, and the partial one is complete, not short.
    assert counts[a.key] == a.n_steps
    assert counts[b.key] == b.n_steps
    _ = rows


def test_resume_off_starts_clean(tmp_path):
    from masattr.judge.score import load_scores, score_corpus

    out = tmp_path / "s.jsonl"
    score_corpus([_rec()], MockClient(), out_path=out)
    score_corpus([_rec()], MockClient(), out_path=out, resume=False)
    assert len(load_scores(out)) == _rec().n_steps


def test_reasoning_scratchpad_is_stripped_before_parsing():
    from masattr.judge.score import _parse_generated

    # The answer follows a closed <think> block; startswith would miss it.
    assert _parse_generated("\n\n<think>\n\n</think>\n\nFalse", "binary") == (0.0, True)
    assert _parse_generated("<think>musing</think> True", "binary") == (1.0, True)
    assert _parse_generated("<think>x</think>\n0.83", "verbalized") == (0.83, True)


def test_unclosed_scratchpad_is_a_parse_failure_not_a_number():
    from masattr.judge.score import _parse_generated

    # Truncated mid-reasoning: the regex would otherwise read the step index out
    # of "evaluate step 1" and report it as a confidence of 1.00.
    p, ok = _parse_generated("\n\n<think>\nThe user wants me to evaluate step 1", "verbalized")
    assert (p, ok) == (0.5, False)
    p, ok = _parse_generated("<think>still thinking", "binary")
    assert (p, ok) == (0.5, False)


def test_every_readout_carries_the_same_scratchpad_suppressor():
    from masattr.judge.prompts import readout as ro

    step = _rec().steps[1]
    # Shared across readouts: E2 is only an ablation if the scaffold is shared.
    for kind in ("ptrue", "verbalized", "binary"):
        assert "<think>" not in ro(step, kind)
