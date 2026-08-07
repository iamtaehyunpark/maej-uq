"""Normalization, attribution rules, and scoring."""

from __future__ import annotations

import random

import numpy as np
import pytest

from masattr.attribute.rules import (
    METHODS,
    agent_first,
    argmin,
    changepoint,
    disagreement,
    first_crossing,
)
from masattr.normalize.fit import (
    NormalizationError,
    choose_threshold,
    coefficient_of_variation,
    fit_all_subsets,
    fit_folds,
    load_folds,
    save_folds,
    stability,
    step_labels,
)
from masattr.eval.ci import bootstrap_ci, reliability
from masattr.eval.scorers import score_pairs, slices, substring_step
from masattr.judge.score import StepScore


def _score(idx, p, agent="A", t="execute", key="hc/f"):
    subset, _, file_id = key.partition("/")
    return StepScore(
        subset=subset,
        file_id=file_id,
        step_idx=idx,
        agent=agent,
        type_norm=t,
        type_source="parsed",
        p_raw=p,
    )


# --- intervals & reliability ------------------------------------------------


def test_reliability_on_a_calibrated_source():
    rng = random.Random(0)
    probs = [rng.random() for _ in range(4000)]
    labels = [rng.random() < p for p in probs]
    rel = reliability(probs, labels)
    assert rel.ece < 0.05
    assert "ECE=" in rel.render()


def test_bootstrap_ci_brackets_the_point():
    ci = bootstrap_ci([1] * 70 + [0] * 30, lambda u: sum(u) / len(u), n_boot=500, seed=1)
    assert ci.lo <= ci.point <= ci.hi


# --- scorers ----------------------------------------------------------------


def test_substring_scorer_reproduces_the_published_artifact():
    # Their test is `actual_step in pred['predicted_step']`: gold inside the
    # prediction. Predicting 12 scores a hit against gold 1 — why exact is primary.
    assert substring_step(12, 1)


def test_dual_scorer_disagrees_where_expected():
    pairs = [(("WebSurfer", 12), ("WebSurfer", 1))]  # (pred, gold)
    assert score_pairs(pairs, scorer="exact", n_boot=10).step_acc == 0.0
    assert score_pairs(pairs, scorer="substring", n_boot=10).step_acc == 1.0


def test_agent_match_collapses_orchestrator_naming():
    pairs = [(("Orchestrator (thought)", 0), ("MagenticOneOrchestrator", 0))]
    assert score_pairs(pairs, scorer="exact", n_boot=10).agent_acc == 1.0


def test_slices_drop_flagged_files(records):
    alg = records["alg"]
    s = slices(alg)
    assert len(s["all"]) == 4
    assert len(s["excl_flagged"]) == 3


def test_slices_add_held_aside_when_present(records):
    alg = records["alg"]
    s = slices(alg, held_aside={"alg/alg_0"})
    assert "excl_held_aside" in s and "excl_all_excluded" in s
    assert len(s["excl_held_aside"]) == 3


# --- attribution rules ------------------------------------------------------


def test_first_crossing_picks_the_earliest_low_step():
    s = [_score(0, 0.9), _score(1, 0.2), _score(2, 0.05)]
    a = first_crossing(s, 0.5)
    assert a.step == 1 and a.detail["crossed"]


def test_first_crossing_answers_even_when_nothing_crosses():
    # Every Who&When trajectory failed; declining would drop files from the
    # denominator instead of scoring a miss.
    s = [_score(0, 0.9), _score(1, 0.8)]
    a = first_crossing(s, 0.5)
    assert a.step == 1 and not a.detail["crossed"]


def test_argmin_differs_from_first_crossing():
    s = [_score(0, 0.9), _score(1, 0.2), _score(2, 0.05)]
    assert argmin(s).step == 2 and first_crossing(s, 0.5).step == 1


def test_changepoint_finds_the_break():
    s = [_score(i, 0.9) for i in range(5)] + [_score(i, 0.1) for i in range(5, 10)]
    assert changepoint(s).step == 5


def test_changepoint_hyperparam_is_frozen():
    from masattr.attribute.rules import CHANGEPOINT_MIN_SEG

    assert CHANGEPOINT_MIN_SEG == 2


def test_agent_first_selects_by_the_agents_best_step():
    s = [
        _score(0, 0.95, "Orchestrator"),
        _score(1, 0.05, "WebSurfer"),  # one catastrophic step, but a good one too
        _score(2, 0.90, "WebSurfer"),
        _score(3, 0.40, "Coder"),
        _score(4, 0.35, "Coder"),  # never good
    ]
    assert argmin(s).agent == "WebSurfer"
    # Selector is per-agent max p: WebSurfer peaks at 0.90, Coder at 0.40.
    assert agent_first(s, 0.5).agent == "Coder"


def test_disagreement_counts():
    s = [_score(0, 0.1), _score(1, 0.9)]
    rows = disagreement({"k": first_crossing(s, 0.5)}, {"k": argmin(s)})
    assert rows[0].n == 1 and rows[0].n_step == 0


def test_all_rules_registered():
    assert set(METHODS) == {
        "changepoint_single",
        "first_crossing",
        "argmin",
        "changepoint",
        "agent_first",
        "relative_crossing",
    }


def test_primary_is_fixed_by_directive():
    from masattr.attribute.rules import PRIMARY, ablation_methods

    assert PRIMARY == "changepoint_single"
    # Every demoted rule is an ablation row, and the k sweep lives only there.
    rows = ablation_methods()
    assert {"first_crossing", "argmin", "changepoint", "agent_first"} <= set(rows)
    assert [r for r in rows if r.startswith("relative_crossing@")] == [
        "relative_crossing@1.5",
        "relative_crossing@2.0",
        "relative_crossing@2.5",
    ]
    assert PRIMARY not in rows


def test_threshold_free_set_needs_no_threshold():
    from masattr.attribute.rules import METHODS as M, THRESHOLD_DEPENDENT, THRESHOLD_FREE

    assert set(THRESHOLD_FREE) | set(THRESHOLD_DEPENDENT) == set(M)
    s = [_score(0, 0.9), _score(1, 0.85), _score(2, 0.1), _score(3, 0.8)]
    for name in THRESHOLD_FREE:
        # Called with no threshold at all, these still answer.
        assert M[name](s).step is not None


def test_relative_crossing_is_early_biased_like_first_crossing():
    from masattr.attribute.rules import relative_crossing

    # Two steps fall k sd below the trajectory mean; the rule takes the first,
    # argmin takes the worst.
    s = [_score(0, 0.9), _score(1, 0.1), _score(2, 0.05), _score(3, 0.9), _score(4, 0.9)]
    assert relative_crossing(s, k=1.0).step == 1
    assert argmin(s).step == 2


def test_relative_crossing_k_is_a_parameter_not_a_constant():
    from masattr.attribute.rules import RELATIVE_K_SWEEP, relative_crossing, resolve_method

    s = [_score(0, 0.9), _score(1, 0.35), _score(2, 0.05), _score(3, 0.9), _score(4, 0.9)]
    # A looser k can only fire at the same step or earlier, never later — which
    # is the invariant the sweep is exploring.
    steps = [relative_crossing(s, k=k).step for k in (1.0, 1.5, 2.0, 2.5)]
    assert steps == sorted(steps)
    assert RELATIVE_K_SWEEP == (1.5, 2.0, 2.5)
    # The parameterised name is the same rule, not a re-implementation.
    for k in RELATIVE_K_SWEEP:
        assert resolve_method(f"relative_crossing@{k}")(s).step == relative_crossing(s, k=k).step


def test_relative_crossing_falls_back_when_nothing_stands_out():
    from masattr.attribute.rules import relative_crossing

    s = [_score(i, 0.5) for i in range(4)]
    a = relative_crossing(s)
    assert a.step is not None and a.detail.get("degenerate")


# --- normalized position ----------------------------------------------------


def test_position_table_detects_a_late_biased_rule():
    from masattr.attribute.rules import position_table

    preds = {"hc/a": first_crossing([_score(0, 0.9), _score(1, 0.9), _score(2, 0.1)], 0.5)}
    table = position_table(preds, {"hc/a": ("A", 0)}, {"hc/a": 3})
    assert table["predicted"]["median"] == 1.0
    assert table["gold"]["median"] == 0.0
    assert table["delta_pred_minus_gold"]["mean"] > 0
    assert table["fraction_predicted_after_gold"] == 1.0


def test_position_table_skips_single_step_trajectories():
    from masattr.attribute.rules import position_table

    preds = {"hc/a": first_crossing([_score(0, 0.1)], 0.5)}
    assert position_table(preds, {"hc/a": ("A", 0)}, {"hc/a": 1})["gold"]["n"] == 0


# --- the primary rule -------------------------------------------------------


def test_changepoint_single_returns_the_first_step_of_regime_two():
    from masattr.attribute.rules import changepoint_single

    s = [_score(i, 0.9) for i in range(5)] + [_score(i, 0.1) for i in range(5, 10)]
    a = changepoint_single(s)
    assert a.step == 5  # the boundary, not the worst step
    assert "fallback" not in a.detail
    assert a.detail["contrast"] > 0


def test_perfect_split_scores_highest_not_zero():
    from masattr.attribute.rules import changepoint_single

    # A perfect split has zero within-segment variance. Without a floor on the
    # pooled spread it divides by zero and the one split that should win is the
    # one that gets skipped.
    perfect = [_score(i, 0.9) for i in range(4)] + [_score(i, 0.1) for i in range(4, 8)]
    noisy = [_score(i, 0.9 - 0.05 * (i % 2)) for i in range(4)] + [
        _score(i, 0.2 + 0.05 * (i % 2)) for i in range(4, 8)
    ]
    a, b = changepoint_single(perfect), changepoint_single(noisy)
    assert a.step == b.step == 4
    assert a.detail["contrast"] > b.detail["contrast"]


def test_a_single_outlier_does_not_move_the_split():
    from masattr.attribute.rules import changepoint_single

    # One catastrophic step early, then a sustained shift later. argmin answers
    # the outlier; a regime split answers where the run actually changed, which
    # is the reason to prefer it.
    s = [_score(i, 0.9) for i in range(10)] + [_score(i, 0.4) for i in range(10, 20)]
    s[3] = _score(3, 0.0)
    assert argmin(s).step == 3
    assert changepoint_single(s).step == 10


def test_boundary_split_falls_back_to_argmin():
    from masattr.attribute.rules import changepoint_single

    s = [_score(i, 0.9) for i in range(6)] + [_score(6, 0.1), _score(7, 0.1)]
    a = changepoint_single(s)
    assert a.detail["fallback"] == "argmin"
    assert a.detail["reason"] == "boundary"


def test_low_contrast_falls_back_to_argmin():
    from masattr.attribute.rules import changepoint_single

    s = [_score(i, v) for i, v in enumerate([0.5, 0.6, 0.4, 0.55, 0.45, 0.52])]
    a = changepoint_single(s, min_contrast=100.0, boundary_fallback=False)
    assert a.detail["fallback"] == "argmin"
    assert a.detail["reason"] == "low_contrast"


def test_flat_trajectory_falls_back_to_argmin():
    from masattr.attribute.rules import changepoint_single

    a = changepoint_single([_score(i, 0.5) for i in range(8)])
    assert a.detail["reason"] == "no_variation"


def test_short_trajectory_falls_back_to_argmin():
    from masattr.attribute.rules import changepoint_single

    a = changepoint_single([_score(0, 0.9), _score(1, 0.1), _score(2, 0.2)])
    assert a.detail["reason"] == "too_short"


def test_changepoint_single_reads_no_threshold():
    from masattr.attribute.rules import changepoint_single

    s = [_score(i, 0.9) for i in range(5)] + [_score(i, 0.1) for i in range(5, 10)]
    # Any threshold, same answer: the rule takes nothing from outside.
    assert changepoint_single(s, 0.0).step == changepoint_single(s, {"execute": 9.9}).step


def test_registered_fallback_condition_changes_behaviour():
    from masattr.attribute.rules import changepoint_single

    s = [_score(i, 0.9) for i in range(6)] + [_score(6, 0.1), _score(7, 0.1)]
    assert changepoint_single(s, boundary_fallback=True).detail["reason"] == "boundary"
    strict = changepoint_single(s, boundary_fallback=False)
    assert strict.detail.get("fallback") is None and strict.step == 6


def test_substring_scorer_is_one_directional_like_theirs():
    from masattr.eval.scorers import substring_agent, substring_step

    # Their evaluate.py tests `actual_step in pred['predicted_step']` — gold
    # contained in the prediction, not symmetric. A symmetric version would be
    # strictly more lenient and would not reproduce the published regime.
    assert substring_step(12, 1)  # gold 1 inside prediction 12 — the artifact
    assert not substring_step(1, 12)  # gold 12 is not inside prediction 1
    assert substring_agent("the WebSurfer agent", "WebSurfer")
    assert not substring_agent("Web", "WebSurfer")


def test_repo_predictions_join_through_question_id():
    from masattr.baselines.whowhen_repo import parse_repo_output

    # Their files are named by ordinal; ours are keyed by question_ID. Joining
    # by position instead would line the two sides up wrongly and score noise.
    ids = {"1": "abc-123", "2": "def-456"}
    text = (
        "Prediction for 1.json:\nAgent Name: Excel_Expert\nStep Number: 0\n"
        "Prediction for 2.json:\nAgent Name: WebSurfer\nStep Number: 7\n"
        "Prediction for 9.json:\nAgent Name: Ghost\nStep Number: 1\n"
    )
    preds = parse_repo_output(text, ids, "alg")
    assert preds == {"alg/abc-123": ("Excel_Expert", 0), "alg/def-456": ("WebSurfer", 7)}


def test_repo_output_parser_skips_unparseable_blocks():
    from masattr.baselines.whowhen_repo import parse_repo_output

    text = "Prediction for 1.json:\nthe model rambled and named nothing\n"
    assert parse_repo_output(text, {"1": "abc"}, "alg") == {}
