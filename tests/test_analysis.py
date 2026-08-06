"""Calibration, aggregation, attribution, metrics."""

from __future__ import annotations

import math
import random

import pytest

from masuq.aggregate import aggregate_trajectory, length_normalized, noisy_or_uncertainty
from masuq.attribution import METHODS, agent_first, argmin_step, changepoint, disagreement, first_crossing
from masuq.calibration import CalibrationError, TypedCalibrator, choose_threshold
from masuq.judge.harness import StepScore
from masuq.metrics import (
    auarc,
    auroc,
    bootstrap_ci,
    reliability,
    score_attribution,
    substring_step_match,
)


def _score(idx, p, agent="A", t="execute", key="k"):
    return StepScore(
        key=key,
        dataset="whowhen",
        subset="hc",
        task_id="t",
        run_id=0,
        step_idx=idx,
        agent=agent,
        type_norm=t,
        type_source="parsed",
        p_raw=p,
    )


# --- metrics ---------------------------------------------------------------


def test_auroc_perfect_and_inverted():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0


def test_auroc_ties_are_half():
    assert auroc([0.5, 0.5], [True, False]) == 0.5


def test_auroc_single_class_is_nan():
    assert math.isnan(auroc([0.1, 0.9], [True, True]))


def test_auarc_rewards_rejecting_the_wrong_ones():
    correct = [True, True, False, False]
    good = auarc([0.0, 0.0, 1.0, 1.0], correct)  # uncertainty tracks failure
    bad = auarc([1.0, 1.0, 0.0, 0.0], correct)
    assert good > bad


def test_reliability_on_a_calibrated_source():
    rng = random.Random(0)
    probs = [rng.random() for _ in range(4000)]
    labels = [rng.random() < p for p in probs]
    rel = reliability(probs, labels, n_bins=10)
    assert rel.ece < 0.05
    assert "ECE=" in rel.render()


def test_bootstrap_ci_brackets_the_point():
    units = [1] * 70 + [0] * 30
    ci = bootstrap_ci(units, lambda u: sum(u) / len(u), n_boot=500, seed=1)
    assert ci.lo <= ci.point <= ci.hi
    assert 0.6 < ci.point < 0.8


# --- dual scorer -----------------------------------------------------------


def test_substring_scorer_reproduces_the_published_artifact():
    # "1" in "12" — the reason exact match is primary.
    assert substring_step_match(1, 12)


def test_dual_scorer_disagrees_where_expected():
    pairs = [(("WebSurfer", 1), ("WebSurfer", 12))]
    exact = score_attribution(pairs, scorer="exact", n_boot=10)
    sub = score_attribution(pairs, scorer="substring", n_boot=10)
    assert exact.step_acc == 0.0 and sub.step_acc == 1.0


def test_agent_match_collapses_orchestrator_naming():
    pairs = [(("Orchestrator (thought)", 0), ("MagenticOneOrchestrator", 0))]
    assert score_attribution(pairs, scorer="exact", n_boot=10).agent_acc == 1.0


# --- calibration -----------------------------------------------------------


def _fit_data(n=2000, seed=0):
    rng = random.Random(seed)
    p, t, y = [], [], []
    for i in range(n):
        raw = rng.random()
        type_norm = ["plan", "execute", "final"][i % 3]
        # Judge is overconfident: true rate is raw**2.
        p.append(raw)
        t.append(type_norm)
        y.append(rng.random() < raw**2)
    return p, t, y


def test_percentile_calibration_reduces_ece():
    p, t, y = _fit_data()
    cal = TypedCalibrator(method="percentile").fit(p, t, y, fit_on="test")
    before = reliability(p, y).ece
    after = reliability(cal.transform(p, t), y).ece
    assert after < before


def test_platt_and_isotonic_also_fit():
    p, t, y = _fit_data()
    for method in ("platt", "isotonic"):
        cal = TypedCalibrator(method=method).fit(p, t, y, fit_on="test")
        out = cal.transform(p, t)
        assert all(0.0 <= v <= 1.0 for v in out)


def test_calibration_is_monotone():
    p, t, y = _fit_data()
    cal = TypedCalibrator(method="percentile").fit(p, t, y, fit_on="test")
    grid = [i / 50 for i in range(51)]
    vals = [cal.transform_one(v, "execute") for v in grid]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


def test_freeze_blocks_refitting():
    p, t, y = _fit_data()
    cal = TypedCalibrator().fit(p, t, y, fit_on="test").freeze()
    with pytest.raises(CalibrationError, match="frozen"):
        cal.fit(p, t, y)


def test_calibrator_roundtrips(tmp_path):
    p, t, y = _fit_data()
    cal = TypedCalibrator().fit(p, t, y, fit_on="test").freeze()
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = TypedCalibrator.load(path)
    assert loaded.frozen
    assert loaded.provenance["fit_on"] == "test"
    assert loaded.transform_one(0.42, "execute") == pytest.approx(cal.transform_one(0.42, "execute"))


def test_rare_types_fall_back_to_pooled():
    p, t, y = _fit_data()
    p += [0.5] * 5
    t += ["delegate"] * 5
    y += [True] * 5
    cal = TypedCalibrator().fit(p, t, y, fit_on="test")
    assert "delegate" not in cal.maps  # below MIN_PER_TYPE_N
    assert cal.transform_one(0.5, "delegate") == cal.pooled.apply(0.5)


def test_choose_threshold_in_range():
    p, t, y = _fit_data()
    th = choose_threshold(p, y)
    assert 0.0 <= th <= 1.0


# --- aggregation -----------------------------------------------------------


def test_noisy_or_matches_the_closed_form():
    p = [0.9, 0.8, 0.5]
    assert noisy_or_uncertainty(p) == pytest.approx(1 - 0.9 * 0.8 * 0.5)


def test_noisy_or_saturates_with_length_but_normalized_does_not():
    short = [0.95] * 4
    long = [0.95] * 100
    assert noisy_or_uncertainty(long) > noisy_or_uncertainty(short)
    assert length_normalized(long) == pytest.approx(length_normalized(short))


def test_aggregate_excludes_final_when_types_given():
    u = aggregate_trajectory("k", [0.9, 0.9, 0.1], types=["plan", "execute", "final"])
    assert u.values["noisy_or_no_final"] < u.values["noisy_or"]
    assert u.primary == u.values["noisy_or"]


# --- attribution -----------------------------------------------------------


def test_first_crossing_picks_the_earliest_low_step():
    scores = [_score(0, 0.9), _score(1, 0.2), _score(2, 0.05)]
    a = first_crossing(scores, 0.5)
    assert a.step == 1 and a.detail["crossed"]


def test_first_crossing_falls_back_to_argmin():
    scores = [_score(0, 0.9), _score(1, 0.8)]
    a = first_crossing(scores, 0.5)
    assert a.step == 1 and not a.detail["crossed"]


def test_argmin_differs_from_first_crossing():
    scores = [_score(0, 0.9), _score(1, 0.2), _score(2, 0.05)]
    assert argmin_step(scores).step == 2
    assert first_crossing(scores, 0.5).step == 1


def test_changepoint_finds_the_break():
    scores = [_score(i, 0.9) for i in range(5)] + [_score(i, 0.1) for i in range(5, 10)]
    assert changepoint(scores).step == 5


def test_agent_first_aggregation_choice_changes_the_answer():
    scores = [
        _score(0, 0.95, "Orchestrator"),
        _score(1, 0.05, "WebSurfer"),  # one catastrophic step
        _score(2, 0.40, "Coder"),
        _score(3, 0.35, "Coder"),
        _score(4, 0.30, "Coder"),  # consistently mediocre, never catastrophic
    ]
    assert argmin_step(scores).agent == "WebSurfer"
    # Mean still follows the single worst step...
    assert agent_first(scores).agent == "WebSurfer"
    # ...while accumulating over an agent's whole contribution finds the agent
    # that is quietly wrong throughout. This is the step-first/agent-first split.
    assert agent_first(scores, agent_stat="noisy_or").agent == "Coder"


def test_disagreement_counts():
    a = {"k": first_crossing([_score(0, 0.1), _score(1, 0.9)], 0.5)}
    b = {"k": argmin_step([_score(0, 0.1), _score(1, 0.9)])}
    rows = disagreement(a, b)
    assert rows[0].n == 1 and rows[0].n_disagree_step == 0


def test_all_methods_registered():
    assert set(METHODS) == {"first_crossing", "argmin", "changepoint", "agent_first"}
