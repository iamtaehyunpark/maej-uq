"""Calibration, attribution rules, and scoring."""

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
from masattr.calib.apply import held_aside, labelled_rows, step_labels
from masattr.calib.fit import CalibrationError, FrozenCalibration, choose_threshold, fit
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
    assert substring_step(1, 12)  # "1" in "12" — why exact match is primary


def test_dual_scorer_disagrees_where_expected():
    pairs = [(("WebSurfer", 1), ("WebSurfer", 12))]
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


# --- calibration ------------------------------------------------------------


def _fit_data(n=2400, seed=0):
    rng = random.Random(seed)
    ps, types, ys = [], [], []
    for i in range(n):
        p = rng.random()
        ps.append(p)
        types.append(["plan", "execute", "final"][i % 3])
        ys.append(rng.random() < p**2)
    return np.asarray(ps), types, np.asarray(ys, dtype=bool)


def test_percentile_calibration_reduces_ece():
    ps, types, ys = _fit_data()
    cal = fit(ps, types, ys)
    before = reliability(ps.tolist(), ys.tolist()).ece
    after = reliability(cal.apply_many(ps.tolist(), types), ys.tolist()).ece
    assert after < before


def test_all_methods_fit():
    ps, types, ys = _fit_data()
    for method in ("percentile", "platt", "isotonic"):
        out = fit(ps, types, ys, method=method).apply_many(ps.tolist(), types)
        assert all(0.0 <= v <= 1.0 for v in out)


def test_calibration_is_monotone():
    ps, types, ys = _fit_data()
    cal = fit(ps, types, ys)
    grid = [i / 50 for i in range(51)]
    vals = [cal.apply_one(v, "execute") for v in grid]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


def test_rare_types_fall_back_to_pooled():
    ps, types, ys = _fit_data()
    ps = np.append(ps, [0.5] * 5)
    types = types + ["delegate"] * 5
    ys = np.append(ys, [True] * 5)
    cal = fit(ps, types, ys)
    assert "delegate" not in cal.maps
    assert cal.apply_one(0.5, "delegate") == cal.pooled.apply(0.5)


def test_calibration_roundtrips_and_detects_tampering(tmp_path):
    ps, types, ys = _fit_data()
    cal = fit(ps, types, ys, fit_on="test")
    path = cal.save(tmp_path / "cal.json")
    loaded = FrozenCalibration.load(path)
    assert loaded.provenance["fit_on"] == "test"
    assert loaded.apply_one(0.42, "execute") == pytest.approx(cal.apply_one(0.42, "execute"))

    import json

    blob = json.loads(path.read_text())
    blob["threshold"] = 0.999
    path.write_text(json.dumps(blob))
    with pytest.raises(CalibrationError, match="content hash mismatch"):
        FrozenCalibration.load(path)


def test_threshold_lands_in_range():
    ps, _, ys = _fit_data()
    assert 0.0 <= choose_threshold(ps.tolist(), ys.tolist()) <= 1.0


# --- derived step labels ----------------------------------------------------


def test_prefix_label_policy_excludes_the_contaminated_tail(records):
    rec = records["hc"][0]  # mistake_step == 2
    labels = step_labels(rec, "prefix")
    assert set(labels) == {0, 1, 2}
    assert labels[0] is True and labels[1] is True and labels[2] is False


def test_point_label_policy_covers_every_step(records):
    rec = records["hc"][0]
    labels = step_labels(rec, "point")
    assert len(labels) == rec.n_steps
    assert labels[2] is False and labels[3] is True


def test_labelled_rows_respect_the_policy(records, scores):
    hc = records["hc"]
    ps_prefix, _, _ = labelled_rows(hc, scores, policy="prefix")
    ps_point, _, _ = labelled_rows(hc, scores, policy="point")
    assert ps_point.size > ps_prefix.size


def test_held_aside_needs_enough_files(records):
    with pytest.raises(ValueError, match="held-aside"):
        held_aside(records["alg"] + records["hc"], seed=0)


def test_held_aside_is_seeded_and_balanced(records):
    pool = records["alg"] + records["hc"]
    a = held_aside(pool, seed=0, per_subset=2)
    b = held_aside(pool, seed=0, per_subset=2)
    assert a == b
    assert sum(1 for k in a if k.startswith("alg/")) == 2
    assert sum(1 for k in a if k.startswith("hc/")) == 2


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
    assert set(METHODS) == {"first_crossing", "argmin", "changepoint", "agent_first"}


# --- per-type thresholds and the pooled-calibration arm ---------------------


def test_fit_produces_per_type_thresholds():
    ps, types, ys = _fit_data(n=3000)
    cal = fit(ps, types, ys)
    assert cal.thresholds  # §5's "crosses *its* calibrated threshold"
    assert set(cal.thresholds) <= set(cal.maps)
    for t in cal.thresholds:
        assert 0.0 <= cal.thresholds[t] <= 1.0


def test_threshold_for_switches_between_the_two_arms():
    ps, types, ys = _fit_data(n=3000)
    cal = fit(ps, types, ys)
    cal.thresholds["plan"] = 0.11
    assert cal.threshold_for("plan", per_type=True) == 0.11
    assert cal.threshold_for("plan", per_type=False) == cal.threshold
    # A type without its own threshold falls back to the global one.
    assert cal.threshold_for("delegate", per_type=True) == cal.threshold


def test_pooled_only_turns_typing_off_and_changes_nothing_else():
    ps, types, ys = _fit_data(n=3000)
    cal = fit(ps, types, ys)
    pooled = cal.pooled_only()
    assert pooled.maps == {} and pooled.thresholds == {}
    assert pooled.method == cal.method and pooled.threshold == cal.threshold
    assert pooled.apply_one(0.4, "plan") == pooled.apply_one(0.4, "execute")


def test_per_type_thresholds_change_the_crossing():
    # A plan step at 0.45 crosses a per-type threshold of 0.6 but not a global 0.3.
    s = [_score(0, 0.45, t="plan"), _score(1, 0.05, t="execute")]
    assert first_crossing(s, 0.3).step == 1
    assert first_crossing(s, {"plan": 0.6, "": 0.3}).step == 0


def test_threshold_mapping_falls_back_to_the_default_entry():
    s = [_score(0, 0.2, t="delegate")]
    assert first_crossing(s, {"plan": 0.9, "": 0.5}).step == 0


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
