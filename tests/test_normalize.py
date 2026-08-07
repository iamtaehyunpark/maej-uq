"""Per-type normalization by leave-one-file-out CV, and its stability report."""

from __future__ import annotations

import json

import pytest

from masattr.judge.score import StepScore
from masattr.normalize.apply import (
    apply_folds,
    field_sanity,
    thresholds_for,
    worst_threshold_cv,
)
from masattr.normalize.fit import (
    MIN_PER_TYPE_N,
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


def _score(key, idx, p, t="execute"):
    subset, _, file_id = key.partition("/")
    return StepScore(
        subset=subset,
        file_id=file_id,
        step_idx=idx,
        agent="A",
        type_norm=t,
        type_source="parsed",
        p_raw=p,
    )


def _corpus(records, scores):
    return records, scores


# --- derived labels ---------------------------------------------------------


def test_prefix_policy_excludes_the_contaminated_tail(records):
    rec = records["hc"][0]  # mistake_step == 2
    labels = step_labels(rec, "prefix")
    assert set(labels) == {0, 1, 2}
    assert labels[0] and labels[1] and not labels[2]


def test_point_policy_covers_every_step(records):
    rec = records["hc"][0]
    labels = step_labels(rec, "point")
    assert len(labels) == rec.n_steps
    assert not labels[2] and labels[3]


def test_unknown_label_policy_is_rejected(records):
    with pytest.raises(NormalizationError, match="label policy"):
        step_labels(records["hc"][0], "vibes")


# --- fitting ----------------------------------------------------------------


def test_every_file_gets_a_fold_fit_without_it(records, scores):
    recs = records["hc"]
    folds = fit_folds(recs, scores, subset="hc")
    assert set(folds) == {r.key for r in recs}
    for key, fold in folds.items():
        assert fold.held_out == key
        assert fold.n_train_files == len(recs) - 1


def test_subsets_are_normalized_independently(records, scores):
    all_records = records["alg"] + records["hc"]
    folds = fit_all_subsets(all_records, scores)
    assert {f.subset for f in folds.values()} == {"alg", "hc"}
    # An HC fold's training files are all HC.
    hc_fold = next(f for f in folds.values() if f.subset == "hc")
    assert hc_fold.n_train_files == len(records["hc"]) - 1


def test_leave_one_out_needs_two_files(records, scores):
    with pytest.raises(NormalizationError, match="at least 2 files"):
        fit_folds(records["hc"][:1], scores, subset="hc")


def test_z_scores_centre_the_training_distribution():
    recs, sc = [], {}
    from masattr.record import Record, Step

    for i in range(6):
        steps = tuple(
            Step(j, "A", "A", "content here", "execute", "parsed") for j in range(4)
        )
        rec = Record(
            subset="hc",
            file_id=f"f{i}",
            query="q",
            ground_truth="g",
            steps=steps,
            label_mistake_agent="A",
            label_mistake_step=3,
        )
        recs.append(rec)
        sc[rec.key] = [_score(rec.key, j, 0.5 + 0.1 * i) for j in range(4)]

    folds = fit_folds(recs, sc, subset="hc")
    apply_folds(sc, folds)
    # The held-out file is scored under the others' statistics, so an extreme
    # file lands far from zero rather than being normalized onto itself.
    extreme = sc["hc/f5"][0].p_norm
    middle = sc["hc/f2"][0].p_norm
    assert abs(extreme) > abs(middle)


def test_apply_is_strict_about_missing_folds(records, scores):
    folds = fit_folds(records["hc"], scores, subset="hc")
    with pytest.raises(NormalizationError, match="no fold statistics"):
        apply_folds({**scores}, folds)  # alg files have no hc fold


def test_raw_score_survives_normalization(records, scores):
    hc = {k: v for k, v in scores.items() if k.startswith("hc/")}
    folds = fit_folds(records["hc"], hc, subset="hc")
    before = {k: [r.p_raw for r in v] for k, v in hc.items()}
    apply_folds(hc, folds)
    assert {k: [r.p_raw for r in v] for k, v in hc.items()} == before
    assert all(r.p_norm is not None for v in hc.values() for r in v)
    assert all(r.p == r.p_norm for v in hc.values() for r in v)


def test_folds_roundtrip_and_detect_tampering(records, scores, tmp_path):
    folds = fit_folds(records["hc"], scores, subset="hc")
    path = save_folds(folds, tmp_path / "folds.json")
    loaded = load_folds(path)
    assert set(loaded) == set(folds)
    assert loaded[next(iter(folds))].threshold == folds[next(iter(folds))].threshold

    blob = json.loads(path.read_text())
    first = next(iter(blob["folds"]))
    blob["folds"][first]["threshold"] = 99.0
    path.write_text(json.dumps(blob))
    with pytest.raises(NormalizationError, match="content hash mismatch"):
        load_folds(path)


def test_rare_types_fall_back_to_pooled_statistics(records, scores):
    folds = fit_folds(records["hc"], scores, subset="hc")
    fold = next(iter(folds.values()))
    # The fixtures are far below MIN_PER_TYPE_N, so no type earns its own stats.
    assert MIN_PER_TYPE_N > 10
    assert fold.per_type == {}
    assert fold.stats_for("plan") is fold.pooled


def test_thresholds_for_switches_between_the_arms(records, scores):
    folds = fit_folds(records["hc"], scores, subset="hc")
    fold = next(iter(folds.values()))
    fold.thresholds["execute"] = -1.5
    typed = thresholds_for(folds, typed=True)
    globals_ = thresholds_for(folds, typed=False)
    assert isinstance(typed[fold.held_out], dict)
    assert typed[fold.held_out]["execute"] == -1.5
    assert globals_[fold.held_out] == fold.threshold


# --- stability --------------------------------------------------------------


def test_cv_is_zero_for_a_constant_threshold():
    assert coefficient_of_variation([0.3, 0.3, 0.3]) == 0.0
    assert coefficient_of_variation([0.0, 0.0]) == 0.0


def test_cv_grows_with_spread():
    tight = coefficient_of_variation([-0.5, -0.52, -0.48])
    loose = coefficient_of_variation([-0.5, -2.0, 1.5])
    assert loose > tight


def test_stability_reports_per_subset(records, scores):
    folds = fit_all_subsets(records["alg"] + records["hc"], scores)
    stab = stability(folds)
    assert set(stab) == {"alg", "hc"}
    assert stab["hc"]["n_folds"] == len(records["hc"])
    name, cv = worst_threshold_cv(stab)
    assert cv >= 0 and "/" in name


def test_choose_threshold_handles_a_single_class():
    assert choose_threshold([0.1, 0.2], [True, True]) == 0.0


# --- field sanity -----------------------------------------------------------


def test_field_sanity_flags_a_constant_field(records):
    flat = {
        r.key: [_score(r.key, s.idx, 0.5, s.type_norm) for s in r.steps]
        for r in records["hc"]
    }
    sanity = field_sanity(records["hc"], flat)
    assert sanity["degenerate"]
    assert any("constant" in d for d in sanity["degenerate"])


def test_field_sanity_flags_saturation(records):
    sat = {
        r.key: [
            _score(r.key, s.idx, 1.0 if s.idx % 2 else 0.0, s.type_norm) for s in r.steps
        ]
        for r in records["hc"]
    }
    sanity = field_sanity(records["hc"], sat)
    assert any("saturated" in d for d in sanity["degenerate"])


def test_field_sanity_passes_a_varied_field(records, scores):
    hc = {k: v for k, v in scores.items() if k.startswith("hc/")}
    sanity = field_sanity(records["hc"], hc)
    assert set(sanity["cells"])
    for cell in sanity["cells"].values():
        assert cell["n"] > 0 and "median" in cell
