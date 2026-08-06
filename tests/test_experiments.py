"""End-to-end wiring: load → judge (mock) → calibrate → aggregate → attribute."""

from __future__ import annotations

from masuq.judge import MockGenerator, MockPrefixScorer, group_by_record, judge_corpus
from masuq.loaders import join_labels, load_subset
from masuq.experiments import (
    exp0_calibration_transfer as exp0,
    exp_attribution,
    exp_baselines,
    exp_label_audit,
    exp_trajectory,
)


def _matu(paths, subset):
    records, _ = load_subset(subset, paths.get(subset))
    join_labels(records, paths.get(f"{subset}_labels"))
    scorer = MockPrefixScorer(seed=1)
    grouped = group_by_record(
        [s for ts in judge_corpus(records, scorer) for s in ts.scores]
    )
    return records, grouped


def _whowhen(paths, subset):
    records, _ = load_subset(subset, paths.get(subset))
    scorer = MockPrefixScorer(seed=2)
    grouped = group_by_record(
        [s for ts in judge_corpus(records, scorer) for s in ts.scores]
    )
    return records, grouped


def test_exp0_runs_and_writes_artifacts(paths, tmp_path):
    fit_r, fit_s = _matu(paths, "autogen_mmlu")
    test_r, test_s = _matu(paths, "camel_math")
    res, cal = exp0.run(fit_r, fit_s, test_r, test_s, out_dir=tmp_path)
    assert res.n_fit > 0 and res.n_test > 0
    assert isinstance(res.transfers, bool)
    assert cal.frozen
    assert (tmp_path / "calibrator_frozen.json").exists()
    assert (tmp_path / "threshold.json").exists()
    assert "Experiment 0" in (tmp_path / "exp0.md").read_text()
    # The decision is stated either way — it is the falsifier, not a formality.
    assert res.fallback in res.to_dict()["decision"]


def test_exp0_fallback_loo_calibration(paths):
    records, grouped = _matu(paths, "camel_math")
    labels = {r.key: bool(r.label_correct) for r in records}
    out = exp0.loo_calibrate(grouped, labels)
    assert set(out) == set(grouped)
    assert all(0.0 <= v <= 1.0 for vs in out.values() for v in vs)


def test_trajectory_track(paths, tmp_path):
    cells = {s: _matu(paths, s) for s in ("autogen_mmlu", "camel_math")}
    results = exp_trajectory.run(cells, use_calibrated=False, out_dir=tmp_path, n_boot=50)
    for r in results.values():
        assert r.n_trajectories == 12
        assert "noisy_or" in r.aggregators
        assert "AUROC" in r.render()
    assert (tmp_path / "trajectory.json").exists()


def test_attribution_track_reports_all_four_variants(paths, tmp_path):
    subsets = {s: _whowhen(paths, s) for s in ("alg", "hc")}
    results = exp_attribution.run(subsets, threshold=0.5, use_calibrated=False, out_dir=tmp_path, n_boot=50)
    alg = results["alg"]
    variants = alg.scores["first_crossing"]
    assert set(variants) == {
        "exact/all",
        "exact/excl_flagged",
        "substring/all",
        "substring/excl_flagged",
    }
    assert variants["exact/all"]["n"] == 4
    assert variants["exact/excl_flagged"]["n"] == 3  # one flagged file dropped
    assert results["hc"].disagreement_role  # orchestrator/worker only for HC
    assert (tmp_path / "attribution.md").exists()


def test_baselines_all_three_methods(paths, tmp_path):
    records, _ = load_subset("hc", paths.get("hc"))
    results = exp_baselines.run(
        {"hc": records}, {"mock": MockGenerator()}, out_dir=tmp_path, n_boot=20
    )
    assert {r.method for r in results} == {"all_at_once", "step_by_step", "binary_search"}
    assert all(r.n == 3 for r in results)
    assert all("exact/all" in r.scores for r in results)
    assert (tmp_path / "baselines.md").exists()


def test_baseline_answer_parsing():
    assert exp_baselines.parse_answer('{"agent": "WebSurfer", "step": 3}') == ("WebSurfer", 3)
    assert exp_baselines.parse_answer("agent: Coder\nstep: 7") == ("Coder", 7)
    assert exp_baselines.parse_answer("no idea") == (None, None)


def test_label_audit(paths, tmp_path):
    records, _ = load_subset("autogen_mmlu", paths.get("autogen_mmlu"))
    join_labels(records, paths.get("autogen_mmlu_labels"))
    judges = [MockGenerator(), MockGenerator(), MockGenerator()]
    report = exp_label_audit.run(
        records, judges, subset="autogen_mmlu", n=10, out_dir=tmp_path
    )
    assert report.n == 10
    assert 0.0 <= report.agreement <= 1.0
    assert (tmp_path / "label_audit_autogen_mmlu.md").exists()


def test_label_audit_requires_three_judges(paths):
    records, _ = load_subset("autogen_mmlu", paths.get("autogen_mmlu"))
    try:
        exp_label_audit.run(records, [MockGenerator()], subset="autogen_mmlu", n=2)
    except ValueError as e:
        assert "3-judge" in str(e)
    else:
        raise AssertionError("expected a ValueError for the wrong judge count")
