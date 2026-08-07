"""End-to-end: the manifest order, driven through the CLI with the mock judge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from masattr import specs
from masattr.cli import main


@pytest.fixture(autouse=True, scope="module")
def frozen_specs():
    specs.freeze()


def _base(data_root, tmp_path, name):
    return [
        "--data-root",
        str(data_root),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--out-dir",
        str(tmp_path / name),
        "--n-boot",
        "20",
    ]


def _judge(data_root, tmp_path, extra=()):
    """Score both subsets and return the score-file paths."""
    scores_dir = tmp_path / "scores"
    rc = main(
        ["judge", *_base(data_root, tmp_path, "judge"), "--scores-dir", str(scores_dir), *extra]
    )
    assert rc == 0
    return sorted(str(p) for p in scores_dir.glob("*.jsonl"))


def test_freeze_and_verify_round_trip():
    hashes = specs.freeze()
    assert set(hashes) == {"prompts", "type_rules", "e0_criteria", "judge"}
    assert specs.verify(strict=True) == []


def test_verify_detects_prompt_drift(monkeypatch):
    monkeypatch.setattr(specs, "live_artifacts", lambda: {"prompts": "tampered"})
    with pytest.raises(RuntimeError, match="drifted"):
        specs.verify(strict=True)


def test_load_reports_violations_on_fixtures(data_root, tmp_path, capsys):
    rc = main(["load", *_base(data_root, tmp_path, "load"), "--assert"])
    out = capsys.readouterr().out
    assert rc == 1  # fixtures are not the real corpus; counts must fail loudly
    assert "126 files" in out
    assert '"n_files": 4' in out


def test_typecheck_prints_confusion(data_root, tmp_path, capsys):
    main(["typecheck", *_base(data_root, tmp_path, "tc")])
    text = capsys.readouterr().out
    assert "Type rules vs parsed types" in text
    assert "parsed \\ rule" in text
    assert (tmp_path / "tc" / "typecheck.json").exists()


def test_judge_writes_scores_and_a_manifest(data_root, tmp_path, capsys):
    paths = _judge(data_root, tmp_path)
    capsys.readouterr()
    assert len(paths) == 2
    rows = [json.loads(l) for l in Path(paths[0]).read_text().splitlines()]
    assert {"p_raw", "type_norm", "agent", "with_gt", "use_types"} <= set(rows[0])
    manifest = json.loads((tmp_path / "judge" / "manifest.json").read_text())
    assert manifest["spec_hashes"]["prompts"]
    assert manifest["commit"]


def test_e1_refuses_a_threshold_it_would_have_to_invent(data_root, tmp_path, capsys):
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    with pytest.raises(SystemExit, match="statistics fit on the files being scored"):
        main(["e1", *_base(data_root, tmp_path, "e1"), "--scores", *scores])


def test_e1_primary_table(data_root, tmp_path, capsys):
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    rc = main(["e1", *_base(data_root, tmp_path, "e1"), "--scores", *scores, "--threshold", "0.5"])
    assert rc == 0
    res = json.loads((tmp_path / "e1" / "results.json").read_text())
    cfg = next(iter(res["configs"].values()))
    assert set(cfg["scores"]) == {
        "first_crossing",
        "argmin",
        "changepoint",
        "agent_first",
        "relative_crossing",
    }
    variants = set(cfg["scores"]["first_crossing"])
    assert {"exact/all", "substring/all", "exact/excl_flagged"} <= variants
    md = (tmp_path / "e1" / "results.md").read_text()
    assert "Exact match is primary" in md


def test_e1_reports_hc_role_disagreement(data_root, tmp_path, capsys):
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    main(["e1", *_base(data_root, tmp_path, "e1b"), "--scores", *scores, "--threshold", "0.5"])
    res = json.loads((tmp_path / "e1b" / "results.json").read_text())
    hc = [v for k, v in res["configs"].items() if "subset=hc" in k]
    assert hc and hc[0]["disagreement_by_role"]


def test_ablation_refuses_a_single_arm(data_root, tmp_path, capsys):
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    with pytest.raises(SystemExit, match="not an ablation"):
        main(["e2", *_base(data_root, tmp_path, "e2"), "--scores", *scores, "--threshold", "0.5"])


def test_readout_ablation_with_both_arms(data_root, tmp_path, capsys):
    a = _judge(data_root, tmp_path)
    b = _judge(data_root, tmp_path, extra=["--readout", "verbalized"])
    capsys.readouterr()
    rc = main(
        ["e2", *_base(data_root, tmp_path, "e2b"), "--scores", *a, *b, "--threshold", "0.5"]
    )
    assert rc == 0
    res = json.loads((tmp_path / "e2b" / "results.json").read_text())
    assert "readout" in res["varied_axes"]
    assert len(res["configs"]) == 4  # 2 subsets x 2 readouts


def test_evidence_ablation_includes_the_hindsight_ceiling(data_root, tmp_path, capsys):
    a = _judge(data_root, tmp_path)
    b = _judge(data_root, tmp_path, extra=["--policy", "hindsight"])
    capsys.readouterr()
    rc = main(["e5", *_base(data_root, tmp_path, "e5"), "--scores", *a, *b, "--threshold", "0.5"])
    assert rc == 0
    res = json.loads((tmp_path / "e5" / "results.json").read_text())
    assert any("policy=hindsight" in k for k in res["configs"])


def test_typing_ablation(data_root, tmp_path, capsys):
    a = _judge(data_root, tmp_path)
    b = _judge(data_root, tmp_path, extra=["--no-types"])
    capsys.readouterr()
    rc = main(["e4", *_base(data_root, tmp_path, "e4"), "--scores", *a, *b, "--threshold", "0.5"])
    assert rc == 0
    res = json.loads((tmp_path / "e4" / "results.json").read_text())
    assert "use_types" in res["varied_axes"]


def test_baselines_run_all_three_methods(data_root, tmp_path, capsys):
    rc = main(["baselines", *_base(data_root, tmp_path, "bl"), "--generators", "mock"])
    assert rc == 0
    res = json.loads((tmp_path / "bl" / "results.json").read_text())
    assert {r["method"] for r in res["runs"]} == {"all_at_once", "step_by_step", "binary_search"}
    assert all(r["impl"] == "local" for r in res["runs"])
    md = (tmp_path / "bl" / "results.md").read_text()
    assert "NOT reproduced numbers" in md


def test_baselines_local_impl_is_marked_in_the_manifest(data_root, tmp_path, capsys):
    main(["baselines", *_base(data_root, tmp_path, "bl2"), "--generators", "mock"])
    manifest = json.loads((tmp_path / "bl2" / "manifest.json").read_text())
    assert any("NOT the Who&When reproduction" in n for n in manifest["notes"])


def test_e7_surrogate(data_root, tmp_path, capsys):
    rc = main(["e7", *_base(data_root, tmp_path, "e7")])
    assert rc == 0
    res = json.loads((tmp_path / "e7" / "results.json").read_text())
    assert any(k.startswith("mean_logprob") for k in res["arms"])
    assert any(k.startswith("mean_entropy") for k in res["arms"])


def test_e9_stratifies_e1_output(data_root, tmp_path, capsys):
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    main(["e1", *_base(data_root, tmp_path, "e1c"), "--scores", *scores, "--threshold", "0.5"])
    capsys.readouterr()
    rc = main(
        [
            "e9",
            *_base(data_root, tmp_path, "e9"),
            "--e1-results",
            str(tmp_path / "e1c" / "results.json"),
        ]
    )
    assert rc == 0
    res = json.loads((tmp_path / "e9" / "results.json").read_text())
    assert set(res["strata"]) == {"gold_step_type", "gold_role", "trajectory_length", "subset"}
    assert res["n_predictions"] > 0


def test_cli_usage_lists_the_manifest_order(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "manifest order" in out
    for cmd in ("load", "typecheck", "judge", "e0", "e1", "e9"):
        assert f"  {cmd}" in out


def test_unknown_command_exits_two(capsys):
    assert main(["e8"]) == 2  # E8 is gated behind an owner decision and not built


# --- anomaly policy, retype, and the new arms -------------------------------


def test_load_defaults_to_flag_policy(data_root, tmp_path, capsys):
    rc = main(["load", *_base(data_root, tmp_path, "loadflag")])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0  # `flag` is the default, so the corpus loads
    assert out["alg"]["n_files"] == 4 and out["hc"]["n_files"] == 3


def test_manifest_logs_anomalous_file_ids(parquet_root, tmp_path, capsys):
    scores_dir = tmp_path / "sc"
    main(
        [
            "judge",
            *_base(parquet_root, tmp_path, "manif"),
            "--scores-dir",
            str(scores_dir),
            "--anomaly-policy",
            "flag",
        ]
    )
    capsys.readouterr()
    m = json.loads((tmp_path / "manif" / "manifest.json").read_text())
    assert m["anomalous_files"]["hc"] == ["hc_1"]
    assert m["anomalous_files"]["alg"] == ["alg_1"]
    assert m["model_families"]["judge"] == "mock"


def test_e1_reports_the_anomaly_slice(parquet_root, tmp_path, capsys):
    scores_dir = tmp_path / "sc2"
    main(["judge", *_base(parquet_root, tmp_path, "j2"), "--scores-dir", str(scores_dir)])
    capsys.readouterr()
    scores = sorted(str(p) for p in scores_dir.glob("*.jsonl"))
    main(["e1", *_base(parquet_root, tmp_path, "e1anom"), "--scores", *scores, "--threshold", "0.5"])
    res = json.loads((tmp_path / "e1anom" / "results.json").read_text())
    cfg = next(iter(res["configs"].values()))
    assert "exact/excl_anomalous" in cfg["scores"]["first_crossing"]
    assert "positions" in cfg


def test_retype_gate_blocks_a_useless_splitter(data_root, tmp_path, capsys):
    rc = main(["retype", *_base(data_root, tmp_path, "rt"), "--splitter", "mock"])
    out = capsys.readouterr().out
    assert rc == 1  # the mock splitter cannot beat the baseline
    assert "GATE FAILED" in out
    assert (tmp_path / "rt" / "retype_gate.json").exists()


def test_retype_refuses_a_same_family_splitter(data_root, tmp_path):
    from masattr.models import DisjointnessError

    with pytest.raises(DisjointnessError):
        main(
            [
                "retype",
                *_base(data_root, tmp_path, "rt2"),
                "--splitter",
                "hf:Qwen/Qwen3-8B",
                "--judge",
                "hf:Qwen3.6-35B-A3B",
            ]
        )


def test_e5_prefix_window_is_an_ablation_arm(data_root, tmp_path, capsys):
    a = _judge(data_root, tmp_path)
    b = _judge(data_root, tmp_path, extra=["--prefix-window", "2"])
    capsys.readouterr()
    rc = main(["e5", *_base(data_root, tmp_path, "e5w"), "--scores", *a, *b, "--threshold", "0.5"])
    assert rc == 0
    res = json.loads((tmp_path / "e5w" / "results.json").read_text())
    assert "prefix_window" in res["varied_axes"]




# --- E0 redefined: field sanity + threshold stability -----------------------


def test_e0_refuses_unregistered_criteria(data_root, tmp_path, capsys):
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    with pytest.raises(RuntimeError, match="registered"):
        main(["e0", *_base(data_root, tmp_path, "e0draft"), "--scores", *scores])


def test_e0_reports_field_and_stability(data_root, tmp_path, register_criteria, capsys):
    register_criteria()
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    rc = main(
        [
            "e0",
            *_base(data_root, tmp_path, "e0"),
            "--scores",
            *scores,
            "--folds-out",
            str(tmp_path / "folds.json"),
        ]
    )
    assert rc in (0, 2, 3)
    res = json.loads((tmp_path / "e0" / "results.json").read_text())
    assert res["field_sanity"]["cells"]
    assert set(res["stability"]) == {"alg", "hc"}
    assert (tmp_path / "folds.json").exists()
    md = (tmp_path / "e0" / "results.md").read_text()
    assert "Pre-registered decision" in md


def test_e0_decision_switches_the_primary_rule(data_root, tmp_path, register_criteria, capsys):
    # Register a bound no corpus can meet, so the switch fires.
    register_criteria(max_threshold_cv=-1.0)
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    rc = main(
        [
            "e0",
            *_base(data_root, tmp_path, "e0sw"),
            "--scores",
            *scores,
            "--folds-out",
            str(tmp_path / "folds_sw.json"),
        ]
    )
    assert rc in (2, 3)  # falsified: threshold unstable
    decision = json.loads((tmp_path / "e0sw" / "e0_decision.json").read_text())
    assert decision["threshold_unstable"]
    assert decision["primary_rule"] == "relative_crossing"
    assert decision["criterion_hash"]


def test_e1_uses_the_primary_rule_e0_chose(data_root, tmp_path, register_criteria, capsys):
    register_criteria(max_threshold_cv=-1.0)
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    main(
        [
            "e0",
            *_base(data_root, tmp_path, "e0d"),
            "--scores",
            *scores,
            "--folds-out",
            str(tmp_path / "folds_d.json"),
        ]
    )
    capsys.readouterr()
    rc = main(
        [
            "e1",
            *_base(data_root, tmp_path, "e1d"),
            "--scores",
            *scores,
            "--folds",
            str(tmp_path / "folds_d.json"),
            "--decision",
            str(tmp_path / "e0d" / "e0_decision.json"),
        ]
    )
    assert rc == 0
    res = json.loads((tmp_path / "e1d" / "results.json").read_text())
    assert res["primary_rule"] == "relative_crossing"
    assert res["normalized"] and res["n_folds"] > 0


def test_e1_refuses_to_run_unnormalized(data_root, tmp_path, capsys):
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    with pytest.raises(SystemExit, match="statistics fit on the files being scored"):
        main(["e1", *_base(data_root, tmp_path, "e1u"), "--scores", *scores])


def test_e4_pooled_normalization_arm(data_root, tmp_path, register_criteria, capsys):
    register_criteria()
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    main(
        [
            "e0",
            *_base(data_root, tmp_path, "e0p"),
            "--scores",
            *scores,
            "--folds-out",
            str(tmp_path / "folds_p.json"),
        ]
    )
    capsys.readouterr()
    rc = main(
        [
            "e3",
            *_base(data_root, tmp_path, "e3p"),
            "--scores",
            *scores,
            "--folds",
            str(tmp_path / "folds_p.json"),
            "--pooled-normalization",
            "--global-threshold",
        ]
    )
    assert rc == 0
    res = json.loads((tmp_path / "e3p" / "results.json").read_text())
    assert res["typed_normalization"] is False
    assert res["typed_thresholds"] is False


def test_judge_spec_files_start_as_drafts():
    from masattr import specs

    assert specs.e0_criteria()["status"] == "draft"
    assert specs.judge_spec()["status"] == "draft"
    with pytest.raises(RuntimeError, match="confirmed"):
        specs.require_status("judge", specs.judge_spec(), "confirmed", "why")
