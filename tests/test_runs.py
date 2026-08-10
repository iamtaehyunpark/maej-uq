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


@pytest.fixture(autouse=True)
def _registered(register_criteria):
    """Attribution runs refuse draft criteria, so every run test registers them."""
    return register_criteria()


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


def _folds(data_root, tmp_path, scores, name="fld"):
    """Run E0 to fit the leave-one-out folds the attribution runs require."""
    path = tmp_path / f"{name}.json"
    main(["e0", *_base(data_root, tmp_path, name), "--scores", *scores, "--folds-out", str(path)])
    return str(path)


def test_freeze_and_verify_round_trip():
    hashes = specs.freeze()
    assert set(hashes) == {"prompts", "type_rules", "criteria", "judge", "rule_directive"}
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
    folds = _folds(data_root, tmp_path, scores, "f_e1")
    capsys.readouterr()
    rc = main(["e1", *_base(data_root, tmp_path, "e1"), "--scores", *scores, "--folds", folds])
    assert rc == 0
    res = json.loads((tmp_path / "e1" / "results.json").read_text())
    cfg = next(iter(res["configs"].values()))
    assert set(cfg["scores"]) == {
        "changepoint_single",
        "first_crossing",
        "argmin",
        "changepoint",
        "relative_crossing@1.5",
        "relative_crossing@2.0",
        "relative_crossing@2.5",
    }
    assert res["primary_rule"] == "changepoint_single"
    assert res["rule_provenance"]
    variants = set(cfg["scores"]["first_crossing"])
    assert {"exact/all", "substring/all", "exact/excl_flagged"} <= variants
    md = (tmp_path / "e1" / "results.md").read_text()
    assert "Exact match is primary" in md


def test_e1_reports_hc_role_disagreement(data_root, tmp_path, capsys):
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    folds = _folds(data_root, tmp_path, scores, "f_e1b")
    capsys.readouterr()
    main(["e1", *_base(data_root, tmp_path, "e1b"), "--scores", *scores, "--folds", folds])
    res = json.loads((tmp_path / "e1b" / "results.json").read_text())
    hc = [v for k, v in res["configs"].items() if "subset=hc" in k]
    assert hc and hc[0]["disagreement_by_role"]


def test_ablation_refuses_a_single_arm(data_root, tmp_path, capsys):
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    folds = _folds(data_root, tmp_path, scores, "f_e2")
    capsys.readouterr()
    with pytest.raises(SystemExit, match="not an ablation"):
        main(["e2", *_base(data_root, tmp_path, "e2"), "--scores", *scores, "--folds", folds])


def test_readout_ablation_with_both_arms(data_root, tmp_path, capsys):
    a = _judge(data_root, tmp_path)
    b = _judge(data_root, tmp_path, extra=["--readout", "verbalized"])
    capsys.readouterr()
    rc = main(
        ["e2", *_base(data_root, tmp_path, "e2b"), "--scores", *a, *b, "--folds", _folds(data_root, tmp_path, a + b, "f_e2b")]
    )
    assert rc == 0
    res = json.loads((tmp_path / "e2b" / "results.json").read_text())
    assert "readout" in res["varied_axes"]
    assert len(res["configs"]) == 4  # 2 subsets x 2 readouts


def test_evidence_ablation_includes_the_hindsight_ceiling(data_root, tmp_path, capsys):
    a = _judge(data_root, tmp_path)
    b = _judge(data_root, tmp_path, extra=["--policy", "hindsight"])
    capsys.readouterr()
    rc = main(["e5", *_base(data_root, tmp_path, "e5"), "--scores", *a, *b, "--folds", _folds(data_root, tmp_path, a + b, "f_e5")])
    assert rc == 0
    res = json.loads((tmp_path / "e5" / "results.json").read_text())
    assert any("policy=hindsight" in k for k in res["configs"])


def test_typing_ablation(data_root, tmp_path, capsys):
    a = _judge(data_root, tmp_path)
    b = _judge(data_root, tmp_path, extra=["--no-types"])
    capsys.readouterr()
    rc = main(["e4", *_base(data_root, tmp_path, "e4"), "--scores", *a, *b, "--folds", _folds(data_root, tmp_path, a + b, "f_e4")])
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
    rc = main(["e7", *_base(data_root, tmp_path, "e7"), "--proxy-lm", "mock"])
    assert rc == 0
    res = json.loads((tmp_path / "e7" / "results.json").read_text())
    assert any(k.startswith("mean_logprob") for k in res["arms"])
    assert any(k.startswith("mean_entropy") for k in res["arms"])


def test_e9_stratifies_e1_output(data_root, tmp_path, capsys):
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    folds = _folds(data_root, tmp_path, scores, "f_e1c")
    capsys.readouterr()
    main(["e1", *_base(data_root, tmp_path, "e1c"), "--scores", *scores, "--folds", folds])
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
    assert set(res["strata"]) == {
        "gold_step_type",
        "gold_role",
        "trajectory_length",
        "subset",
        "level",
    }
    # No level in the parquet, so the axis is present and honestly empty.
    assert [r["stratum"] for r in res["strata"]["level"]] == ["absent"]
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
    folds = _folds(parquet_root, tmp_path, scores, "f_anom")
    capsys.readouterr()
    main(["e1", *_base(parquet_root, tmp_path, "e1anom"), "--scores", *scores, "--folds", folds])
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
    rc = main(["e5", *_base(data_root, tmp_path, "e5w"), "--scores", *a, *b, "--folds", _folds(data_root, tmp_path, a + b, "f_e5w")])
    assert rc == 0
    res = json.loads((tmp_path / "e5w" / "results.json").read_text())
    assert "prefix_window" in res["varied_axes"]




# --- E0 redefined: field sanity + threshold stability -----------------------


def test_e0_reports_field_and_stability(data_root, tmp_path, capsys):
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
    assert rc in (0, 3)  # 3 only when the field itself is degenerate
    res = json.loads((tmp_path / "e0" / "results.json").read_text())
    assert res["field_sanity"]["cells"]
    assert set(res["stability"]) == {"alg", "hc"}
    assert "decides_nothing" in res
    assert (tmp_path / "folds.json").exists()
    md = (tmp_path / "e0" / "results.md").read_text()
    assert "decides nothing" in md
    # E0 no longer writes a decision, and nothing downstream reads one.
    assert not (tmp_path / "e0" / "e0_decision.json").exists()


def test_e1_refuses_to_run_unnormalized(data_root, tmp_path, capsys):
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    with pytest.raises(SystemExit, match="statistics fit on the files being scored"):
        main(["e1", *_base(data_root, tmp_path, "e1u"), "--scores", *scores])


def test_e4_pooled_normalization_arm(data_root, tmp_path, capsys):
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


def test_judge_roles_are_declared_and_pairwise_disjoint():
    from masattr import specs

    assert specs.check_families() == []
    assert specs.role("judge_primary")["status"] == "confirmed"
    assert specs.client_spec("judge_primary") == "served:Qwen/Qwen3.6-35B-A3B"
    assert specs.client_spec("judge_primary", "hf") == "hf:Qwen/Qwen3.6-35B-A3B"


def test_a_draft_role_refuses_to_resolve(tmp_path, monkeypatch):
    import json as _json

    from masattr import specs
    from masattr.runs._shared import resolve_model

    blob = specs.judge_spec()
    blob["judge_secondary"] = {**blob["judge_secondary"], "status": "draft"}
    path = tmp_path / "judge.json"
    path.write_text(_json.dumps(blob))
    monkeypatch.setattr(specs, "JUDGE_FILE", path)

    with pytest.raises(RuntimeError, match="not 'confirmed'"):
        resolve_model("judge_secondary")
    assert resolve_model("judge_primary") == "served:Qwen/Qwen3.6-35B-A3B"
    assert resolve_model("mock") == "mock"
    assert resolve_model("hf:some/other-model") == "hf:some/other-model"


def test_declared_family_is_verified_not_trusted(tmp_path, monkeypatch):
    import json as _json

    from masattr import specs

    blob = specs.judge_spec()
    blob["judge_primary"] = {**blob["judge_primary"], "family": "llama"}
    path = tmp_path / "judge.json"
    path.write_text(_json.dumps(blob))
    monkeypatch.setattr(specs, "JUDGE_FILE", path)
    problems = specs.check_families(strict=False)
    assert any("resolves to 'qwen'" in p for p in problems)


def test_a_same_family_type_classifier_is_caught(tmp_path, monkeypatch):
    import json as _json

    from masattr import specs

    blob = specs.judge_spec()
    blob["type_classifier"] = {
        "id": "Qwen/Qwen3-8B",
        "family": "qwen",
        "status": "confirmed",
    }
    path = tmp_path / "judge.json"
    path.write_text(_json.dumps(blob))
    monkeypatch.setattr(specs, "JUDGE_FILE", path)
    problems = specs.check_families(strict=False)
    assert any("type_classifier" in p and "both family" in p for p in problems)
    with pytest.raises(RuntimeError, match="confirmed"):
        specs.require_status("judge", specs.judge_spec(), "confirmed", "why")


# --- Step-1 config, verified against the registered files -------------------


def test_criteria_are_registered_and_in_z_units():
    from masattr import specs

    c = specs.criteria()
    assert c["status"] == "registered"
    assert c["changepoint_min_contrast"] == 1.0
    assert c["changepoint_min_contrast_units"] == "z"


def test_all_roles_confirmed_and_disjoint():
    from masattr import specs
    from masattr.runs._shared import resolve_model

    assert specs.check_families() == []
    for role in ("judge_primary", "judge_secondary", "type_classifier", "proxy_lm"):
        assert specs.role(role)["status"] == "confirmed"
        assert resolve_model(role).startswith("served:")


def test_primary_rule_refuses_unnormalized_scores(data_root, tmp_path, capsys):
    scores = _judge(data_root, tmp_path)
    capsys.readouterr()
    # The registered contrast bound is in z-units; --threshold must not be a way
    # to run the primary rule against raw scores.
    with pytest.raises(SystemExit, match="z-normalized units"):
        main(["e1", *_base(data_root, tmp_path, "e1raw"), "--scores", *scores, "--threshold", "0.5"])


def test_smoke_runs_all_three_arms_and_gates(data_root, tmp_path, capsys):
    rc = main(
        [
            "smoke",
            *_base(data_root, tmp_path, "smk"),
            "--scores-dir",
            str(tmp_path / "smk_scores"),
            "--n-files",
            "4",
        ]
    )
    out = capsys.readouterr().out
    assert rc in (0, 4)
    res = json.loads((tmp_path / "smk" / "results.json").read_text())
    assert set(res["arms"]) == {
        "W0_nogt", "W0_gt", "W_resp_nogt", "W_resp_gt", "W_own_nogt", "W_own_gt",
    }
    assert "GATE" in out
    assert (tmp_path / "smk" / "curves.csv").exists()


def test_smoke_curves_mark_the_annotated_step(data_root, tmp_path, capsys):
    import csv as _csv

    main(
        [
            "smoke",
            *_base(data_root, tmp_path, "smk2"),
            "--scores-dir",
            str(tmp_path / "smk2_scores"),
            "--n-files",
            "4",
        ]
    )
    capsys.readouterr()
    with (tmp_path / "smk2" / "curves.csv").open() as fh:
        rows = list(_csv.DictReader(fh))
    assert rows
    assert {"arm", "with_gt", "file", "step", "p_raw", "is_mistake_step"} <= set(rows[0])
    # Exactly one marked step per (arm, gt, file).
    from collections import Counter

    marked = Counter(
        (r["arm"], r["with_gt"], r["file"]) for r in rows if r["is_mistake_step"] == "1"
    )
    assert marked and set(marked.values()) == {1}


def test_smoke_arms_differ_only_in_lookahead():
    from masattr.runs.smoke import ARMS

    # The arms are a lookahead axis. Base assembly — including the near-empty
    # rescue — is identical across them; conflating the two was a real bug.
    assert ARMS == (("W0", "none"), ("W+resp", "resp"), ("W+own", "own"))


# --- level axis and baseline transport --------------------------------------


def test_level_carries_its_scale_and_never_mixes_them(records):
    from masattr.loaders._common import level_of

    assert level_of({"level": "2"}) == ("2", "numeric")
    assert level_of({"level": 3}) == ("3", "numeric")
    assert level_of({"level": "Hard"}) == ("Hard", "verbal")
    assert level_of({}) == ("", "absent")
    # Untouched by the parquet path, which drops the column entirely.
    assert all(r.level_scale == "absent" for r in records["hc"])


def test_enrich_levels_joins_on_question_id(tmp_path, records):
    import json as _json

    from masattr.loaders._common import enrich_levels

    d = tmp_path / "theirs"
    d.mkdir()
    hc = records["hc"]
    (d / "1.json").write_text(_json.dumps({"question_ID": hc[0].file_id, "level": "2"}))
    (d / "2.json").write_text(_json.dumps({"question_ID": "not-in-corpus", "level": "3"}))
    out = enrich_levels(hc, d)
    assert out[0].level == "2" and out[0].level_scale == "numeric"
    assert all(r.level_scale == "absent" for r in out[1:])


def test_baseline_command_has_no_azure_arguments():
    from masattr.baselines.whowhen_repo import repo_command

    cmd = repo_command(
        Path("inference.py"),
        method="all_at_once",
        model="gpt-4o",
        directory_path="d",
        is_handcrafted=False,
        api_key="k",
    )
    assert "--azure_endpoint" not in cmd and "--api_version" not in cmd
    assert cmd[cmd.index("--api_key") + 1] == "k"


def test_openai_shim_is_valid_and_redirects_azure(tmp_path):
    import py_compile

    from masattr.baselines.whowhen_repo import write_openai_shim

    d = write_openai_shim(tmp_path / "snap.txt")
    shim = d / "sitecustomize.py"
    py_compile.compile(str(shim), doraise=True)
    src = shim.read_text()
    # It redirects rather than patching their tree, and drops the Azure-only args.
    assert "openai.AzureOpenAI = _Client" in src
    assert "azure_endpoint" in src and "api_version" in src
    assert "MASATTR_SNAPSHOT_RECEIPT" in src


def test_smoke_sample_covers_the_required_cases(records):
    from masattr.runs.smoke import REQUIRED, sample_files
    from masattr.typing.normalize import is_orchestrator

    pool = records["alg"] + records["hc"]
    picked, covered = sample_files(pool, 4, seed=0)
    assert len(picked) == 4
    assert set(covered) == {name for name, _, _ in REQUIRED}
    # Where a case exists in the corpus it must be represented, not left to luck.
    for name, subset, pred in REQUIRED:
        exists = any(r.subset == subset and pred(r) for r in pool)
        if exists:
            assert covered[name], f"{name} exists in the corpus but was not sampled"
            assert covered[name] in {r.key for r in picked}
    _ = is_orchestrator


def test_smoke_sample_is_deterministic(records):
    from masattr.runs.smoke import sample_files

    pool = records["alg"] + records["hc"]
    a, _ = sample_files(pool, 4, seed=0)
    b, _ = sample_files(pool, 4, seed=0)
    assert [r.key for r in a] == [r.key for r in b]


def test_roles_resolve_per_transport():
    from masattr.runs._shared import resolve_model

    assert resolve_model("judge_primary", "served") == "served:Qwen/Qwen3.6-35B-A3B"
    assert resolve_model("judge_primary", "hf") == "hf:Qwen/Qwen3.6-35B-A3B"
    assert resolve_model("mock", "served") == "mock"


def test_served_client_sums_answer_mass_and_disables_thinking():
    """P(True) must renormalise over summed spellings, and the request must
    carry the template's own thinking switch rather than a prompt hack."""
    from masattr.judge.client import ServedClient

    c = ServedClient.__new__(ServedClient)
    c.model, c.system, c.top_logprobs = "m", "sys", 20
    c._prefix, c.n_calls, c.n_topk_miss = "PREFIX", 0, 0

    body = c._chat_body("READOUT", max_tokens=1)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1]["content"] == "PREFIXREADOUT"
    assert "<think>" not in body["messages"][1]["content"]

    import math

    def fake_post(path, payload):
        assert path == "/chat/completions"
        lp = lambda t, p: {"token": t, "logprob": math.log(p)}
        return {
            "choices": [{"logprobs": {"content": [{"top_logprobs": [
                lp("True", 0.30), lp(" True", 0.30), lp("False", 0.20), lp("x", 0.20),
            ]}]}}],
            "usage": {},
        }

    c._post = fake_post
    p, tr = c.p_true("READOUT")
    # 0.6 true vs 0.2 false -> 0.75, not the 0.6 a max() over spellings gives
    assert abs(p - 0.75) < 1e-9
    assert abs(tr.extra["mass_on_answer"] - 0.8) < 1e-9
