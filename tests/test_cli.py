from __future__ import annotations

import json

from masuq.cli import main


def _base(data_root, cache):
    return ["--data-root", str(data_root), "--cache-dir", str(cache)]


def test_paths_reports_status(data_root, tmp_path, capsys):
    rc = main([*_base(data_root, tmp_path / "c"), "paths"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert all(v["exists"] for v in out["entries"].values())


def test_load_all_subsets(data_root, tmp_path, capsys):
    rc = main([*_base(data_root, tmp_path / "c"), "load"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert set(out) == {"camel_math", "autogen_mmlu", "alg", "hc"}
    assert out["alg"]["n_records"] == 4
    assert out["hc"]["n_steps"] == 12


def test_load_assert_reports_violations_without_crashing(data_root, tmp_path, capsys):
    # The fixtures deliberately do not match the real corpus sizes, so the
    # pre-registered expectations must fail loudly rather than silently pass.
    rc = main([*_base(data_root, tmp_path / "c"), "load", "--subset", "alg", "--assert"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["alg"]["expectation_violations"]


def test_typecheck_prints_confusion(data_root, tmp_path, capsys):
    main([*_base(data_root, tmp_path / "c"), "typecheck"])
    text = capsys.readouterr().out
    assert "Classifier vs native types" in text
    assert "ref \\ pred" in text


def test_smoke_reports_distribution_per_type(data_root, tmp_path, capsys):
    rc = main([*_base(data_root, tmp_path / "c"), "smoke", "--subset", "hc", "--n-steps", "8"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["hc"]["p_by_type"]
    assert out["hc"]["cost"]["n_assessments"] > 0


def test_judge_writes_scores(data_root, tmp_path, capsys):
    scores = tmp_path / "scores.jsonl"
    rc = main(
        [
            *_base(data_root, tmp_path / "c"),
            "judge",
            "--subset",
            "hc",
            "--out-scores",
            str(scores),
        ]
    )
    assert rc == 0
    assert scores.exists()
    rows = [json.loads(l) for l in scores.read_text().splitlines()]
    assert len(rows) == 12
    assert {"p_raw", "type_norm", "agent"} <= set(rows[0])


def test_attribution_refuses_a_threshold_it_would_have_to_invent(data_root, tmp_path, capsys):
    scores = tmp_path / "s.jsonl"
    main([*_base(data_root, tmp_path / "c"), "judge", "--subset", "hc", "--out-scores", str(scores)])
    capsys.readouterr()
    try:
        main(
            [
                *_base(data_root, tmp_path / "c"),
                "attribution",
                "--subsets",
                "hc",
                "--scores",
                str(scores),
            ]
        )
    except SystemExit as e:
        assert "leak" in str(e)
    else:
        raise AssertionError("expected the CLI to refuse an unspecified threshold")


def test_full_pipeline_through_cli(data_root, tmp_path, capsys):
    base = _base(data_root, tmp_path / "c")
    fit = tmp_path / "fit.jsonl"
    test = tmp_path / "test.jsonl"
    main([*base, "judge", "--subset", "autogen_mmlu", "--out-scores", str(fit)])
    main([*base, "judge", "--subset", "camel_math", "--out-scores", str(test)])
    capsys.readouterr()

    rc = main(
        [
            *base,
            "exp0",
            "--fit-scores",
            str(fit),
            "--test-scores",
            str(test),
            "--out-dir",
            str(tmp_path / "exp0"),
        ]
    )
    assert rc in (0, 2)  # 2 = falsified, which is a legitimate outcome
    assert (tmp_path / "exp0" / "calibrator_frozen.json").exists()

    hc = tmp_path / "hc.jsonl"
    main([*base, "judge", "--subset", "hc", "--out-scores", str(hc)])
    capsys.readouterr()
    rc = main(
        [
            *base,
            "attribution",
            "--subsets",
            "hc",
            "--scores",
            str(hc),
            "--calibrator",
            str(tmp_path / "exp0" / "calibrator_frozen.json"),
            "--threshold-file",
            str(tmp_path / "exp0" / "threshold.json"),
            "--n-boot",
            "20",
            "--out-dir",
            str(tmp_path / "attr"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "attr" / "attribution.json").exists()
