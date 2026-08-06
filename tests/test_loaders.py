from __future__ import annotations

import json

import pytest

from masattr.loaders._common import LoaderError, check_expectations
from masattr.loaders.whowhen_ag import load as load_alg
from masattr.loaders.whowhen_hc import load as load_hc
from masattr.record import FLAG_AGENT_STEP_MISMATCH, RecordError
from masattr.typing.normalize import collapse_orchestrator, parse_hc_role


def test_alg_shape(records):
    alg = records["alg"]
    assert len(alg) == 4
    r = alg[0]
    assert r.subset == "alg" and r.dataset == "whowhen"
    assert r.label_mistake_step == 1  # cast from the string "1"
    assert r.steps[0].agent == "Manager"  # agent := name, not role
    assert all(s.type_source == "classified" for s in r.steps)


def test_hc_parsed_types(records):
    r = records["hc"][0]
    assert [s.type_norm for s in r.steps] == ["plan", "delegate", "execute", "final"]
    assert r.steps[1].agent == "Orchestrator"  # delegation keeps the orchestrator as actor
    assert r.steps[2].agent == "WebSurfer"
    assert r.steps[0].type_source == "parsed"


@pytest.mark.parametrize(
    "role,expected",
    [
        ("Orchestrator (thought)", ("Orchestrator", "plan", None)),
        ("Orchestrator (-> WebSurfer)", ("Orchestrator", "delegate", "WebSurfer")),
        ("Orchestrator (final answer)", ("Orchestrator", "final", None)),
        ("WebSurfer", ("WebSurfer", "execute", None)),
        ("FileSurfer", ("FileSurfer", "execute", None)),
    ],
)
def test_parse_hc_role(role, expected):
    assert parse_hc_role(role) == expected


def test_agent_step_mismatch_flag(records):
    flagged = [r for r in records["alg"] if FLAG_AGENT_STEP_MISMATCH in r.flags]
    assert [r.file_id for r in flagged] == ["alg_3"]
    assert flagged[0].label_mistake_agent == "Coder"


def test_orchestrator_collapse_prevents_spurious_flags():
    assert collapse_orchestrator("Orchestrator (thought)") == "orchestrator"
    assert collapse_orchestrator("MagenticOneOrchestrator") == "orchestrator"
    assert collapse_orchestrator("chat_manager") == "orchestrator"
    assert collapse_orchestrator("WebSurfer") == "websurfer"


def test_records_are_frozen(records):
    r = records["alg"][0]
    with pytest.raises(Exception):
        r.file_id = "nope"  # type: ignore[misc]


def test_uncastable_mistake_step_hard_fails(tmp_path):
    d = tmp_path / "alg"
    d.mkdir()
    (d / "bad.json").write_text(
        json.dumps(
            {
                "question": "q",
                "ground_truth": "g",
                "mistake_agent": "A",
                "mistake_step": "not-a-number",
                "history": [{"role": "assistant", "name": "A", "content": "x"}],
            }
        )
    )
    with pytest.raises(LoaderError, match="uncastable mistake_step"):
        load_alg(d)


def test_out_of_range_mistake_step_hard_fails(tmp_path):
    d = tmp_path / "alg"
    d.mkdir()
    (d / "bad.json").write_text(
        json.dumps(
            {
                "question": "q",
                "ground_truth": "g",
                "mistake_agent": "A",
                "mistake_step": "7",
                "history": [{"role": "assistant", "name": "A", "content": "x"}],
            }
        )
    )
    with pytest.raises(RecordError, match="outside"):
        load_alg(d)


def test_empty_step_content_hard_fails(tmp_path):
    d = tmp_path / "alg"
    d.mkdir()
    (d / "bad.json").write_text(
        json.dumps(
            {
                "question": "q",
                "ground_truth": "g",
                "mistake_agent": "A",
                "mistake_step": "0",
                "history": [{"role": "assistant", "name": "A", "content": "   "}],
            }
        )
    )
    with pytest.raises(RecordError, match="empty content"):
        load_alg(d)


def test_missing_source_hard_fails(tmp_path):
    with pytest.raises(LoaderError, match="missing or unrecognised subset source"):
        load_hc(tmp_path / "nope")


def test_expectations_fail_loudly_on_fixtures(records):
    # The fixtures are not the real corpus, so the pre-registered counts must
    # report violations rather than quietly passing.
    problems = check_expectations(records["alg"], "alg", strict=False)
    assert any("126 files" in p for p in problems)
    with pytest.raises(AssertionError):
        check_expectations(records["hc"], "hc")


# --- released parquet format ------------------------------------------------


def test_loads_the_released_parquet(parquet_root):
    alg, report = load_alg(
        parquet_root / "who_and_when" / "Algorithm-Generated.parquet", anomaly_policy="flag"
    )
    assert len(alg) == 4
    assert {r.file_id for r in alg} == {f"alg_{i}" for i in range(4)}  # from question_ID
    assert alg[0].ground_truth == "France"
    assert report.n_steps == 12


def test_hc_groundtruth_column_has_no_underscore(parquet_root):
    hc, _ = load_hc(
        parquet_root / "who_and_when" / "Hand-Crafted.parquet", anomaly_policy="flag"
    )
    # The release spells it `groundtruth` on HC and `ground_truth` on AG; missing
    # the alias would silently empty the reference for the with-GT setting.
    assert all(r.ground_truth == "Lima" for r in hc)


def test_anomaly_policy_fail_names_the_conflict(parquet_root):
    with pytest.raises(LoaderError, match="Both cannot hold"):
        load_hc(parquet_root / "who_and_when" / "Hand-Crafted.parquet")


def test_anomaly_policy_flag_keeps_and_marks(parquet_root):
    hc, report = load_hc(
        parquet_root / "who_and_when" / "Hand-Crafted.parquet", anomaly_policy="flag"
    )
    assert len(hc) == 3  # count assert survives
    bad = [r for r in hc if r.is_anomalous]
    assert [r.file_id for r in bad] == ["hc_1"]
    assert "mistake_step_out_of_range" in bad[0].flags
    assert report.counters["mistake_step_out_of_range"] == 1


def test_anomaly_policy_drop_breaks_the_count(parquet_root):
    hc, report = load_hc(
        parquet_root / "who_and_when" / "Hand-Crafted.parquet", anomaly_policy="drop"
    )
    assert len(hc) == 2  # which is exactly why `drop` is not the default
    assert report.counters["dropped"] == 1


def test_out_of_range_is_not_double_counted_as_agent_mismatch(parquet_root):
    hc, _ = load_hc(
        parquet_root / "who_and_when" / "Hand-Crafted.parquet", anomaly_policy="flag"
    )
    bad = next(r for r in hc if r.is_anomalous)
    assert FLAG_AGENT_STEP_MISMATCH not in bad.flags


def test_empty_content_flagged_on_alg(parquet_root):
    alg, report = load_alg(
        parquet_root / "who_and_when" / "Algorithm-Generated.parquet", anomaly_policy="flag"
    )
    assert report.counters["empty_step_content"] == 1
    assert [r.file_id for r in alg if r.is_anomalous] == ["alg_1"]


def test_paths_resolve_parquet_first(parquet_root):
    from masattr.paths import resolve

    p = resolve(root=parquet_root)
    assert p.get("alg").name == "Algorithm-Generated.parquet"
    assert p.get("hc").exists()
