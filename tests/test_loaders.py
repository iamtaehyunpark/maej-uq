from __future__ import annotations

import json
import pickle

import pytest

from masuq.loaders import join_labels, load_subset
from masuq.loaders.labels import LabelJoinError, flatten_accuracy_dict
from masuq.loaders.whowhen import collapse_orchestrator, parse_hc_role
from masuq.schema import FLAG_AGENT_STEP_MISMATCH


def test_camel_loader_shape(paths):
    records, report = load_subset("camel_math", paths.get("camel_math"))
    assert len(records) == 12  # 3 tasks x 4 runs
    assert report.n_steps == 48
    r = records[0]
    assert r.dataset == "matu" and r.subset == "camel_math"
    assert r.query == "Compute 17 * 3."
    assert all(s.type_source == "classified" for s in r.steps)
    assert all(s.agent == s.role_raw for s in r.steps)


def test_autogen_native_types(paths):
    records, report = load_subset("autogen_mmlu", paths.get("autogen_mmlu"))
    assert len(records) == 12
    r = records[0]
    assert r.query is None  # spec §0: task-id key only
    assert [s.type_norm for s in r.steps] == ["plan", "execute", "execute", "final"]
    assert all(s.type_source == "native" for s in r.steps)
    assert r.steps[0].agent == "StarAgent"
    assert report.counters["type_plan"] == 12


def test_autogen_unmapped_type_degrades_to_classified(tmp_path):
    blob = {"t": [[{"role": "a", "agent": "A", "turn": 0, "type": "wat", "output": "hello"}]]}
    p = tmp_path / "log.json"
    p.write_text(json.dumps(blob))
    records, report = load_subset("autogen_mmlu", p)
    assert report.counters["type_unmapped"] == 1
    # It must not claim a native type it does not understand.
    assert records[0].steps[0].type_source == "classified"


def test_whowhen_ag(paths):
    records, report = load_subset("alg", paths.get("alg"))
    assert len(records) == 4
    r = records[0]
    assert r.label_correct is None  # W&W carries no per-run correctness label
    assert r.label_mistake_step == 1  # cast from the string "1"
    assert r.steps[0].agent == "Manager"  # agent := name, not role


def test_whowhen_hc_role_parsing(paths):
    records, _ = load_subset("hc", paths.get("hc"))
    r = records[0]
    assert [s.type_norm for s in r.steps] == ["plan", "delegate", "execute", "final"]
    assert r.steps[1].agent == "Orchestrator"  # delegation keeps the orchestrator as actor
    assert r.steps[2].agent == "WebSurfer"
    assert r.steps[0].type_source == "parsed"


@pytest.mark.parametrize(
    "role,expected",
    [
        ("Orchestrator (thought)", ("Orchestrator", "plan", None)),
        ("Orchestrator (-> WebSurfer)", ("Orchestrator", "delegate", "WebSurfer")),
        ("WebSurfer", ("WebSurfer", "execute", None)),
        ("FileSurfer", ("FileSurfer", "execute", None)),
        ("Assistant", ("Assistant", "execute", None)),
    ],
)
def test_parse_hc_role(role, expected):
    assert parse_hc_role(role) == expected


def test_agent_step_mismatch_flag(paths):
    records, report = load_subset("alg", paths.get("alg"))
    flagged = [r for r in records if FLAG_AGENT_STEP_MISMATCH in r.flags]
    assert len(flagged) == 1
    assert flagged[0].label_mistake_agent == "Coder"
    assert report.counters["agent_step_mismatch"] == 1


def test_orchestrator_collapse_prevents_spurious_flags():
    assert collapse_orchestrator("Orchestrator (thought)") == "orchestrator"
    assert collapse_orchestrator("MagenticOneOrchestrator") == "orchestrator"
    assert collapse_orchestrator("chat_manager") == "orchestrator"
    assert collapse_orchestrator("WebSurfer") == "websurfer"


def test_mistake_step_cast_failure_is_flagged(tmp_path):
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
    records, report = load_subset("alg", d)
    assert records[0].label_mistake_step is None
    assert "mistake_step_cast_failed" in records[0].flags
    assert report.counters["mistake_step_cast_failed"] == 1


def test_label_join_strict(paths):
    records, _ = load_subset("camel_math", paths.get("camel_math"))
    report = join_labels(records, paths.get("camel_math_labels"))
    assert report.n_joined == len(records)
    assert report.key_layout == "task_to_list"
    assert all(r.label_correct is not None for r in records)


def test_label_join_hard_fails_on_misalignment(paths, tmp_path):
    records, _ = load_subset("camel_math", paths.get("camel_math"))
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(pickle.dumps({"task_999": [True, True, True, True]}))
    with pytest.raises(LabelJoinError, match="alignment failed"):
        join_labels(records, bad)


def test_flatten_accuracy_dict_layouts():
    flat, layout = flatten_accuracy_dict({"a": [True, False]})
    assert flat == {("a", 0): True, ("a", 1): False} and layout == "task_to_list"
    flat, layout = flatten_accuracy_dict({"a": {0: 1, 1: 0}})
    assert flat == {("a", 0): True, ("a", 1): False} and layout == "task_to_runmap"
    flat, layout = flatten_accuracy_dict({("a", 0): True})
    assert layout == "flat_tuple_key"


def test_mixed_layouts_rejected():
    with pytest.raises(Exception):
        flatten_accuracy_dict({"a": [True], "b": {0: True}})
