from __future__ import annotations

import json

from masuq.loaders import load_subset
from masuq.typing_ import classify_step, is_answer_emission, is_orchestrator
from masuq.typing_.validate import validate_against_known


def test_plan_json_detected():
    v = classify_step(
        json.dumps({"analyst_task": "do x", "verifier_task": "check x"}), "StarAgent"
    )
    assert v.type_norm == "plan" and v.rule == "plan_json"


def test_plan_json_inside_string_payload():
    # StarAgent embeds the plan as a JSON *string*; key sniffing must still fire.
    content = 'Here you go: "{\\"analyst_task\\": \\"solve\\"}"'
    assert classify_step(content, "StarAgent").type_norm == "plan"


def test_answer_emission():
    assert is_answer_emission("The answer is 51.")
    assert is_answer_emission("B")
    assert is_answer_emission("\\boxed{42}")
    assert not is_answer_emission("Let me search the web.")


def test_final_only_when_last_or_terminating():
    mid = classify_step("The answer is 51.", "Solver", is_last=False)
    last = classify_step("The answer is 51.", "Solver", is_last=True)
    assert last.type_norm == "final" and last.rule == "answer_emission_last"
    assert mid.type_norm == "final" and mid.rule == "answer_emission"


def test_delegation_language():
    v = classify_step("WebSurfer, please proceed with the search.", "Orchestrator")
    assert v.type_norm == "delegate"


def test_tool_output_is_execute():
    v = classify_step("exitcode: 0\nAddress: http://example.com", "Executor")
    assert v.type_norm == "execute" and v.rule == "tool_output"


def test_is_orchestrator():
    assert is_orchestrator("Orchestrator") and is_orchestrator("chat_manager")
    assert not is_orchestrator("WebSurfer")


def test_validation_report_against_native_types(paths):
    records, _ = load_subset("autogen_mmlu", paths.get("autogen_mmlu"))
    rep = validate_against_known(records, reference_source="native", subset="autogen_mmlu")
    assert rep.n_steps == 48
    assert 0.0 <= rep.agreement <= 1.0
    assert "plan" in rep.confusion
    assert "agreement=" in rep.render()


def test_validation_report_against_parsed_types(paths):
    records, _ = load_subset("hc", paths.get("hc"))
    rep = validate_against_known(records, reference_source="parsed", subset="hc")
    assert rep.n_steps > 0
    assert rep.coverage > 0.0


def test_classifier_never_overrides_known_types(paths):
    records, _ = load_subset("autogen_mmlu", paths.get("autogen_mmlu"))
    from masuq.typing_ import apply_classifier

    before = [s.type_norm for s in records[0].steps]
    apply_classifier(records[0].steps)  # overwrite=False
    assert [s.type_norm for s in records[0].steps] == before
