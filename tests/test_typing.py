from __future__ import annotations

import json

from masattr.typing.normalize import classify, is_answer_emission, is_orchestrator
from masattr.typing.validate import validate
from masattr.typing.validate import audit_sample


def test_plan_json_detected():
    v = classify(json.dumps({"analyst_task": "do x", "verifier_task": "check x"}), "StarAgent")
    assert v.type_norm == "plan" and v.rule == "plan_json"


def test_plan_json_embedded_as_string():
    assert classify('Here: "{\\"analyst_task\\": \\"solve\\"}"', "StarAgent").type_norm == "plan"


def test_answer_emission():
    assert is_answer_emission("The answer is 51.")
    assert is_answer_emission("B")
    assert not is_answer_emission("Let me search the web.")


def test_final_needs_last_or_terminate():
    assert classify("The answer is 51.", "Solver", is_last=True).rule == "answer_emission_last"
    assert classify("Done. TERMINATE", "Solver").type_norm == "final"


def test_delegation_forms():
    assert classify("WebSurfer, please proceed with the search.", "Orchestrator").type_norm == "delegate"
    assert classify("WebSurfer, find the capital of Peru.", "Orchestrator").rule == "addressed_agent"
    assert classify("Next speaker: Coder", "Manager").type_norm == "delegate"


def test_tool_output_is_execute():
    assert classify("exitcode: 0\nAddress: http://example.com", "Executor").rule == "tool_output"


def test_is_orchestrator():
    assert is_orchestrator("Orchestrator") and is_orchestrator("chat_manager")
    assert not is_orchestrator("WebSurfer")


def test_rules_never_override_parsed_types(records):
    hc = records["hc"][0]
    parsed = [(s.type_norm, s.type_source) for s in hc.steps if s.type_source == "parsed"]
    assert parsed  # HC has parsed types...
    assert all(src == "parsed" for _, src in parsed)  # ...and they survived the rule pass


def test_validation_gate_reports_a_confusion_matrix(records):
    rep = validate(records["hc"], subset="hc")
    assert rep.n_steps > 0
    assert 0.0 <= rep.agreement <= 1.0
    assert "parsed \\ rule" in rep.render()
    assert isinstance(rep.passes, bool)
    assert rep.per_rule  # per-rule precision is auditable, not just the total


def test_audit_sample_is_deterministic(records):
    pool = [r for recs in records.values() for r in recs]
    a = audit_sample(pool, n=10, seed=0)
    b = audit_sample(pool, n=10, seed=0)
    assert [x["key"] for x in a] == [x["key"] for x in b]
    assert all(x["manual_label"] is None for x in a)
