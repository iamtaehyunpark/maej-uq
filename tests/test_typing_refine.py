"""Hierarchical typing: the coarse/fine split, its gate, and the disjointness rule."""

from __future__ import annotations

import pytest

from masattr.models import DisjointnessError, check_disjoint, family_of
from masattr.typing.refine import (
    SPLIT_GATE,
    MockSplitter,
    build_splitter,
    coarse_of,
    refine_records,
    validate_splitter,
)


class _AlwaysPlan(MockSplitter):
    name = "always_plan"

    def split(self, record, idx):
        from masattr.typing.refine import SplitVerdict

        return SplitVerdict("plan")


class _Oracle(MockSplitter):
    name = "oracle"

    def split(self, record, idx):
        from masattr.typing.refine import SplitVerdict

        return SplitVerdict(record.steps[idx].type_norm)


def test_coarse_collapses_only_coordination():
    assert coarse_of("plan") == coarse_of("delegate") == "coordination"
    assert coarse_of("execute") == "execute" and coarse_of("final") == "final"


def test_gate_rejects_the_majority_class_splitter(records):
    rep = validate_splitter(records["hc"], _AlwaysPlan())
    # Always guessing the majority class can clear a raw accuracy bar while
    # having learned nothing, so the gate requires beating that baseline too.
    assert rep.agreement == pytest.approx(rep.majority_baseline)
    assert not rep.passes


def test_gate_accepts_a_splitter_that_actually_splits(records):
    rep = validate_splitter(records["hc"], _Oracle())
    assert rep.agreement == 1.0
    assert rep.agreement > rep.majority_baseline
    assert rep.passes
    assert rep.coarse_agreement == 1.0
    assert "plan vs delegate" in rep.render()


def test_report_carries_the_rules_baseline_on_the_same_steps(records):
    rep = validate_splitter(records["hc"], _Oracle())
    assert 0.0 <= rep.rules_agreement <= 1.0
    assert rep.to_dict()["gate"] == SPLIT_GATE


def test_refine_touches_only_classified_coordination(records):
    hc = records["hc"]  # every coordination step here is parsed
    refined, n = refine_records(hc, _AlwaysPlan())
    assert n == 0
    assert [s.type_norm for s in refined[0].steps] == [s.type_norm for s in hc[0].steps]

    alg = records["alg"]  # entirely classified
    refined, n = refine_records(alg, _AlwaysPlan())
    assert n > 0
    assert all(
        s.type_norm == "plan"
        for r in refined
        for s in r.steps
        if s.type_source == "classified" and s.type_norm in ("plan", "delegate")
    )


# --- Part C §Validity: family disjointness ----------------------------------


@pytest.mark.parametrize(
    "model,family",
    [
        ("hf:Qwen/Qwen3.6-35B-A3B", "qwen"),
        ("openai:gpt-4o", "gpt"),
        ("hf:meta-llama/Llama-3.3-70B", "llama"),
        ("hf:mistralai/Mistral-7B", "mistral"),
        ("something-unheard-of", "unknown"),
    ],
)
def test_family_of(model, family):
    assert family_of(model) == family


def test_same_family_is_rejected():
    with pytest.raises(DisjointnessError, match="both family"):
        check_disjoint("type-classifier", "hf:Qwen/Qwen3-8B", "judge", "hf:Qwen3.6-35B-A3B")


def test_unknown_family_never_passes_silently():
    msg = check_disjoint("judge", "mystery-model", "labeling_judge", "openai:gpt-4o", strict=False)
    assert msg and "cannot verify" in msg


def test_build_splitter_enforces_disjointness_from_the_judge():
    with pytest.raises(DisjointnessError):
        build_splitter("hf:Qwen/Qwen3-8B", judge_model="hf:Qwen3.6-35B-A3B")


def test_build_splitter_lets_a_disjoint_family_through():
    # The check happens before the model loads, so a cross-family splitter gets
    # as far as needing torch — anything but a DisjointnessError means it passed.
    with pytest.raises(Exception) as exc:
        build_splitter("hf:meta-llama/Llama-3.3-70B", judge_model="hf:Qwen3.6-35B-A3B")
    assert not isinstance(exc.value, DisjointnessError)


def test_manifest_records_families_and_flags_violations():
    from masattr.manifest import Manifest

    m = Manifest(experiment="t")
    m.record_models(type_classifier="hf:Qwen/Qwen3-8B", judge="hf:Qwen3.6-35B-A3B")
    assert m.model_families["judge"] == "qwen"
    assert any("VALIDITY" in n for n in m.notes)
