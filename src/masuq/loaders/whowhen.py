"""Who&When adapters — Algorithm-Generated (``alg``) and Hand-Crafted (``hc``).

Both subsets are directories of per-trajectory JSON files carrying a question,
a ground truth, a step history, and the failure annotation
(``mistake_agent``/``mistake_step``/``mistake_reason``). There is no per-run
correctness label: every Who&When trajectory failed by construction, which is
why this corpus drives the *attribution* track only (spec §5).

Two subset-specific things happen here:

* **AG** steps are ``{content, role, name}`` → ``agent := name``, types classified.
* **HC** steps are ``{content, role}`` with a *compound* role that encodes both
  agent and act — ``Orchestrator (thought)``, ``Orchestrator (-> WebSurfer)``,
  ``WebSurfer`` — so types are ``parsed``, not classified (spec §1).

The ``agent_step_mismatch`` flag is set wherever the annotated mistake agent
does not match the agent of the annotated step *after orchestrator collapse*.
The spec expects 3 such files in AG and 3 in HC; the counts are asserted by
:mod:`masuq.loaders.expectations` rather than trusted.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from ..schema import (
    FLAG_AGENT_STEP_MISMATCH,
    FLAG_EMPTY_TRAJECTORY,
    FLAG_MISTAKE_STEP_CAST_FAILED,
    FLAG_MISTAKE_STEP_OUT_OF_RANGE,
    Record,
    Step,
    TypeNorm,
)
from ..typing_.classifier import apply_classifier, is_answer_emission
from .base import LoaderError, LoadReport, as_text, cast_step_index, finish, read_json, require

SUBSET_AG = "alg"
SUBSET_HC = "hc"

_HISTORY_KEYS = ("history", "steps", "messages", "conversation", "trajectory")
_QUERY_KEYS = ("question", "query", "task", "problem")
_TRUTH_KEYS = ("ground_truth", "answer", "gt", "final_answer")

#: ``Orchestrator (thought)`` / ``Orchestrator (-> WebSurfer)`` / ``WebSurfer``
_COMPOUND_ROLE = re.compile(r"^\s*([^()]+?)\s*(?:\(\s*(.*?)\s*\))?\s*$")
_ARROW_QUALIFIER = re.compile(r"^->\s*(.+)$")

#: Qualifier → normalised type for HC compound roles.
_QUALIFIER_MAP: dict[str, TypeNorm] = {
    "thought": "plan",
    "thinking": "plan",
    "plan": "plan",
    "planning": "plan",
    "final answer": "final",
    "final": "final",
    "answer": "final",
    "termination condition met": "final",
}

#: Worker agents in the HC (Magentic-One style) traces.
_HC_WORKERS = {"websurfer", "assistant", "filesurfer", "coder", "computerterminal", "executor"}


def parse_hc_role(role_raw: str) -> tuple[str, TypeNorm, str | None]:
    """Parse an HC compound role into ``(agent, type_norm, delegate_target)``.

    ``Orchestrator (thought)``          → ("Orchestrator", "plan", None)
    ``Orchestrator (-> WebSurfer)``     → ("Orchestrator", "delegate", "WebSurfer")
    ``WebSurfer``                       → ("WebSurfer", "execute", None)
    """
    m = _COMPOUND_ROLE.match(role_raw or "")
    if not m:
        return (role_raw.strip() or "unknown", "unknown", None)
    head = (m.group(1) or "").strip()
    qual = (m.group(2) or "").strip()

    if qual:
        arrow = _ARROW_QUALIFIER.match(qual)
        if arrow:
            # Delegation: the acting agent stays the orchestrator (spec §1).
            return head or "Orchestrator", "delegate", arrow.group(1).strip()
        mapped = _QUALIFIER_MAP.get(qual.lower())
        if mapped:
            return head or "Orchestrator", mapped, None
        # Unrecognised qualifier: keep the agent, defer the type.
        return head or "unknown", "unknown", None

    if head.lower() in _HC_WORKERS:
        return head, "execute", None
    if head.lower().startswith("orchestr"):
        return head, "plan", None
    return head or "unknown", "unknown", None


def collapse_orchestrator(agent: str) -> str:
    """Normalise orchestrator naming for the mismatch check.

    Annotations name the orchestrator inconsistently across files
    (``Orchestrator``, ``orchestrator (thought)``, ``MagenticOneOrchestrator``);
    collapsing them prevents spurious ``agent_step_mismatch`` flags while still
    catching genuine agent-vs-step disagreements.
    """
    a = (agent or "").strip().lower()
    a = re.sub(r"\s*\(.*?\)\s*$", "", a)
    a = a.replace("-", "_").replace(" ", "_")
    if "orchestr" in a or a in {"manager", "chat_manager", "coordinator", "supervisor"}:
        return "orchestrator"
    return a


def _pick(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _history(d: dict[str, Any], where: str) -> list[Any]:
    hist = _pick(d, _HISTORY_KEYS)
    if hist is None:
        raise LoaderError(f"{where}: no history list; keys={sorted(d)[:12]}")
    if not isinstance(hist, list):
        raise LoaderError(f"{where}: history is {type(hist)}")
    return hist


def _load_file(path: Path, subset: str, report: LoadReport) -> Record:
    blob = read_json(path)
    if not isinstance(blob, dict):
        raise LoaderError(f"{path.name}: root must be an object, got {type(blob)}")
    where = f"{subset}/{path.stem}"

    steps: list[Step] = []
    for i, raw in enumerate(_history(blob, where)):
        if not isinstance(raw, dict):
            raise LoaderError(f"{where}: step {i} is {type(raw)}, expected object")
        role = as_text(require(raw, "role", f"{where} step {i}", "name"))
        content = as_text(require(raw, "content", f"{where} step {i}", "output", "message"))

        if subset == SUBSET_HC:
            agent, type_norm, target = parse_hc_role(role)
            type_source = "parsed" if type_norm != "unknown" else "classified"
            if target:
                report.bump("delegate_parsed")
            if type_norm == "unknown":
                report.bump("role_unparsed")
                report.warn(f"{where} step {i}: unparsed compound role {role!r}")
        else:
            name = raw.get("name")
            agent = as_text(name).strip() if name else role.strip()
            type_norm, type_source = "unknown", "classified"
            if not name:
                report.bump("missing_name_fallback_role")

        steps.append(
            Step(
                idx=i,
                agent=agent or f"agent_{i}",
                role_raw=role,
                content=content,
                type_raw=None,
                type_norm=type_norm,
                type_source=type_source,
            )
        )

    # HC: the last step counts as `final` when it actually emits an answer;
    # otherwise its parsed type stands (spec §1).
    if subset == SUBSET_HC and steps:
        last = steps[-1]
        if is_answer_emission(last.content):
            last.type_norm = "final"
            last.type_source = "parsed"
            report.bump("final_promoted")

    apply_classifier(steps)  # fills only the steps still marked `classified`

    mistake_agent = _pick(blob, ("mistake_agent", "agent"))
    mistake_step_raw = blob.get("mistake_step", blob.get("step"))
    mistake_step, ok = cast_step_index(mistake_step_raw, where, report)

    rec = Record(
        dataset="whowhen",
        subset=subset,
        task_id=path.stem,
        run_id=0,
        query=as_text(_pick(blob, _QUERY_KEYS)) or None,
        ground_truth=as_text(_pick(blob, _TRUTH_KEYS)) or None,
        steps=steps,
        label_correct=None,  # Who&When has no per-run correctness label
        label_mistake_agent=as_text(mistake_agent) or None if mistake_agent else None,
        label_mistake_step=mistake_step,
        label_mistake_reason=as_text(_pick(blob, ("mistake_reason", "reason"))) or None,
        source_file=str(path),
    )

    if not steps:
        rec.add_flag(FLAG_EMPTY_TRAJECTORY)
        report.bump("empty_trajectories")
    if not ok:
        rec.add_flag(FLAG_MISTAKE_STEP_CAST_FAILED)
        report.bump("mistake_step_cast_failed")
    if mistake_step is not None and steps and not (0 <= mistake_step < len(steps)):
        rec.add_flag(FLAG_MISTAKE_STEP_OUT_OF_RANGE)
        report.bump("mistake_step_out_of_range")
        report.warn(f"{where}: mistake_step {mistake_step} outside 0..{len(steps) - 1}")

    if _has_agent_step_mismatch(rec):
        rec.add_flag(FLAG_AGENT_STEP_MISMATCH)
        report.bump("agent_step_mismatch")

    return rec


def _has_agent_step_mismatch(rec: Record) -> bool:
    """Does the annotated agent disagree with the annotated step's agent?"""
    if rec.label_mistake_agent is None or rec.label_mistake_step is None:
        return False
    if not (0 <= rec.label_mistake_step < len(rec.steps)):
        return False  # already flagged out-of-range; not an agent disagreement
    annotated = collapse_orchestrator(rec.label_mistake_agent)
    actual = collapse_orchestrator(rec.steps[rec.label_mistake_step].agent)
    return annotated != actual


def _json_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files = sorted(p for p in root.rglob("*.json") if not p.name.startswith("."))
    if not files:
        raise LoaderError(f"no .json trajectory files under {root}")
    return files


def load_ag(path: str | Path) -> tuple[list[Record], LoadReport]:
    """Load the Algorithm-Generated subset from a directory of JSON files."""
    return _load_dir(Path(path), SUBSET_AG)


def load_hc(path: str | Path) -> tuple[list[Record], LoadReport]:
    """Load the Hand-Crafted subset from a directory of JSON files."""
    return _load_dir(Path(path), SUBSET_HC)


def _load_dir(root: Path, subset: str) -> tuple[list[Record], LoadReport]:
    report = LoadReport(subset=subset, source=str(root))
    records = [_load_file(p, subset, report) for p in _json_files(root)]
    return records, finish(records, report)


def flagged(records: Iterable[Record]) -> list[Record]:
    return [r for r in records if FLAG_AGENT_STEP_MISMATCH in r.flags]
