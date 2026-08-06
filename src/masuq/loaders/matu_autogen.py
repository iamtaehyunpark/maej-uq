"""MATU-AutoGen adapter — ``conversation_logs_MMLU_Autogen_qwen2.5.json`` (spec §1).

Steps carry ``{role, agent, turn, type, output}``. The ``type`` field is native,
so ``type_source="native"`` and this subset is the reference against which the
rule classifier is validated (spec §2) and on which calibration is fit once and
frozen (spec §4).

``query`` is null here: the log is keyed by task id only. MMLU questions can be
recovered by id later if a policy needs them; v1 does not depend on it.

Note (spec §1): StarAgent ``plan`` outputs embed delegation payloads
(``analyst_task``/``verifier_task`` inside a JSON string). v1 treats the whole
step as ``plan``; splitting plan-from-delegate is deferred.
"""

from __future__ import annotations

from pathlib import Path

from ..schema import FLAG_EMPTY_TRAJECTORY, Record, Step, TypeNorm
from .base import LoadReport, as_text, read_json, require, finish
from .matu_common import iter_task_runs

SUBSET = "autogen_mmlu"

#: Native ``type`` → normalised type (spec §1).
NATIVE_TYPE_MAP: dict[str, TypeNorm] = {
    "plan": "plan",
    "solve": "execute",
    "final": "final",
    # Observed variants kept explicit rather than fuzzy-matched.
    "execute": "execute",
    "answer": "final",
}


def map_native_type(raw: str | None, report: LoadReport, where: str) -> tuple[TypeNorm, str]:
    """Map a native type string; unknown values become ``unknown`` and are counted.

    Returns ``(type_norm, type_source)`` — an unmapped value degrades to
    ``classified`` so the classifier can fill it in rather than silently
    asserting a native type we do not actually understand.
    """
    if raw is None:
        report.bump("type_missing")
        return "unknown", "classified"
    key = str(raw).strip().lower()
    if key in NATIVE_TYPE_MAP:
        report.bump(f"type_{key}")
        return NATIVE_TYPE_MAP[key], "native"
    report.bump("type_unmapped")
    report.warn(f"{where}: unmapped native type {raw!r}")
    return "unknown", "classified"


def load(path: str | Path, *, classify_unmapped: bool = True) -> tuple[list[Record], LoadReport]:
    report = LoadReport(subset=SUBSET, source=str(path))
    blob = read_json(path)
    records: list[Record] = []

    for task_id, run_id, raw_steps, meta in iter_task_runs(blob, report):
        where = f"autogen/{task_id}#{run_id}"
        steps: list[Step] = []
        for i, raw in enumerate(raw_steps):
            if not isinstance(raw, dict):
                raise TypeError(f"{where}: step {i} is {type(raw)}, expected object")
            role = as_text(require(raw, "role", f"{where} step {i}", "name"))
            agent = as_text(raw.get("agent") or raw.get("name") or role).strip()
            content = as_text(require(raw, "output", f"{where} step {i}", "content", "message"))
            type_raw = raw.get("type")
            type_raw = None if type_raw is None else str(type_raw)
            type_norm, type_source = map_native_type(type_raw, report, f"{where} step {i}")
            steps.append(
                Step(
                    idx=i,
                    agent=agent or f"agent_{i}",
                    role_raw=role,
                    content=content,
                    type_raw=type_raw,
                    type_norm=type_norm,
                    type_source=type_source,
                )
            )

        rec = Record(
            dataset="matu",
            subset=SUBSET,
            task_id=task_id,
            run_id=run_id,
            query=None,  # spec §0: task-id key only for this subset
            ground_truth=None,
            steps=steps,
            source_file=str(path),
        )
        if not steps:
            rec.add_flag(FLAG_EMPTY_TRAJECTORY)
            report.bump("empty_trajectories")
        if classify_unmapped and any(s.type_source == "classified" for s in rec.steps):
            from ..typing_.classifier import apply_classifier

            apply_classifier(rec.steps)  # only touches type_source == "classified"
        records.append(rec)

    return records, finish(records, report)
