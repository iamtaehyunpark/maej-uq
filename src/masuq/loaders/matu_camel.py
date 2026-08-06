"""MATU-CAMEL adapter — ``conversation_logs_Math_qwen2.5.json`` (spec §1).

Steps carry ``{role, output}`` only: there is no type field and none can be
parsed out of the role, so every step is ``type_source="classified"`` and the
rule classifier of :mod:`masuq.typing_.classifier` supplies ``type_norm``.
``agent := role``. The query is present in this corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schema import FLAG_EMPTY_TRAJECTORY, Record, Step
from ..typing_.classifier import apply_classifier
from .base import LoadReport, as_text, read_json, require, finish
from .matu_common import iter_task_runs

SUBSET = "camel_math"

_QUERY_KEYS = ("query", "question", "problem", "task", "prompt")
_TRUTH_KEYS = ("ground_truth", "answer", "gt", "solution", "label")


def _pick(meta: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def load(path: str | Path, *, classify: bool = True) -> tuple[list[Record], LoadReport]:
    report = LoadReport(subset=SUBSET, source=str(path))
    blob = read_json(path)
    records: list[Record] = []

    for task_id, run_id, raw_steps, meta in iter_task_runs(blob, report):
        where = f"camel/{task_id}#{run_id}"
        steps: list[Step] = []
        for i, raw in enumerate(raw_steps):
            if not isinstance(raw, dict):
                raise TypeError(f"{where}: step {i} is {type(raw)}, expected object")
            role = as_text(require(raw, "role", f"{where} step {i}", "name", "agent"))
            content = as_text(require(raw, "output", f"{where} step {i}", "content", "message"))
            steps.append(
                Step(
                    idx=i,
                    agent=role.strip() or f"agent_{i}",
                    role_raw=role,
                    content=content,
                    type_raw=None,
                    type_norm="unknown",
                    type_source="classified",
                )
            )

        rec = Record(
            dataset="matu",
            subset=SUBSET,
            task_id=task_id,
            run_id=run_id,
            query=_pick(meta, _QUERY_KEYS),
            ground_truth=_pick(meta, _TRUTH_KEYS),
            steps=steps,
            source_file=str(path),
        )
        if not steps:
            rec.add_flag(FLAG_EMPTY_TRAJECTORY)
            report.bump("empty_trajectories")
        if rec.query is None:
            report.bump("missing_query")
        if classify:
            apply_classifier(rec.steps)
        records.append(rec)

    if report.counters.get("missing_query"):
        report.warn(
            f"{report.counters['missing_query']} CAMEL runs lack a query field; "
            "spec §1 expects the query to be present — check the dump version"
        )
    return records, finish(records, report)
