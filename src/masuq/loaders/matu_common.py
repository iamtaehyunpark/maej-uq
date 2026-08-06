"""Shared traversal for the two MATU conversation-log files.

MATU logs are nested ``task_id -> runs -> steps``. The middle level is a list in
some dumps and a dict keyed by run index in others, and the query may sit either
beside the runs or inside the first step. We normalise the nesting here so the
two adapters only have to describe their *step* schema.
"""

from __future__ import annotations

from typing import Any, Iterator

from .base import LoaderError, LoadReport


def iter_task_runs(
    blob: Any, report: LoadReport
) -> Iterator[tuple[str, int, list[Any], dict[str, Any]]]:
    """Yield ``(task_id, run_id, steps, meta)`` for every run in a MATU log.

    ``meta`` carries any run- or task-level fields found beside the step list
    (e.g. ``question``/``query``/``answer``), so adapters can pick up the query
    without re-walking the tree.
    """
    if not isinstance(blob, dict):
        raise LoaderError(f"MATU log root must be an object keyed by task_id, got {type(blob)}")

    for task_id, task_val in blob.items():
        task_meta: dict[str, Any] = {}
        runs_obj = task_val

        if isinstance(task_val, dict) and not _is_run_map(task_val):
            # Task-level object wrapping the runs plus metadata.
            runs_obj = None
            for key in ("runs", "conversations", "logs", "trajectories", "history"):
                if key in task_val:
                    runs_obj = task_val[key]
                    break
            task_meta = {k: v for k, v in task_val.items() if not isinstance(v, (list, dict))}
            if runs_obj is None:
                # Single unwrapped run stored directly under the task.
                raise LoaderError(
                    f"task {task_id!r}: cannot locate run list; keys={sorted(task_val)[:12]}"
                )

        for run_id, steps in _iter_runs(task_id, runs_obj, report):
            meta = dict(task_meta)
            if isinstance(steps, dict):
                meta.update({k: v for k, v in steps.items() if not isinstance(v, (list, dict))})
                inner = None
                for key in ("steps", "messages", "conversation", "history", "chat_history"):
                    if key in steps:
                        inner = steps[key]
                        break
                if inner is None:
                    raise LoaderError(
                        f"task {task_id!r} run {run_id}: no step list; keys={sorted(steps)[:12]}"
                    )
                steps = inner
            if not isinstance(steps, list):
                raise LoaderError(f"task {task_id!r} run {run_id}: steps is {type(steps)}")
            yield str(task_id), run_id, steps, meta


def _is_run_map(d: dict) -> bool:
    """A dict of runs is keyed by integer-like keys."""
    if not d:
        return False
    return all(str(k).lstrip("-").isdigit() for k in d.keys())


def _iter_runs(task_id: str, runs_obj: Any, report: LoadReport) -> Iterator[tuple[int, Any]]:
    if isinstance(runs_obj, list):
        if runs_obj and isinstance(runs_obj[0], dict) and _looks_like_step(runs_obj[0]):
            # A bare step list: one implicit run.
            report.bump("implicit_single_run")
            yield 0, runs_obj
            return
        for i, run in enumerate(runs_obj):
            yield i, run
    elif isinstance(runs_obj, dict):
        for k in sorted(runs_obj, key=lambda x: int(str(x))):
            yield int(str(k)), runs_obj[k]
    else:
        raise LoaderError(f"task {task_id!r}: runs is {type(runs_obj)}")


def _looks_like_step(d: dict) -> bool:
    keys = {k.lower() for k in d}
    return bool(keys & {"output", "content", "message"}) and bool(keys & {"role", "agent", "name"})
