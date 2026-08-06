"""Shared loader plumbing: strict key access, content coercion, load reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..schema import Record, SchemaError


class LoaderError(SchemaError):
    """Raised when source data does not match the documented adapter schema."""


@dataclass(slots=True)
class LoadReport:
    """What a loader saw. Emitted alongside records so counts are auditable."""

    subset: str
    source: str
    n_records: int = 0
    n_steps: int = 0
    warnings: list[str] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def bump(self, key: str, by: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + by

    def to_dict(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "source": self.source,
            "n_records": self.n_records,
            "n_steps": self.n_steps,
            "counters": dict(self.counters),
            "n_warnings": len(self.warnings),
            "warnings": self.warnings[:50],
        }


def read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise LoaderError(f"missing source file: {p}")
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def require(d: dict, key: str, where: str, *aliases: str) -> Any:
    """Fetch ``key`` (or the first present alias) or hard-fail with context."""
    if key in d:
        return d[key]
    for a in aliases:
        if a in d:
            return d[a]
    raise LoaderError(f"{where}: missing key {key!r} (aliases {aliases}); saw {sorted(d)[:12]}")


def as_text(value: Any) -> str:
    """Coerce a step payload to text without losing structure.

    Message contents in these corpora are sometimes strings, sometimes lists of
    content blocks, sometimes dicts. Serialising rather than dropping keeps the
    judge's evidence faithful to the log.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(as_text(item))
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        for k in ("text", "content", "output", "message"):
            if k in value and isinstance(value[k], str):
                return value[k]
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def cast_step_index(value: Any, where: str, report: LoadReport) -> tuple[int | None, bool]:
    """Cast an annotation's ``mistake_step`` to int. Returns ``(value, ok)``.

    Who&When stores this as a string in most files and as an int in a few; a
    handful are unparseable. Cast failures are logged and flagged, never guessed.
    """
    if value is None:
        return None, True
    if isinstance(value, bool):
        report.warn(f"{where}: mistake_step is a bool ({value!r})")
        return None, False
    if isinstance(value, int):
        return value, True
    text = str(value).strip()
    try:
        return int(text), True
    except ValueError:
        report.warn(f"{where}: uncastable mistake_step {value!r}")
        return None, False


def finish(records: Sequence[Record], report: LoadReport) -> LoadReport:
    report.n_records = len(records)
    report.n_steps = sum(len(r.steps) for r in records)
    for r in records:
        r.validate()
    return report
