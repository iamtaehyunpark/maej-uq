"""MATU per-run correctness labels from ``accuracy_dict_*.pkl`` (spec §1).

The join key is ``task_id × run``. The spec is explicit that alignment must be
*asserted, not assumed*: a silent key mismatch would corrupt every downstream
number (calibration is fit on these labels, and the trajectory-track AUROC is
computed against them). So :func:`join_labels` hard-fails on any key it cannot
resolve unless the caller explicitly opts into a lenient audit pass.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..schema import Record
from .base import LoaderError


class LabelJoinError(LoaderError):
    """Raised when the accuracy dict cannot be aligned with the loaded records."""


@dataclass(slots=True)
class JoinReport:
    source: str
    n_records: int = 0
    n_joined: int = 0
    n_true: int = 0
    n_label_keys: int = 0
    n_unused_label_keys: int = 0
    missing: list[str] = field(default_factory=list)
    unused: list[str] = field(default_factory=list)
    key_layout: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "key_layout": self.key_layout,
            "n_records": self.n_records,
            "n_joined": self.n_joined,
            "n_true": self.n_true,
            "accuracy": (self.n_true / self.n_joined) if self.n_joined else None,
            "n_label_keys": self.n_label_keys,
            "n_unused_label_keys": self.n_unused_label_keys,
            "missing_sample": self.missing[:20],
            "unused_sample": self.unused[:20],
        }


def load_accuracy_dict(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise LabelJoinError(f"missing accuracy dict: {p}")
    with p.open("rb") as fh:
        return pickle.load(fh)


def _as_bool(value: Any, where: str) -> bool:
    """Coerce a label cell to bool without inventing a truth value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        raise LabelJoinError(f"{where}: non-binary numeric label {value!r}")
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "correct", "yes"}:
            return True
        if v in {"false", "0", "incorrect", "no"}:
            return False
    raise LabelJoinError(f"{where}: uninterpretable label {value!r} ({type(value).__name__})")


def flatten_accuracy_dict(blob: Any) -> tuple[dict[tuple[str, int], bool], str]:
    """Normalise the pickle into ``{(task_id, run_id): correct}`` plus a layout note.

    Three layouts are accepted, all seen in MATU dumps:

    * ``{task_id: [bool, ...]}``        — run index is the list position
    * ``{task_id: {run: bool}}``        — run index is the inner key
    * ``{(task_id, run): bool}``        — already flat
    """
    if not isinstance(blob, Mapping):
        raise LabelJoinError(f"accuracy dict must be a mapping, got {type(blob)}")

    flat: dict[tuple[str, int], bool] = {}
    layouts: set[str] = set()

    for k, v in blob.items():
        if isinstance(k, tuple) and len(k) == 2:
            flat[(str(k[0]), int(k[1]))] = _as_bool(v, f"label[{k}]")
            layouts.add("flat_tuple_key")
        elif isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            for i, cell in enumerate(v):
                flat[(str(k), i)] = _as_bool(cell, f"label[{k}][{i}]")
            layouts.add("task_to_list")
        elif isinstance(v, Mapping):
            for rk, cell in v.items():
                flat[(str(k), int(str(rk)))] = _as_bool(cell, f"label[{k}][{rk}]")
            layouts.add("task_to_runmap")
        else:
            flat[(str(k), 0)] = _as_bool(v, f"label[{k}]")
            layouts.add("task_to_scalar")

    if len(layouts) > 1:
        raise LabelJoinError(f"mixed accuracy-dict layouts: {sorted(layouts)}")
    return flat, next(iter(layouts), "empty")


def join_labels(
    records: Sequence[Record],
    accuracy_path: str | Path,
    *,
    strict: bool = True,
) -> JoinReport:
    """Attach ``label_correct`` to every record, in place.

    With ``strict=True`` (the default, and what the pilot runs) any record
    without a label, or any label key with no record, is a hard failure. Set
    ``strict=False`` only for an exploratory audit of a new dump.
    """
    blob = load_accuracy_dict(accuracy_path)
    flat, layout = flatten_accuracy_dict(blob)

    report = JoinReport(
        source=str(accuracy_path),
        n_records=len(records),
        n_label_keys=len(flat),
        key_layout=layout,
    )
    seen: set[tuple[str, int]] = set()

    for rec in records:
        key = (rec.task_id, rec.run_id)
        if key not in flat:
            report.missing.append(rec.key)
            continue
        rec.label_correct = flat[key]
        report.n_joined += 1
        report.n_true += int(flat[key])
        seen.add(key)

    unused = set(flat) - seen
    report.n_unused_label_keys = len(unused)
    report.unused = [f"{t}#{r}" for t, r in sorted(unused, key=lambda x: (str(x[0]), x[1]))]

    if strict and (report.missing or unused):
        raise LabelJoinError(
            f"accuracy-dict alignment failed for {accuracy_path} (layout={layout}): "
            f"{len(report.missing)} records without a label "
            f"(e.g. {report.missing[:5]}), "
            f"{len(unused)} label keys without a record (e.g. {report.unused[:5]}). "
            "Spec §1 requires task_id×run alignment to be asserted, not assumed."
        )
    return report


def label_vector(records: Iterable[Record]) -> list[bool]:
    out = []
    for r in records:
        if r.label_correct is None:
            raise LabelJoinError(f"{r.key}: unlabelled record reached label_vector()")
        out.append(r.label_correct)
    return out
