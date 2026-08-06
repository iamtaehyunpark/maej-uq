"""Schema adapters: four source formats → one unified record type (spec §1)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..schema import Record
from .base import LoaderError, LoadReport
from .expectations import (
    EXPECTATIONS,
    ExpectationError,
    check_matu_cell,
    check_subset,
    check_whowhen_steps,
)
from .labels import JoinReport, LabelJoinError, join_labels
from . import matu_autogen, matu_camel, whowhen

#: subset name → loader taking a path and returning ``(records, report)``
LOADERS: dict[str, Callable[..., tuple[list[Record], LoadReport]]] = {
    "camel_math": matu_camel.load,
    "autogen_mmlu": matu_autogen.load,
    "alg": whowhen.load_ag,
    "hc": whowhen.load_hc,
}


def load_subset(subset: str, path: str | Path, **kw) -> tuple[list[Record], LoadReport]:
    if subset not in LOADERS:
        raise LoaderError(f"unknown subset {subset!r}; known: {sorted(LOADERS)}")
    return LOADERS[subset](path, **kw)


__all__ = [
    "EXPECTATIONS",
    "ExpectationError",
    "JoinReport",
    "LOADERS",
    "LabelJoinError",
    "LoadReport",
    "LoaderError",
    "check_matu_cell",
    "check_subset",
    "check_whowhen_steps",
    "join_labels",
    "load_subset",
    "matu_autogen",
    "matu_camel",
    "whowhen",
]
