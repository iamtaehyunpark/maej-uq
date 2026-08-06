"""Unified record schema — the single target of all loaders (spec §0).

Every adapter in :mod:`masuq.loaders` produces :class:`Record` objects, and every
downstream stage (typing, judge, calibration, attribution) consumes only these.
Nothing below this module knows which corpus a record came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Iterator, Literal, Sequence

Dataset = Literal["matu", "whowhen"]
Subset = Literal["camel_math", "autogen_mmlu", "alg", "hc"]
TypeNorm = Literal["plan", "delegate", "execute", "final", "unknown"]
TypeSource = Literal["native", "parsed", "classified"]

TYPE_NORMS: tuple[str, ...] = ("plan", "delegate", "execute", "final", "unknown")
TYPE_SOURCES: tuple[str, ...] = ("native", "parsed", "classified")

#: Pre-registered exclusion flags. Records carrying a flag are still loaded; the
#: analysis layer reports with and without them (spec §6, §8).
FLAG_AGENT_STEP_MISMATCH = "agent_step_mismatch"
FLAG_MISTAKE_STEP_CAST_FAILED = "mistake_step_cast_failed"
FLAG_MISTAKE_STEP_OUT_OF_RANGE = "mistake_step_out_of_range"
FLAG_EMPTY_TRAJECTORY = "empty_trajectory"

KNOWN_FLAGS: tuple[str, ...] = (
    FLAG_AGENT_STEP_MISMATCH,
    FLAG_MISTAKE_STEP_CAST_FAILED,
    FLAG_MISTAKE_STEP_OUT_OF_RANGE,
    FLAG_EMPTY_TRAJECTORY,
)


class SchemaError(ValueError):
    """Raised when a record violates the unified schema.

    Loaders hard-fail rather than coerce: a silently-repaired record is a
    provenance hole, and the pilot's whole claim rests on provenance.
    """


@dataclass(slots=True)
class Step:
    """One step of a multi-agent trajectory."""

    idx: int
    agent: str
    role_raw: str
    content: str
    type_raw: str | None = None
    type_norm: TypeNorm = "unknown"
    type_source: TypeSource = "classified"

    def validate(self, where: str = "") -> None:
        if self.idx < 0:
            raise SchemaError(f"{where}: negative step idx {self.idx}")
        if not isinstance(self.agent, str) or not self.agent:
            raise SchemaError(f"{where}: step {self.idx} has empty agent")
        if self.type_norm not in TYPE_NORMS:
            raise SchemaError(f"{where}: step {self.idx} bad type_norm {self.type_norm!r}")
        if self.type_source not in TYPE_SOURCES:
            raise SchemaError(f"{where}: step {self.idx} bad type_source {self.type_source!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Step":
        return cls(
            idx=int(d["idx"]),
            agent=str(d["agent"]),
            role_raw=str(d.get("role_raw", "")),
            content=str(d.get("content", "")),
            type_raw=d.get("type_raw"),
            type_norm=d.get("type_norm", "unknown"),
            type_source=d.get("type_source", "classified"),
        )


@dataclass(slots=True)
class Record:
    """One trajectory: a task run by a multi-agent system, plus whatever labels exist.

    ``run_id`` is 0 for Who&When (single trajectory per file) and 0..N-1 for MATU
    (N repeated runs of the same task).
    """

    dataset: Dataset
    subset: Subset
    task_id: str
    run_id: int = 0
    query: str | None = None
    ground_truth: str | None = None
    steps: list[Step] = field(default_factory=list)
    label_correct: bool | None = None
    label_mistake_agent: str | None = None
    label_mistake_step: int | None = None
    label_mistake_reason: str | None = None
    flags: list[str] = field(default_factory=list)
    source_file: str | None = None

    # ---- identity ----------------------------------------------------------

    @property
    def key(self) -> str:
        """Stable unique id across the merged corpus."""
        return f"{self.dataset}/{self.subset}/{self.task_id}#{self.run_id}"

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    # ---- validation --------------------------------------------------------

    def validate(self) -> None:
        where = self.key
        if self.dataset not in ("matu", "whowhen"):
            raise SchemaError(f"{where}: bad dataset {self.dataset!r}")
        if self.subset not in ("camel_math", "autogen_mmlu", "alg", "hc"):
            raise SchemaError(f"{where}: bad subset {self.subset!r}")
        if not self.task_id:
            raise SchemaError(f"{where}: empty task_id")
        if self.run_id < 0:
            raise SchemaError(f"{where}: negative run_id")
        for i, s in enumerate(self.steps):
            if s.idx != i:
                raise SchemaError(f"{where}: step idx {s.idx} out of order at position {i}")
            s.validate(where)
        for fl in self.flags:
            if fl not in KNOWN_FLAGS:
                raise SchemaError(f"{where}: unregistered flag {fl!r}")
        if self.dataset == "whowhen":
            if self.label_correct is not None:
                raise SchemaError(f"{where}: whowhen carries no per-run correctness label")
        if self.label_mistake_step is not None and self.steps:
            if not (0 <= self.label_mistake_step < len(self.steps)):
                if FLAG_MISTAKE_STEP_OUT_OF_RANGE not in self.flags:
                    raise SchemaError(
                        f"{where}: mistake_step {self.label_mistake_step} outside "
                        f"0..{len(self.steps) - 1} and not flagged"
                    )

    def add_flag(self, flag: str) -> None:
        if flag not in KNOWN_FLAGS:
            raise SchemaError(f"{self.key}: unregistered flag {flag!r}")
        if flag not in self.flags:
            self.flags.append(flag)

    # ---- (de)serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Record":
        return cls(
            dataset=d["dataset"],
            subset=d["subset"],
            task_id=str(d["task_id"]),
            run_id=int(d.get("run_id", 0)),
            query=d.get("query"),
            ground_truth=d.get("ground_truth"),
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            label_correct=d.get("label_correct"),
            label_mistake_agent=d.get("label_mistake_agent"),
            label_mistake_step=d.get("label_mistake_step"),
            label_mistake_reason=d.get("label_mistake_reason"),
            flags=list(d.get("flags", [])),
            source_file=d.get("source_file"),
        )


# ---- corpus-level helpers --------------------------------------------------


def write_jsonl(records: Iterable[Record], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[Record]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Record.from_dict(json.loads(line))


def corpus_stats(records: Sequence[Record]) -> dict[str, Any]:
    """Summary used by the §7 build-order assertions and by ``masuq load --stats``."""
    from collections import Counter

    type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    agent_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    n_steps = 0
    for r in records:
        n_steps += len(r.steps)
        for s in r.steps:
            type_counts[s.type_norm] += 1
            source_counts[s.type_source] += 1
            agent_counts[s.agent] += 1
        for fl in r.flags:
            flag_counts[fl] += 1
    return {
        "n_records": len(records),
        "n_steps": n_steps,
        "n_labelled_correct": sum(1 for r in records if r.label_correct is not None),
        "n_labelled_mistake": sum(1 for r in records if r.label_mistake_step is not None),
        "type_norm": dict(type_counts),
        "type_source": dict(source_counts),
        "flags": dict(flag_counts),
        "n_agents": len(agent_counts),
        "top_agents": dict(agent_counts.most_common(12)),
    }
