"""Unified record (spec v2 Part C §1).

One frozen dataclass carries data through every stage. Stages consume and
return records, or plain arrays keyed by ``(file_id, step_idx)``. No pandas
before ``eval/``.

Frozen on purpose: a record is a loaded fact about a trajectory, and no later
stage has any business editing one. Scores live in separate arrays keyed back
to ``(file_id, idx)``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Iterator, Literal, Sequence

Subset = Literal["alg", "hc"]
TypeNorm = Literal["plan", "delegate", "execute", "final", "unknown"]
TypeSource = Literal["parsed", "classified"]

TYPE_NORMS: tuple[str, ...] = ("plan", "delegate", "execute", "final", "unknown")

#: The only pre-registered flag: annotation names an agent that does not act at
#: the annotated step (3 AG + 3 HC known files).
FLAG_AGENT_STEP_MISMATCH = "agent_step_mismatch"
KNOWN_FLAGS: tuple[str, ...] = (FLAG_AGENT_STEP_MISMATCH,)


class RecordError(ValueError):
    """A record violates the schema. Loaders raise; they never repair."""


@dataclass(frozen=True, slots=True)
class Step:
    idx: int
    agent: str
    role_raw: str
    content: str
    type_norm: TypeNorm = "unknown"
    type_source: TypeSource = "classified"

    def typed(self, type_norm: str, type_source: str) -> "Step":
        return replace(self, type_norm=type_norm, type_source=type_source)


@dataclass(frozen=True, slots=True)
class Record:
    subset: Subset
    file_id: str
    query: str
    ground_truth: str
    steps: tuple[Step, ...]
    label_mistake_agent: str
    label_mistake_step: int
    label_mistake_reason: str = ""
    flags: tuple[str, ...] = ()
    dataset: str = "whowhen"
    source_file: str = ""

    @property
    def key(self) -> str:
        return f"{self.subset}/{self.file_id}"

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def gold(self) -> tuple[str, int]:
        return self.label_mistake_agent, self.label_mistake_step

    def with_steps(self, steps: Sequence[Step]) -> "Record":
        return replace(self, steps=tuple(steps))

    def with_flag(self, flag: str) -> "Record":
        if flag not in KNOWN_FLAGS:
            raise RecordError(f"{self.key}: unregistered flag {flag!r}")
        return self if flag in self.flags else replace(self, flags=self.flags + (flag,))

    def validate(self) -> "Record":
        """Loader asserts of Part C §1. Every one is a hard failure."""
        where = self.key
        if self.subset not in ("alg", "hc"):
            raise RecordError(f"{where}: bad subset {self.subset!r}")
        if not self.steps:
            raise RecordError(f"{where}: empty trajectory")
        for i, s in enumerate(self.steps):
            if s.idx != i:
                raise RecordError(f"{where}: step idx {s.idx} at position {i}")
            if not s.content.strip():
                raise RecordError(f"{where}: step {i} has empty content")
            if not s.agent:
                raise RecordError(f"{where}: step {i} has empty agent")
            if s.type_norm not in TYPE_NORMS:
                raise RecordError(f"{where}: step {i} bad type_norm {s.type_norm!r}")
            if s.type_source not in ("parsed", "classified"):
                raise RecordError(f"{where}: step {i} bad type_source {s.type_source!r}")
        if not 0 <= self.label_mistake_step < len(self.steps):
            raise RecordError(
                f"{where}: mistake_step {self.label_mistake_step} outside "
                f"0..{len(self.steps) - 1}"
            )
        if not self.label_mistake_agent:
            raise RecordError(f"{where}: empty mistake_agent")
        for f in self.flags:
            if f not in KNOWN_FLAGS:
                raise RecordError(f"{where}: unregistered flag {f!r}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Record":
        return cls(
            subset=d["subset"],
            file_id=str(d["file_id"]),
            query=d.get("query", ""),
            ground_truth=d.get("ground_truth", ""),
            steps=tuple(
                Step(
                    idx=int(s["idx"]),
                    agent=s["agent"],
                    role_raw=s.get("role_raw", ""),
                    content=s.get("content", ""),
                    type_norm=s.get("type_norm", "unknown"),
                    type_source=s.get("type_source", "classified"),
                )
                for s in d["steps"]
            ),
            label_mistake_agent=d["label_mistake_agent"],
            label_mistake_step=int(d["label_mistake_step"]),
            label_mistake_reason=d.get("label_mistake_reason", ""),
            flags=tuple(d.get("flags", ())),
            source_file=d.get("source_file", ""),
        )


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
            if line.strip():
                yield Record.from_dict(json.loads(line))


def stats(records: Sequence[Record]) -> dict[str, Any]:
    from collections import Counter

    types: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    agents: Counter[str] = Counter()
    for r in records:
        for s in r.steps:
            types[s.type_norm] += 1
            sources[s.type_source] += 1
            agents[s.agent] += 1
    return {
        "n_files": len(records),
        "n_steps": sum(r.n_steps for r in records),
        "n_flagged": sum(1 for r in records if FLAG_AGENT_STEP_MISMATCH in r.flags),
        "steps_max": max((r.n_steps for r in records), default=0),
        "type_norm": dict(types),
        "type_source": dict(sources),
        "n_agents": len(agents),
        "top_agents": dict(agents.most_common(10)),
    }
