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

#: Pre-registered: the annotation names an agent that does not act at the
#: annotated step (3 AG + 3 HC known files, confirmed against the release).
FLAG_AGENT_STEP_MISMATCH = "agent_step_mismatch"

#: Discovered in the release, not anticipated by spec v2 Part C §1, which asserts
#: both "126/58 files" and "mistake_step within bounds / every step has content"
#: — the data violates the second for 5 files, so the two asserts cannot both
#: hold. These flags exist so the conflict is visible and dual-reportable rather
#: than resolved silently by dropping files. See ANOMALY_POLICIES.
FLAG_MISTAKE_STEP_OUT_OF_RANGE = "mistake_step_out_of_range"  # 3 HC files
FLAG_EMPTY_STEP_CONTENT = "empty_step_content"  # 2 AG files

KNOWN_FLAGS: tuple[str, ...] = (
    FLAG_AGENT_STEP_MISMATCH,
    FLAG_MISTAKE_STEP_OUT_OF_RANGE,
    FLAG_EMPTY_STEP_CONTENT,
)

#: Flags that mark a *schema* violation rather than an annotation disagreement.
#: A record carrying one is not a clean datapoint for step-level scoring.
ANOMALY_FLAGS: tuple[str, ...] = (
    FLAG_MISTAKE_STEP_OUT_OF_RANGE,
    FLAG_EMPTY_STEP_CONTENT,
)


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

    def anomalies(self) -> list[tuple[str, str]]:
        """Schema violations this record carries, as ``(flag, message)``.

        Separated from :meth:`validate` because the release contains five files
        that violate spec v2 Part C §1 while also being required by its own
        126/58 count assert. Surfacing them as flags lets the run decide, and
        report, what happened to them.
        """
        out: list[tuple[str, str]] = []
        empty = [s.idx for s in self.steps if not s.content.strip()]
        if empty:
            out.append(
                (FLAG_EMPTY_STEP_CONTENT, f"steps {empty} have empty content")
            )
        if not 0 <= self.label_mistake_step < len(self.steps):
            out.append(
                (
                    FLAG_MISTAKE_STEP_OUT_OF_RANGE,
                    f"mistake_step {self.label_mistake_step} outside "
                    f"0..{len(self.steps) - 1}",
                )
            )
        return out

    def validate(self) -> "Record":
        """Loader asserts of Part C §1.

        Hard failure for anything malformed. An *anomaly* raises too, unless the
        record already carries its flag — flagging is how a run says it saw the
        problem and chose to keep the file.
        """
        where = self.key
        if self.subset not in ("alg", "hc"):
            raise RecordError(f"{where}: bad subset {self.subset!r}")
        if not self.steps:
            raise RecordError(f"{where}: empty trajectory")
        for i, s in enumerate(self.steps):
            if s.idx != i:
                raise RecordError(f"{where}: step idx {s.idx} at position {i}")
            if not s.agent:
                raise RecordError(f"{where}: step {i} has empty agent")
            if s.type_norm not in TYPE_NORMS:
                raise RecordError(f"{where}: step {i} bad type_norm {s.type_norm!r}")
            if s.type_source not in ("parsed", "classified"):
                raise RecordError(f"{where}: step {i} bad type_source {s.type_source!r}")
        if not self.label_mistake_agent:
            raise RecordError(f"{where}: empty mistake_agent")
        for f in self.flags:
            if f not in KNOWN_FLAGS:
                raise RecordError(f"{where}: unregistered flag {f!r}")
        unflagged = [(f, m) for f, m in self.anomalies() if f not in self.flags]
        if unflagged:
            raise RecordError(
                f"{where}: " + "; ".join(m for _, m in unflagged)
            )
        return self

    @property
    def is_anomalous(self) -> bool:
        return any(f in ANOMALY_FLAGS for f in self.flags)

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
    flags: Counter[str] = Counter()
    for r in records:
        flags.update(r.flags)
    return {
        "n_files": len(records),
        "n_steps": sum(r.n_steps for r in records),
        "n_flagged": flags[FLAG_AGENT_STEP_MISMATCH],
        "flags": dict(flags),
        "steps_max": max((r.n_steps for r in records), default=0),
        "type_norm": dict(types),
        "type_source": dict(sources),
        "n_agents": len(agents),
        "top_agents": dict(agents.most_common(10)),
    }
