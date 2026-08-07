"""Shared loader plumbing and the pre-registered corpus asserts (spec v3 Part C §1)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..record import FLAG_AGENT_STEP_MISMATCH, Record, RecordError
from ..typing.normalize import collapse_orchestrator

#: subset → (n_files, n_flagged). Asserted, not trusted.
EXPECTED: dict[str, tuple[int, int]] = {"alg": (126, 3), "hc": (58, 3)}
EXPECTED_TOTAL_STEPS = 4092

_HISTORY_KEYS = ("history", "steps", "messages", "conversation", "trajectory")
_QUERY_KEYS = ("question", "query", "task", "problem")
#: ``groundtruth`` (no underscore) is the Hand-Crafted spelling in the released
#: parquet; ``ground_truth`` is the Algorithm-Generated one. Both must be here or
#: the with-GT setting silently reads an empty reference on half the corpus.
_TRUTH_KEYS = ("ground_truth", "groundtruth", "answer", "gt", "final_answer")

#: Column carrying the trajectory identity in the released parquet.
_ID_KEYS = ("question_ID", "question_id", "id", "file_id")

#: What to do with the five released files that violate the per-step asserts.
#: ``fail`` — refuse to load (spec-literal; the corpus will not load at all).
#: ``flag`` — load, flag, and let the run dual-report them.
#: ``drop`` — exclude them, which also breaks the 126/58 count assert.
ANOMALY_POLICIES = ("fail", "flag", "drop")


class LoaderError(RecordError):
    """Source data does not match the documented schema."""


@dataclass(slots=True)
class LoadReport:
    subset: str
    source: str
    n_files: int = 0
    n_steps: int = 0
    counters: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def bump(self, k: str, by: int = 1) -> None:
        self.counters[k] = self.counters.get(k, 0) + by

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "source": self.source,
            "n_files": self.n_files,
            "n_steps": self.n_steps,
            "counters": dict(self.counters),
            "notes": self.notes[:50],
        }


def rows(path: Path) -> list[tuple[str, dict]]:
    """Yield ``(file_id, row)`` from either release format.

    Who&When ships as one parquet per subset (126 / 58 rows), which is what the
    loaders read in practice. A directory of per-trajectory JSON is also
    accepted, because that is the shape the upstream repo's own examples use and
    the shape the fixtures take.
    """
    if path.is_dir():
        files = sorted(p for p in path.rglob("*.json") if not p.name.startswith("."))
        if not files:
            raise LoaderError(f"no .json trajectory files under {path}")
        return [(p.stem, read_json(p)) for p in files]
    if path.suffix == ".parquet":
        return read_parquet(path)
    if path.suffix == ".json":
        return [(path.stem, read_json(path))]
    raise LoaderError(f"missing or unrecognised subset source: {path}")


def read_parquet(path: Path) -> list[tuple[str, dict]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - env dependent
        raise LoaderError(
            "reading the released Who&When parquet needs pyarrow: pip install pyarrow"
        ) from e
    table = pq.read_table(path).to_pylist()
    out = []
    for i, row in enumerate(table):
        ident = next((str(row[k]) for k in _ID_KEYS if row.get(k)), None)
        if ident is None:
            raise LoaderError(f"{path.name} row {i}: no id column (looked for {_ID_KEYS})")
        out.append((ident, row))
    ids = [i for i, _ in out]
    if len(set(ids)) != len(ids):
        raise LoaderError(f"{path.name}: duplicate question_ID values")
    return out


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        blob = json.load(fh)
    if not isinstance(blob, dict):
        raise LoaderError(f"{path.name}: root must be an object, got {type(blob).__name__}")
    return blob


def as_text(value: Any) -> str:
    """Coerce a payload to text without discarding structure."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [
            str(v["text"]) if isinstance(v, dict) and "text" in v else as_text(v) for v in value
        ]
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        for k in ("text", "content", "output", "message"):
            if isinstance(value.get(k), str):
                return value[k]
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def pick(d: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def history(d: dict, where: str) -> list[dict]:
    h = pick(d, _HISTORY_KEYS)
    if h is None:
        raise LoaderError(f"{where}: no history list; keys={sorted(d)[:12]}")
    if not isinstance(h, list):
        raise LoaderError(f"{where}: history is {type(h).__name__}")
    for i, item in enumerate(h):
        if not isinstance(item, dict):
            raise LoaderError(f"{where}: step {i} is {type(item).__name__}, expected object")
    return h


def require(d: dict, key: str, where: str, *aliases: str) -> Any:
    if key in d:
        return d[key]
    for a in aliases:
        if a in d:
            return d[a]
    raise LoaderError(f"{where}: missing key {key!r} (aliases {aliases}); saw {sorted(d)[:12]}")


def cast_step(value: Any, where: str) -> int:
    """``mistake_step`` string→int. Hard-fails; no flag class covers this.

    The annotated step *is* the label, so a file whose label cannot be read is
    not a datapoint — unlike the two record-level anomalies, which are flagged
    and dual-reported because the counts depend on them.
    """
    if value is None:
        raise LoaderError(f"{where}: mistake_step is null")
    if isinstance(value, bool):
        raise LoaderError(f"{where}: mistake_step is a bool ({value!r})")
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except ValueError as e:
        raise LoaderError(f"{where}: uncastable mistake_step {value!r}") from e


def flag_mismatch(rec: Record, report: LoadReport) -> Record:
    """Set ``agent_step_mismatch`` where the annotated agent is not the agent of
    the annotated step, after orchestrator-name collapse."""
    if not 0 <= rec.label_mistake_step < len(rec.steps):
        # Already flagged out-of-range: there is no step to disagree with, and
        # calling that an agent mismatch would double-count one annotation fault.
        return rec
    annotated = collapse_orchestrator(rec.label_mistake_agent)
    actual = collapse_orchestrator(rec.steps[rec.label_mistake_step].agent)
    if annotated == actual:
        return rec
    report.bump("agent_step_mismatch")
    report.note(
        f"{rec.file_id}: annotation names {rec.label_mistake_agent!r} but step "
        f"{rec.label_mistake_step} is {rec.steps[rec.label_mistake_step].agent!r}"
    )
    return rec.with_flag(FLAG_AGENT_STEP_MISMATCH)


def apply_anomaly_policy(rec: Record, policy: str, report: LoadReport) -> Record | None:
    """Resolve a record's schema anomalies under the chosen policy.

    Returns the validated record, or ``None`` when the policy is to drop it.
    """
    if policy not in ANOMALY_POLICIES:
        raise LoaderError(f"unknown anomaly policy {policy!r}; known: {ANOMALY_POLICIES}")
    anomalies = rec.anomalies()
    if not anomalies:
        return rec.validate()

    detail = "; ".join(m for _, m in anomalies)
    if policy == "fail":
        raise LoaderError(
            f"{rec.key}: {detail}.\n"
            "Spec v2 Part C §1 asserts this cannot happen, but the release contains "
            "5 such files (3 HC with mistake_step past the trajectory end, 2 AG with "
            "an empty-content step), while the same section asserts 126/58 files. "
            "Both cannot hold. Choose: --anomaly-policy flag (keep and dual-report) "
            "or --anomaly-policy drop (exclude, which breaks the count assert)."
        )
    for flag, message in anomalies:
        rec = rec.with_flag(flag)
        report.bump(flag)
        report.note(f"{rec.file_id}: {message}")
    if policy == "drop":
        report.bump("dropped")
        return None
    return rec.validate()


def finish(records: list[Record], report: LoadReport) -> LoadReport:
    report.n_files = len(records)
    report.n_steps = sum(r.n_steps for r in records)
    return report


def check_expectations(
    records: Sequence[Record], subset: str, *, strict: bool = True
) -> list[str]:
    """Assert the pre-registered counts for one subset."""
    if subset not in EXPECTED:
        return []
    n_files, n_flagged = EXPECTED[subset]
    problems = []
    if len(records) != n_files:
        problems.append(f"{subset}: expected {n_files} files, got {len(records)}")
    got_flagged = sum(1 for r in records if FLAG_AGENT_STEP_MISMATCH in r.flags)
    if got_flagged != n_flagged:
        problems.append(
            f"{subset}: expected {n_flagged} agent_step_mismatch files, got {got_flagged}"
        )
    if problems and strict:
        raise AssertionError("; ".join(problems))
    return problems


def check_total_steps(
    alg: Sequence[Record], hc: Sequence[Record], *, strict: bool = True
) -> list[str]:
    n = sum(r.n_steps for r in alg) + sum(r.n_steps for r in hc)
    if n == EXPECTED_TOTAL_STEPS:
        return []
    msg = f"whowhen: expected {EXPECTED_TOTAL_STEPS} total steps, got {n}"
    if strict:
        raise AssertionError(msg)
    return [msg]


def level_of(row: Mapping[str, Any]) -> tuple[str, str]:
    """``(level, scale)`` from a source row.

    The released JSON carries ``level`` on every AG file and on 30 of 58 HC
    files, but in two vocabularies: numeric 1/2/3 and verbal Medium/Hard. The
    scale is recorded so strata are formed within a scale — a stratum mixing
    "2" with "Medium" would be an artifact, not a difficulty.
    """
    raw = row.get("level")
    if raw in (None, ""):
        return "", "absent"
    text = str(raw).strip()
    return text, ("numeric" if text.isdigit() else "verbal")


def enrich_levels(records: Sequence[Record], directory: str | Path) -> list[Record]:
    """Attach levels from a directory of their per-trajectory JSON.

    The parquet release drops the column, so a run reading parquet has to pick
    it up from their JSON, joined on ``question_ID``.
    """
    from dataclasses import replace as _replace

    levels: dict[str, tuple[str, str]] = {}
    for p in sorted(Path(directory).glob("*.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        ident = row.get("question_ID")
        if ident:
            levels[str(ident)] = level_of(row)
    out = []
    for rec in records:
        lv = levels.get(rec.file_id)
        out.append(_replace(rec, level=lv[0], level_scale=lv[1]) if lv else rec)
    return out
