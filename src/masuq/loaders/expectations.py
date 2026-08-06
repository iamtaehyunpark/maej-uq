"""Pre-registered corpus counts (spec §7).

These are the assertions the weekend build order is graded against. They live in
code so a loader regression fails loudly instead of quietly changing an N.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..schema import FLAG_AGENT_STEP_MISMATCH, Record


class ExpectationError(AssertionError):
    """A pre-registered count did not hold."""


@dataclass(frozen=True, slots=True)
class Expectation:
    subset: str
    n_records: int | None = None
    n_steps: int | None = None
    n_flagged_mismatch: int | None = None


#: Spec §7.1 — W&W: 126 AG files, 58 HC files, 4092 steps across the two.
#: Spec §7.2 — MATU: 400 tasks × 10 runs per cell, two cells.
EXPECTATIONS: dict[str, Expectation] = {
    "alg": Expectation("alg", n_records=126, n_flagged_mismatch=3),
    "hc": Expectation("hc", n_records=58, n_flagged_mismatch=3),
    "camel_math": Expectation("camel_math", n_records=4000),
    "autogen_mmlu": Expectation("autogen_mmlu", n_records=4000),
}

#: Combined step count across the two Who&When subsets.
WHOWHEN_TOTAL_STEPS = 4092

#: MATU cell shape.
MATU_N_TASKS = 400
MATU_N_RUNS = 10


def check_subset(records: Sequence[Record], subset: str, *, strict: bool = True) -> list[str]:
    """Check one subset against its expectation. Returns the list of violations."""
    exp = EXPECTATIONS.get(subset)
    problems: list[str] = []
    if exp is None:
        return problems

    if exp.n_records is not None and len(records) != exp.n_records:
        problems.append(f"{subset}: expected {exp.n_records} records, got {len(records)}")
    if exp.n_steps is not None:
        n = sum(len(r.steps) for r in records)
        if n != exp.n_steps:
            problems.append(f"{subset}: expected {exp.n_steps} steps, got {n}")
    if exp.n_flagged_mismatch is not None:
        n = sum(1 for r in records if FLAG_AGENT_STEP_MISMATCH in r.flags)
        if n != exp.n_flagged_mismatch:
            problems.append(
                f"{subset}: expected {exp.n_flagged_mismatch} agent_step_mismatch files, got {n}"
            )
    if problems and strict:
        raise ExpectationError("; ".join(problems))
    return problems


def check_whowhen_steps(ag: Sequence[Record], hc: Sequence[Record], *, strict: bool = True) -> list[str]:
    n = sum(len(r.steps) for r in ag) + sum(len(r.steps) for r in hc)
    if n == WHOWHEN_TOTAL_STEPS:
        return []
    msg = [f"whowhen: expected {WHOWHEN_TOTAL_STEPS} total steps, got {n}"]
    if strict:
        raise ExpectationError(msg[0])
    return msg


def check_matu_cell(records: Sequence[Record], subset: str, *, strict: bool = True) -> list[str]:
    """Assert the 400×10 grid: every task has exactly ``MATU_N_RUNS`` runs."""
    from collections import Counter

    per_task = Counter(r.task_id for r in records)
    problems: list[str] = []
    if len(per_task) != MATU_N_TASKS:
        problems.append(f"{subset}: expected {MATU_N_TASKS} tasks, got {len(per_task)}")
    bad = {t: c for t, c in per_task.items() if c != MATU_N_RUNS}
    if bad:
        sample = list(bad.items())[:5]
        problems.append(
            f"{subset}: {len(bad)} tasks do not have {MATU_N_RUNS} runs (e.g. {sample})"
        )
    dup = [r.key for r in records]
    if len(set(dup)) != len(dup):
        problems.append(f"{subset}: duplicate (task_id, run_id) keys present")
    if problems and strict:
        raise ExpectationError("; ".join(problems))
    return problems
