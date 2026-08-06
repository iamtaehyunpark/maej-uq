"""Prefix-conditional judge harness (spec §3).

Per trajectory the harness resets the scorer once, then walks the steps forward,
extending the shared prefix by exactly one step per assessment. Judging a
``T``-step trajectory therefore costs ``O(T)`` prefix tokens plus ``T`` short
readouts, not ``O(T²)``. Wall-clock and token counts are recorded per trajectory
for the cost paragraph the spec asks for.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..schema import Record
from . import prompts
from .backends import PrefixScorer, ScoreTrace
from .evidence import build_evidence, turn_blocks


@dataclass(slots=True)
class StepScore:
    """One assessment: ``p_t`` for a single step, with everything needed to audit it."""

    key: str
    dataset: str
    subset: str
    task_id: str
    run_id: int
    step_idx: int
    agent: str
    type_norm: str
    type_source: str
    p_raw: float
    p_cal: float | None = None
    augmented: bool = False
    n_pointers: int = 0
    prefix_tokens: int = 0
    readout_tokens: int = 0
    seconds: float = 0.0
    judge: str = ""
    readout: str = "ptrue"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StepScore":
        return cls(**{k: v for k, v in d.items() if k in cls.__slots__})


@dataclass(slots=True)
class TrajectoryScores:
    """All step scores for one record, plus its cost row."""

    record_key: str
    scores: list[StepScore] = field(default_factory=list)
    seconds: float = 0.0
    prefix_tokens: int = 0
    readout_tokens: int = 0

    @property
    def p(self) -> list[float]:
        return [s.p_raw for s in self.scores]

    def p_calibrated(self) -> list[float]:
        return [s.p_cal if s.p_cal is not None else s.p_raw for s in self.scores]


def judge_record(
    record: Record,
    scorer: PrefixScorer,
    *,
    readout: str = "ptrue",
    policy: str = "type_conditioned_v1",
    max_steps: int | None = None,
) -> TrajectoryScores:
    """Score every step of one trajectory against a shared, growing prefix."""
    readout_fn: Callable = prompts.READOUTS[readout]
    steps = record.steps if max_steps is None else record.steps[:max_steps]
    blocks = turn_blocks(record.steps)

    header = prompts.SYSTEM + "\n\n"
    header += f"[task]\n{record.query}\n" if record.query else "[task]\n(not available)\n"
    scorer.reset(header)

    out = TrajectoryScores(record_key=record.key)
    t_start = time.perf_counter()

    for t, step in enumerate(steps):
        rendered, _ = _render(step)
        scorer.extend(rendered)

        ev = build_evidence(record, t, policy=policy, blocks=blocks)
        augment = ""
        if ev.augmented:
            augment = (
                "\n[context for a terse step — within-trajectory only]\n"
                + "\n\n".join(ev.pointers)
                + "\n"
            )

        p, trace = scorer.p_true(augment + readout_fn(step))
        out.scores.append(
            StepScore(
                key=record.key,
                dataset=record.dataset,
                subset=record.subset,
                task_id=record.task_id,
                run_id=record.run_id,
                step_idx=step.idx,
                agent=step.agent,
                type_norm=step.type_norm,
                type_source=step.type_source,
                p_raw=p,
                augmented=ev.augmented,
                n_pointers=len(ev.pointers),
                prefix_tokens=trace.prefix_tokens,
                readout_tokens=trace.readout_tokens,
                seconds=trace.seconds,
                judge=scorer.name,
                readout=readout,
            )
        )
        out.readout_tokens += trace.readout_tokens

    out.seconds = time.perf_counter() - t_start
    out.prefix_tokens = out.scores[-1].prefix_tokens if out.scores else 0
    return out


def _render(step):
    from .evidence import render_step

    return render_step(step)


def judge_corpus(
    records: Sequence[Record],
    scorer: PrefixScorer,
    *,
    readout: str = "ptrue",
    policy: str = "type_conditioned_v1",
    out_path: str | Path | None = None,
    progress: Callable[[int, int, TrajectoryScores], None] | None = None,
) -> list[TrajectoryScores]:
    """Judge many trajectories, optionally streaming results to JSONL as they land."""
    results: list[TrajectoryScores] = []
    fh = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for i, rec in enumerate(records):
            ts = judge_record(rec, scorer, readout=readout, policy=policy)
            results.append(ts)
            if fh:
                for s in ts.scores:
                    fh.write(json.dumps(s.to_dict()) + "\n")
                fh.flush()
            if progress:
                progress(i + 1, len(records), ts)
    finally:
        if fh:
            fh.close()
    return results


def cost_summary(results: Iterable[TrajectoryScores]) -> dict[str, Any]:
    """The cost paragraph, as numbers (spec §3)."""
    results = list(results)
    if not results:
        return {"n_trajectories": 0}
    steps = [len(r.scores) for r in results]
    secs = [r.seconds for r in results]
    return {
        "n_trajectories": len(results),
        "n_assessments": sum(steps),
        "steps_per_trajectory": {
            "min": min(steps),
            "median": sorted(steps)[len(steps) // 2],
            "max": max(steps),
        },
        "seconds_total": sum(secs),
        "seconds_per_trajectory_max": max(secs),
        "seconds_per_assessment": sum(secs) / max(sum(steps), 1),
        "prefix_tokens_max": max(r.prefix_tokens for r in results),
        "readout_tokens_total": sum(r.readout_tokens for r in results),
        "quadratic_tokens_avoided": sum(
            r.prefix_tokens * max(len(r.scores) - 1, 0) // 2 for r in results
        ),
    }


def load_scores(path: str | Path) -> list[StepScore]:
    out: list[StepScore] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(StepScore.from_dict(json.loads(line)))
    return out


def group_by_record(scores: Sequence[StepScore]) -> dict[str, list[StepScore]]:
    grouped: dict[str, list[StepScore]] = {}
    for s in scores:
        grouped.setdefault(s.key, []).append(s)
    for v in grouped.values():
        v.sort(key=lambda s: s.step_idx)
    return grouped
