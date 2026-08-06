"""Prefix-conditional scoring loop (spec v2 Part C §3).

Per trajectory the client is reset once and the prefix grows by exactly one step
per assessment, so cost is ``O(T)`` prefix tokens plus ``T`` short readouts. The
loop asserts the shared-prefix path is live rather than assuming it.

Three evidence policies, which are what E5 ablates:

* ``typed`` (default) — prefix 0..t, plus a pointer to the assigned subtask and
  earlier same-turn peers when an ``execute`` step is near-empty.
* ``plain`` — prefix 0..t and nothing else.
* ``hindsight`` — the whole trajectory as the shared prefix for every step. Not
  a method: the ceiling figure. It is the one context-swap kept from paper 1's
  hindsight harness.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..record import Record, Step
from ..typing.normalize import NEAR_EMPTY_CHARS
from .client import JudgeClient
from .prompts import preamble, readout, render_step

POLICIES = ("typed", "plain", "hindsight")
MAX_POINTER_CHARS = 800

_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*%?")


@dataclass(slots=True)
class StepScore:
    """One assessment, keyed back to ``(file_id, step_idx)``."""

    subset: str
    file_id: str
    step_idx: int
    agent: str
    type_norm: str
    type_source: str
    p_raw: float
    p_cal: float | None = None
    augmented: bool = False
    judge: str = ""
    readout: str = "ptrue"
    policy: str = "typed"
    with_gt: bool = False
    use_types: bool = True
    prefix_tokens: int = 0
    readout_tokens: int = 0
    seconds: float = 0.0
    raw_text: str = ""

    @property
    def key(self) -> str:
        return f"{self.subset}/{self.file_id}"

    @property
    def p(self) -> float:
        return self.p_cal if self.p_cal is not None else self.p_raw

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StepScore":
        return cls(**{k: v for k, v in d.items() if k in cls.__slots__})


@dataclass(slots=True)
class TrajectoryScores:
    key: str
    scores: list[StepScore] = field(default_factory=list)
    seconds: float = 0.0
    prefix_tokens: int = 0
    readout_tokens: int = 0


# --- evidence ---------------------------------------------------------------


def turn_blocks(steps: Sequence[Step]) -> list[int]:
    """Segment the trajectory into turn blocks, opening one at each coordination
    step. Gives a subset-independent notion of "same turn" without a native turn
    field."""
    blocks: list[int] = []
    current = 0
    prev_coord = True
    for s in steps:
        coord = s.type_norm in ("plan", "delegate")
        if coord and not prev_coord:
            current += 1
        blocks.append(current)
        prev_coord = coord
    return blocks


def is_near_empty(step: Step) -> bool:
    return len((step.content or "").strip()) < NEAR_EMPTY_CHARS


def pointers(record: Record, t: int, blocks: Sequence[int]) -> list[str]:
    """Assigned subtask + earlier same-turn peers for a terse ``execute`` step.

    Strictly backward-looking. Letting a later step in would make the score
    non-causal and inflate attribution accuracy for free.
    """
    steps = record.steps
    target = steps[t]
    out: list[str] = []
    pat = re.compile(rf"\b{re.escape(target.agent)}\b", re.IGNORECASE)
    for j in range(t - 1, -1, -1):
        s = steps[j]
        if s.type_norm not in ("plan", "delegate"):
            continue
        if pat.search(s.content) or pat.search(s.role_raw) or (j == t - 1 and s.type_norm == "delegate"):
            out.append(f"assigned subtask for {target.agent}:\n{s.content.strip()[:MAX_POINTER_CHARS]}")
            break
    peers = [
        s
        for j, s in enumerate(steps[:t])
        if blocks[j] == blocks[t] and s.agent != target.agent and s.content.strip()
    ]
    for peer in peers[-2:]:
        out.append(
            f"peer step {peer.idx} ({peer.agent}, {peer.type_norm}):\n"
            f"{peer.content.strip()[:MAX_POINTER_CHARS]}"
        )
    return out


def untyped(steps: Sequence[Step]) -> tuple[Step, ...]:
    """Strip act types — the 'typing off' arm of E4."""
    return tuple(s.typed("unknown", s.type_source) for s in steps)


# --- scoring ----------------------------------------------------------------


def score_record(
    record: Record,
    client: JudgeClient,
    *,
    kind: str = "ptrue",
    policy: str = "typed",
    with_gt: bool = False,
    use_types: bool = True,
) -> TrajectoryScores:
    """Score every step of one trajectory against a shared, growing prefix."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; known: {POLICIES}")
    if not client.prefix_sharing:
        raise RuntimeError(
            f"{client.name} does not expose the shared-prefix path; HC reaches 130 "
            "steps and the quadratic path is not runnable (Part C §3)"
        )

    steps = record.steps if use_types else untyped(record.steps)
    blocks = turn_blocks(steps)
    head = preamble(record.query, record.ground_truth, with_gt=with_gt)

    out = TrajectoryScores(key=record.key)
    t0 = time.perf_counter()

    if policy == "hindsight":
        # One fixed prefix carrying the whole trajectory; each readout still
        # costs only its own tokens, so the ceiling stays O(T) to compute.
        client.reset(head + "".join(render_step(s) for s in steps))

    for t, step in enumerate(steps):
        if policy != "hindsight":
            if t == 0:
                client.reset(head)
            client.extend(render_step(step))

        augment = ""
        if policy == "typed" and step.type_norm == "execute" and is_near_empty(step):
            ptrs = pointers(record.with_steps(steps), t, blocks)
            if ptrs:
                augment = (
                    "\n[context for a terse step — within-trajectory only]\n"
                    + "\n\n".join(ptrs)
                    + "\n"
                )

        prompt = augment + readout(step, kind)
        if kind == "ptrue":
            p, trace = client.p_true(prompt)
            text = ""
        else:
            text, trace = client.generate(prompt, max_new_tokens=12)
            p = _parse_generated(text, kind)

        out.scores.append(
            StepScore(
                subset=record.subset,
                file_id=record.file_id,
                step_idx=step.idx,
                agent=step.agent,
                type_norm=step.type_norm,
                type_source=step.type_source,
                p_raw=p,
                augmented=bool(augment),
                judge=client.name,
                readout=kind,
                policy=policy,
                with_gt=with_gt,
                use_types=use_types,
                prefix_tokens=trace.prefix_tokens,
                readout_tokens=trace.readout_tokens,
                seconds=trace.seconds,
                raw_text=text[:80],
            )
        )
        out.readout_tokens += trace.readout_tokens

    out.seconds = time.perf_counter() - t0
    out.prefix_tokens = out.scores[-1].prefix_tokens if out.scores else 0
    return out


def _parse_generated(text: str, kind: str) -> float:
    """Turn a generated readout into a score.

    Unparseable generations become 0.5 and keep their raw text on the row —
    dropping them would quietly flatter the readouts that are worst at
    following the format, which is precisely what E2 is measuring.
    """
    t = (text or "").strip()
    if kind == "binary":
        low = t.lower()
        if low.startswith("true") or low.startswith("yes"):
            return 1.0
        if low.startswith("false") or low.startswith("no"):
            return 0.0
        return 0.5
    m = _NUM.search(t)
    if not m:
        return 0.5
    v = float(m.group(1))
    if v > 1.0:
        v /= 100.0
    return min(max(v, 0.0), 1.0)


def score_corpus(
    records: Sequence[Record],
    client: JudgeClient,
    *,
    kind: str = "ptrue",
    policy: str = "typed",
    with_gt: bool = False,
    use_types: bool = True,
    out_path: str | Path | None = None,
    progress: Callable[[int, int, TrajectoryScores], None] | None = None,
) -> list[TrajectoryScores]:
    results: list[TrajectoryScores] = []
    fh = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for i, rec in enumerate(records):
            ts = score_record(
                rec, client, kind=kind, policy=policy, with_gt=with_gt, use_types=use_types
            )
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
    results = list(results)
    if not results:
        return {"n_trajectories": 0}
    steps = [len(r.scores) for r in results]
    secs = [r.seconds for r in results]
    return {
        "n_trajectories": len(results),
        "n_assessments": sum(steps),
        "steps_min": min(steps),
        "steps_median": sorted(steps)[len(steps) // 2],
        "steps_max": max(steps),
        "seconds_total": round(sum(secs), 3),
        "seconds_per_trajectory_max": round(max(secs), 3),
        "seconds_per_assessment": round(sum(secs) / max(sum(steps), 1), 4),
        "prefix_tokens_max": max(r.prefix_tokens for r in results),
        "readout_tokens_total": sum(r.readout_tokens for r in results),
        "quadratic_prefix_tokens_avoided": sum(
            r.prefix_tokens * max(len(r.scores) - 1, 0) // 2 for r in results
        ),
    }


def load_scores(path: str | Path) -> list[StepScore]:
    with open(path, encoding="utf-8") as fh:
        return [StepScore.from_dict(json.loads(l)) for l in fh if l.strip()]


def by_file(scores: Sequence[StepScore]) -> dict[str, list[StepScore]]:
    out: dict[str, list[StepScore]] = {}
    for s in scores:
        out.setdefault(s.key, []).append(s)
    for v in out.values():
        v.sort(key=lambda s: s.step_idx)
    return out
