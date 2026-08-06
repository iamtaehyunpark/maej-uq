"""Type-conditioned evidence policy v1 (spec §3).

Default policy: for step ``t``, evidence is the query (when available) plus the
ordered steps ``0..t``, each rendered with its agent identity and normalised
type.

The one type-conditioned exception in v1: an ``execute`` step whose content is
near-empty (a bare ``"B"``) carries no assessable evidence on its own, so the
prefix is augmented with an explicit pointer to the subtask that agent was
assigned and to any peer step in the same turn block. This is *within-trajectory*
corroboration only — it never reaches across runs, because doing so would leak
the very multi-run signal the trajectory track is supposed to measure.

Richer policies are deferred to the ablation (spec §3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..schema import Record, Step
from ..typing_.classifier import NEAR_EMPTY_CHARS

#: Cap on a single step's rendered content. Long tool dumps otherwise dominate
#: the prefix budget on HC trajectories; truncation is marked, never silent.
MAX_STEP_CHARS = 4000

#: Cap on the augmentation block so the rescue cannot outweigh the trajectory.
MAX_POINTER_CHARS = 800


@dataclass(slots=True)
class Evidence:
    """The rendered prefix for one assessment, plus what went into it."""

    text: str
    step_idx: int
    augmented: bool = False
    pointers: list[str] = field(default_factory=list)
    truncated_steps: list[int] = field(default_factory=list)


def render_step(step: Step, *, max_chars: int = MAX_STEP_CHARS) -> tuple[str, bool]:
    content = step.content or ""
    truncated = len(content) > max_chars
    if truncated:
        head = content[: max_chars // 2]
        tail = content[-max_chars // 2 :]
        content = f"{head}\n…[{len(step.content) - max_chars} chars elided]…\n{tail}"
    return (
        f"[step {step.idx} | agent={step.agent} | type={step.type_norm}]\n{content}\n",
        truncated,
    )


def turn_blocks(steps: Sequence[Step]) -> list[int]:
    """Assign each step a turn-block id.

    A new block opens at every ``plan``/``delegate`` step: in all four corpora a
    coordinator act is what starts a round of worker activity. This gives a
    subset-independent notion of "peer step at the same turn" without requiring
    a native ``turn`` field (only MATU-AutoGen has one).
    """
    blocks: list[int] = []
    current = 0
    prev_was_coordination = True
    for s in steps:
        is_coord = s.type_norm in ("plan", "delegate")
        if is_coord and not prev_was_coordination:
            current += 1
        blocks.append(current)
        prev_was_coordination = is_coord
    return blocks


_ASSIGNMENT_PAT_CACHE: dict[str, re.Pattern[str]] = {}


def _agent_pattern(agent: str) -> re.Pattern[str]:
    pat = _ASSIGNMENT_PAT_CACHE.get(agent)
    if pat is None:
        pat = re.compile(rf"\b{re.escape(agent)}\b", re.IGNORECASE)
        _ASSIGNMENT_PAT_CACHE[agent] = pat
    return pat


def find_assigned_subtask(steps: Sequence[Step], t: int) -> str | None:
    """Most recent coordination step at or before ``t`` that names step ``t``'s agent."""
    agent = steps[t].agent
    pat = _agent_pattern(agent)
    for j in range(t - 1, -1, -1):
        s = steps[j]
        if s.type_norm not in ("plan", "delegate"):
            continue
        if pat.search(s.content) or pat.search(s.role_raw):
            snippet = s.content.strip()
            return snippet[:MAX_POINTER_CHARS]
        # A delegation immediately preceding the step is its assignment even if
        # the agent is not named verbatim in the body.
        if j == t - 1 and s.type_norm == "delegate":
            return s.content.strip()[:MAX_POINTER_CHARS]
    return None


def find_peer_steps(steps: Sequence[Step], t: int, blocks: Sequence[int]) -> list[Step]:
    """Steps sharing step ``t``'s turn block, excluding ``t`` and later steps.

    Later steps are excluded on purpose: the judge is prefix-conditional, and
    letting a peer from the future into the evidence would make the score
    non-causal and quietly inflate attribution accuracy.
    """
    b = blocks[t]
    return [s for j, s in enumerate(steps[:t]) if blocks[j] == b and s.agent != steps[t].agent]


def is_near_empty(step: Step) -> bool:
    return len((step.content or "").strip()) < NEAR_EMPTY_CHARS


def build_evidence(
    record: Record,
    t: int,
    *,
    policy: str = "type_conditioned_v1",
    blocks: Sequence[int] | None = None,
) -> Evidence:
    """Render the prefix-conditional evidence for assessing step ``t``."""
    steps = record.steps
    if not (0 <= t < len(steps)):
        raise IndexError(f"{record.key}: step {t} out of range 0..{len(steps) - 1}")

    parts: list[str] = []
    if record.query:
        parts.append(f"[task]\n{record.query}\n")
    truncated: list[int] = []
    for s in steps[: t + 1]:
        rendered, was_truncated = render_step(s)
        parts.append(rendered)
        if was_truncated:
            truncated.append(s.idx)

    ev = Evidence(text="", step_idx=t, truncated_steps=truncated)

    if policy == "prefix_only":
        ev.text = "\n".join(parts)
        return ev

    if policy != "type_conditioned_v1":
        raise ValueError(f"unknown evidence policy {policy!r}")

    target = steps[t]
    if target.type_norm == "execute" and is_near_empty(target):
        blocks = list(blocks) if blocks is not None else turn_blocks(steps)
        pointers: list[str] = []
        subtask = find_assigned_subtask(steps, t)
        if subtask:
            pointers.append(f"assigned subtask for {target.agent}:\n{subtask}")
        for peer in find_peer_steps(steps, t, blocks)[-2:]:
            body = (peer.content or "").strip()[:MAX_POINTER_CHARS]
            if body:
                pointers.append(f"peer step {peer.idx} ({peer.agent}, {peer.type_norm}):\n{body}")
        if pointers:
            parts.append(
                "[context for a terse step — within-trajectory only]\n" + "\n\n".join(pointers) + "\n"
            )
            ev.augmented = True
            ev.pointers = pointers

    ev.text = "\n".join(parts)
    return ev


def incremental_segments(record: Record) -> list[str]:
    """The trajectory rendered as append-only chunks, for KV prefix sharing.

    Segment 0 is the task header; segment ``i+1`` is step ``i``. Feeding these to
    :meth:`PrefixScorer.extend` in order reproduces the ``prefix_only`` evidence
    while touching each token once.
    """
    segs = [f"[task]\n{record.query}\n" if record.query else "[task]\n(not available)\n"]
    for s in record.steps:
        rendered, _ = render_step(s)
        segs.append(rendered)
    return segs
