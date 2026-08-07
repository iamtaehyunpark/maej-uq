"""Prefix-conditional scoring loop (spec v3 Part C §3).

Per trajectory the client is reset once and the prefix grows by exactly one step
per assessment, so cost is ``O(T)`` prefix tokens plus ``T`` short readouts. The
loop asserts the shared-prefix path is live rather than assuming it.

Evidence has two independent parts, and conflating them was a real bug once.

**Base assembly** (the ``policy``) governs how the prefix ``0..t`` is built:

* ``typed`` (default) — plus a pointer to the assigned subtask and earlier
  same-turn peers when an ``execute`` step is near-empty.
* ``plain`` — prefix and nothing else.
* ``hindsight`` — the whole trajectory as the shared prefix for every step. Not
  a method: the ceiling figure.

**Lookahead** (the ``lookahead`` arm) governs what, if anything, is appended
*after* step ``t``:

* ``none`` (**W0**) — nothing. Prefix-conditional.
* ``resp`` (**W+resp**) — the immediately following contiguous steps by *other*
  agents, capped at 2: the realized response to step ``t``.
* ``own`` (**W+own**) — W+resp plus the acting agent's own next appearance.
* ``deleg`` (**W+deleg**) — ``resp``, except that a ``delegate`` step gets a
  window extended to the delegated subtask's resolution: following steps until
  control returns to the delegator, capped at ``DELEG_CAP``.

``deleg`` is **smoke-motivated**, added after Stage-0 and before E1, and every
row it produces says so. The Stage-0 read showed delegation faults scoring
*worse* than chance while worker faults scored better, and W+resp made them
worse still — a cap-2 window mostly captures the assignee's compliant
acknowledgment, while the struggle that makes the delegation wrong surfaces
later. This arm tests that explanation. It is an E5 ablation row; the primary
arm is W0 and was locked by the pre-fixed rule before this existed.

W+resp exists because a delegation error's evidence is the assignee's downstream
struggle, which same-turn peers cannot see. The near-empty-execute rescue is
base assembly and stays on in every arm — it is not one of the arms.

**Note on asymmetry.** ``resp`` and ``own`` deliberately look ahead, so those
arms are *not* prefix-conditional. The lookahead is rendered into the readout
segment rather than the shared prefix — it differs per step, so it cannot be
cached — and every score row records which arm produced it.
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

#: Lookahead arms. ``none`` is prefix-conditional; the others are not.
LOOKAHEAD = ("none", "resp", "own", "deleg")

#: How many following steps by other agents count as "the realized response".
RESP_CAP = 2

#: Window for a delegation's resolution. Wider than ``RESP_CAP`` because a
#: delegation's wrongness is consequence-visible: the assignee acknowledges
#: first and struggles afterwards.
DELEG_CAP = 5
MAX_POINTER_CHARS = 800

#: Pre-registered prefix budget, in characters. Measured in characters rather
#: than tokens because this module is model-agnostic; at the usual ~4 chars per
#: token this sits comfortably under ``HFClient.max_prefix_tokens``, which
#: enforces the real cap. HC's 130-step logs reach ~38k estimated tokens, so a
#: truncation policy is pre-registered rather than left to context capacity.
PREFIX_BUDGET_CHARS = 90_000

#: How much of a demoted execute step survives, after its header line.
HEADER_CHARS = 120

#: Rebuilds target this fraction of the budget, not the budget itself. Rebuilding
#: to exactly the budget leaves no headroom, so the very next step breaches it
#: again and the trajectory rebuilds every step — measured at 104 rebuilds across
#: 14 HC trajectories before this was added.
RETAIN_TARGET = 0.8

#: Rebuild count above which the log complains. The budget is still enforced past
#: it: exceeding the prefix cap silently is worse than an expensive trajectory.
REBUILD_WARN_AT = 8

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
    #: z-scored under the file's leave-one-out fold statistics. The raw score
    #: stays beside it so both are visible downstream and in the JSONL.
    p_norm: float | None = None
    augmented: bool = False
    judge: str = ""
    readout: str = "ptrue"
    policy: str = "typed"
    with_gt: bool = False
    use_types: bool = True
    subtask_pointer: bool = True
    peer_corroboration: bool = True
    prefix_window: int = 0
    lookahead: str = "none"
    n_lookahead: int = 0
    n_demoted: int = 0
    #: False when a generated readout could not be parsed. Such rows are scored
    #: 0.5 and flagged, never dropped — discarding them would flatter whichever
    #: readout is worst at following the format, which is what E2 measures.
    parse_ok: bool = True
    prefix_tokens: int = 0
    readout_tokens: int = 0
    seconds: float = 0.0
    raw_text: str = ""

    @property
    def key(self) -> str:
        return f"{self.subset}/{self.file_id}"

    @property
    def p(self) -> float:
        """The score the rules read: normalized when available, raw otherwise."""
        return self.p_norm if self.p_norm is not None else self.p_raw

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
    rebuilds: int = 0
    demoted_steps: list[int] = field(default_factory=list)


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


def pointers(
    record: Record,
    t: int,
    blocks: Sequence[int],
    *,
    subtask: bool = True,
    peers: bool = True,
) -> list[str]:
    """Assigned subtask + earlier same-turn peers for a terse ``execute`` step.

    The two are separately switchable because E5 ablates them separately:
    "does peer corroboration earn its place" is a pending direction decision,
    and it cannot be answered if the subtask pointer moves at the same time.

    Strictly backward-looking. Letting a later step in would make the score
    non-causal and inflate attribution accuracy for free.
    """
    steps = record.steps
    target = steps[t]
    out: list[str] = []
    if subtask:
        pat = re.compile(rf"\b{re.escape(target.agent)}\b", re.IGNORECASE)
        for j in range(t - 1, -1, -1):
            s = steps[j]
            if s.type_norm not in ("plan", "delegate"):
                continue
            if pat.search(s.content) or pat.search(s.role_raw) or (
                j == t - 1 and s.type_norm == "delegate"
            ):
                out.append(
                    f"assigned subtask for {target.agent}:\n"
                    f"{s.content.strip()[:MAX_POINTER_CHARS]}"
                )
                break
    if peers:
        same_turn = [
            s
            for j, s in enumerate(steps[:t])
            if blocks[j] == blocks[t] and s.agent != target.agent and s.content.strip()
        ]
        for peer in same_turn[-2:]:
            out.append(
                f"peer step {peer.idx} ({peer.agent}, {peer.type_norm}):\n"
                f"{peer.content.strip()[:MAX_POINTER_CHARS]}"
            )
    return out


def lookahead_steps(steps: Sequence[Step], t: int, mode: str) -> list[Step]:
    """The steps appended after ``t`` under a lookahead arm.

    ``resp``: the immediately following *contiguous* run of steps by other
    agents, capped at ``RESP_CAP`` — contiguous because the point is the direct
    realized response, not everything that ever followed.
    ``own``: that, plus the acting agent's next appearance anywhere later.
    """
    if mode not in LOOKAHEAD:
        raise ValueError(f"unknown lookahead {mode!r}; known: {LOOKAHEAD}")
    if mode == "none":
        return []
    actor = steps[t].agent
    # A delegation's window runs to the subtask's resolution — until control
    # returns to the delegator — rather than to the next couple of messages.
    cap = (
        DELEG_CAP
        if mode == "deleg" and steps[t].type_norm == "delegate"
        else RESP_CAP
    )
    out: list[Step] = []
    for s in steps[t + 1 :]:
        if s.agent == actor or len(out) >= cap:
            break
        out.append(s)
    if mode == "own":
        nxt = next((s for s in steps[t + 1 :] if s.agent == actor), None)
        if nxt is not None:
            out.append(nxt)
    return out


def render_lookahead(window: Sequence[Step]) -> str:
    if not window:
        return ""
    body = "".join(render_step(s)[0] for s in window)
    return f"\n[what happened next — realized response]\n{body}"


def untyped(steps: Sequence[Step]) -> tuple[Step, ...]:
    """Strip act types — the 'typing off' arm of E4."""
    return tuple(s.typed("unknown", s.type_source) for s in steps)


# --- truncation: type-aware retention ---------------------------------------


def step_header(step: Step) -> str:
    """A demoted step: its row survives, its detail does not."""
    body = (step.content or "").strip().replace("\n", " ")[:HEADER_CHARS]
    elided = max(len(step.content or "") - HEADER_CHARS, 0)
    return (
        f"[step {step.idx} | agent={step.agent} | type={step.type_norm}]\n"
        f"{body}… [{elided} chars withheld — old execution detail]\n"
    )


def retained_render(head: str, steps: Sequence[Step], budget: int) -> tuple[str, list[int]]:
    """Render ``steps`` under the pre-registered retention policy.

    Always kept verbatim: the task header (and, in the with-GT setting, the
    reference answer), and every ``plan``/``delegate`` step — they are
    structural, short, and carry the delegation errors the paper is about.
    Execute and final steps are kept verbatim newest-first while budget allows;
    the rest are demoted to a header line.

    Prefix-only asymmetry is untouched: nothing here reaches past ``steps``.
    What degrades is *old execution detail*, which is the defensible thing to
    degrade.
    """
    coordination = [s for s in steps if s.type_norm in ("plan", "delegate")]
    other = [s for s in steps if s.type_norm not in ("plan", "delegate")]

    # Start from the floor — every structural step verbatim, every execution
    # step demoted — then spend what is left upgrading executions back to full,
    # newest first. Budgeting this way round is what makes the bound hold: the
    # header lines are charged before anything is kept verbatim, so a long
    # trajectory cannot blow the budget on headers it never accounted for.
    rendered: dict[int, str] = {s.idx: render_step(s) for s in coordination}
    rendered.update({s.idx: step_header(s) for s in other})
    used = len(head) + sum(len(v) for v in rendered.values())
    demoted = {s.idx for s in other}

    for s in reversed(other):  # newest execution first
        full = render_step(s)
        upgrade = len(full) - len(rendered[s.idx])
        if used + upgrade > budget:
            continue
        rendered[s.idx] = full
        used += upgrade
        demoted.discard(s.idx)

    return head + "".join(rendered[s.idx] for s in steps), sorted(demoted)


# --- scoring ----------------------------------------------------------------


def score_record(
    record: Record,
    client: JudgeClient,
    *,
    kind: str = "ptrue",
    policy: str = "typed",
    with_gt: bool = False,
    use_types: bool = True,
    subtask_pointer: bool = True,
    peer_corroboration: bool = True,
    prefix_window: int | None = None,
    lookahead: str = "none",
    budget_chars: int = PREFIX_BUDGET_CHARS,
) -> TrajectoryScores:
    """Score every step of one trajectory against a shared, growing prefix.

    ``prefix_window`` is the prefix-slice arm of E5: when set, step ``t`` is
    judged against only the last ``prefix_window`` steps rather than all of
    ``0..t``. It forces a rebuild per step, so it is an ablation, not a mode to
    run the primary numbers in.
    """
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
    used = 0
    demoted: list[int] = []

    if policy == "hindsight":
        # One fixed prefix carrying the whole trajectory; each readout still
        # costs only its own tokens, so the ceiling stays O(T) to compute.
        text, demoted = retained_render(head, steps, budget_chars)
        client.reset(text)
        out.rebuilds = int(bool(demoted))

    for t, step in enumerate(steps):
        if policy != "hindsight":
            window = steps[max(0, t + 1 - prefix_window) : t + 1] if prefix_window else None
            if window is not None:
                # Prefix slice: the visible history changes shape every step, so
                # the shared prefix cannot be reused.
                text, demoted = retained_render(head, window, budget_chars)
                client.reset(text)
                used = len(text)
            else:
                if t == 0:
                    client.reset(head)
                    used = len(head)
                rendered = render_step(step)
                if used + len(rendered) > budget_chars:
                    # Over budget: rebuild the shared prefix with old execution
                    # detail demoted, leaving headroom so the next step does not
                    # immediately breach again.
                    text, demoted = retained_render(
                        head, steps[: t + 1], int(budget_chars * RETAIN_TARGET)
                    )
                    client.reset(text)
                    used = len(text)
                    out.rebuilds += 1
                    out.demoted_steps = list(demoted)
                else:
                    client.extend(rendered)
                    used += len(rendered)

        augment = ""
        if policy == "typed" and step.type_norm == "execute" and is_near_empty(step):
            ptrs = pointers(
                record.with_steps(steps),
                t,
                blocks,
                subtask=subtask_pointer,
                peers=peer_corroboration,
            )
            if ptrs:
                augment = (
                    "\n[context for a terse step — within-trajectory only]\n"
                    + "\n\n".join(ptrs)
                    + "\n"
                )

        window = lookahead_steps(steps, t, lookahead)
        prompt = augment + render_lookahead(window) + readout(step, kind)
        parse_ok = True
        if kind == "ptrue":
            p, trace = client.p_true(prompt)
            text = ""
        else:
            text, trace = client.generate(prompt, max_new_tokens=12)
            p, parse_ok = _parse_generated(text, kind)

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
                subtask_pointer=subtask_pointer,
                peer_corroboration=peer_corroboration,
                prefix_window=prefix_window or 0,
                lookahead=lookahead,
                n_lookahead=len(window),
                n_demoted=len(demoted),
                parse_ok=parse_ok,
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


def _parse_generated(text: str, kind: str) -> tuple[float, bool]:
    """Turn a generated readout into a score.

    Unparseable generations become 0.5 and keep their raw text on the row —
    dropping them would quietly flatter the readouts that are worst at
    following the format, which is precisely what E2 is measuring.
    """
    t = (text or "").strip()
    if kind == "binary":
        low = t.lower()
        if low.startswith("true") or low.startswith("yes"):
            return 1.0, True
        if low.startswith("false") or low.startswith("no"):
            return 0.0, True
        return 0.5, False
    m = _NUM.search(t)
    if not m:
        return 0.5, False
    v = float(m.group(1))
    if v > 1.0:
        v /= 100.0
    return min(max(v, 0.0), 1.0), True


def score_corpus(
    records: Sequence[Record],
    client: JudgeClient,
    *,
    kind: str = "ptrue",
    policy: str = "typed",
    with_gt: bool = False,
    use_types: bool = True,
    subtask_pointer: bool = True,
    peer_corroboration: bool = True,
    prefix_window: int | None = None,
    lookahead: str = "none",
    budget_chars: int = PREFIX_BUDGET_CHARS,
    out_path: str | Path | None = None,
    resume: bool = True,
    progress: Callable[[int, int, TrajectoryScores], None] | None = None,
) -> list[TrajectoryScores]:
    """Score a corpus, appending to ``out_path`` and skipping completed files.

    Resume is on by default because these runs are long and the box is shared:
    losing an hour of finished work to someone else's memory spike is avoidable.
    A file counts as done only when its row count matches its step count, so a
    trajectory interrupted mid-way is redone rather than left short.
    """
    results: list[TrajectoryScores] = []
    done: set[str] = set()
    if out_path and resume and Path(out_path).exists():
        counts: dict[str, int] = {}
        for row in load_scores(out_path):
            counts[row.key] = counts.get(row.key, 0) + 1
        want = {r.key: r.n_steps for r in records}
        done = {k for k, n in counts.items() if n == want.get(k)}
        stale = [k for k in counts if k not in done]
        if stale:
            # Drop partial trajectories so the file never holds a half-scored one.
            rows = [r for r in load_scores(out_path) if r.key in done]
            with open(out_path, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r.to_dict()) + "\n")
    fh = open(out_path, "a" if done else "w", encoding="utf-8") if out_path else None
    try:
        for i, rec in enumerate(records):
            if rec.key in done:
                continue
            ts = score_record(
                rec,
                client,
                kind=kind,
                policy=policy,
                with_gt=with_gt,
                use_types=use_types,
                subtask_pointer=subtask_pointer,
                peer_corroboration=peer_corroboration,
                prefix_window=prefix_window,
                lookahead=lookahead,
                budget_chars=budget_chars,
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
        # Truncation is pre-registered, so its extent is reported, not discovered.
        "prefix_rebuilds": sum(r.rebuilds for r in results),
        "trajectories_truncated": sum(1 for r in results if r.rebuilds),
        "fraction_trajectories_truncated": round(
            sum(1 for r in results if r.rebuilds) / len(results), 4
        ),
        "assessments_with_demoted_steps": sum(
            1 for r in results for s in r.scores if s.n_demoted
        ),
        "fraction_assessments_truncated": round(
            sum(1 for r in results for s in r.scores if s.n_demoted) / max(sum(steps), 1), 4
        ),
        "max_demoted_steps": max((s.n_demoted for r in results for s in r.scores), default=0),
        # Parse failures are reported, not dropped: silently discarding them
        # would flatter whichever readout follows the format worst.
        "n_parse_failures": sum(
            1 for r in results for s in r.scores if not s.parse_ok
        ),
        "parse_failure_rate": round(
            sum(1 for r in results for s in r.scores if not s.parse_ok)
            / max(sum(steps), 1),
            4,
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
