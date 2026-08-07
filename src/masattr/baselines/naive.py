"""Judge-free sanity baselines (pilot baseline spec B1).

Five predictors that use no model at all — only trajectory shape and the label
distribution's known properties. Every field-based row has to clear these, and
the two that encode a corpus regularity are the sharp ones:

* ``prior_position`` bakes in the reported early skew (gold normalized median
  ≈ 0.33). If a scored field cannot beat a constant guess at 0.33, the field is
  not contributing localization.
* ``majority_agent`` bakes in trajectory composition. On HC the orchestrator
  owns most steps, so this row is the one to read against E1's orchestrator
  cell before crediting the field for that number.

Each returns ``(agent, step)`` directly; no attribution rule is involved.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from typing import Sequence

from ..record import Record

#: The reported early skew of the decisive step, as a normalized position.
PRIOR_POSITION = 0.33


def _owner(record: Record, step: int) -> str:
    step = max(0, min(step, record.n_steps - 1))
    return record.steps[step].agent


def prior_position(record: Record) -> tuple[str, int]:
    step = int(round(PRIOR_POSITION * (record.n_steps - 1)))
    return _owner(record, step), step


def first_step(record: Record) -> tuple[str, int]:
    return _owner(record, 0), 0


def last_step(record: Record) -> tuple[str, int]:
    return _owner(record, record.n_steps - 1), record.n_steps - 1


def majority_agent(record: Record) -> tuple[str, int]:
    """Most-step-owning agent; step = the median index among that agent's steps."""
    counts = Counter(s.agent for s in record.steps)
    agent = max(sorted(counts), key=lambda a: counts[a])
    idxs = [s.idx for s in record.steps if s.agent == agent]
    return agent, int(statistics.median(idxs))


PREDICTORS = {
    "prior_position": prior_position,
    "majority_agent": majority_agent,
    "first_step": first_step,
    "last_step": last_step,
}


def uniform_random_expectations(
    records: Sequence[Record], *, draws: int = 100, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """Per-file expected accuracy of a uniform random step, over seeded draws.

    Returns ``{key: (agent_rate, step_rate)}``. Reporting the per-file
    expectation rather than one sampled prediction keeps the bootstrap over
    files meaningful: the randomness is averaged out inside each file, so the
    interval reflects file-to-file variation, which is what the CI is for.
    """
    rng = random.Random(seed)
    out: dict[str, tuple[float, float]] = {}
    for rec in records:
        gold_agent, gold_step = rec.gold
        from ..typing.normalize import collapse_orchestrator

        want = collapse_orchestrator(gold_agent)
        a = s = 0
        for _ in range(draws):
            i = rng.randrange(rec.n_steps)
            a += collapse_orchestrator(rec.steps[i].agent) == want
            s += i == gold_step
        out[rec.key] = (a / draws, s / draws)
    return out
