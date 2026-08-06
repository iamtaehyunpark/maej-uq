"""Who&When baseline reproduction — all_at_once / step_by_step / binary_search (spec §6).

Each method is run twice: once with the upstream README default (gpt-4o) and
once with our judge model. Running both is what separates "their method is
weaker" from "their judge was stronger" — without the second arm, any gap we
report is confounded with judge capability.

Scoring uses the dual scorer of :mod:`masuq.metrics`, and results are reported
with and without the ``agent_step_mismatch``-flagged files.

AgenTracer / StepFinder are cited from their published numbers only; the pilot
does not reproduce them (spec §6).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..judge.backends import TextGenerator
from ..metrics import score_attribution
from ..schema import FLAG_AGENT_STEP_MISMATCH, Record

MAX_STEP_CHARS = 1500


def _render_steps(record: Record, lo: int, hi: int) -> str:
    out = []
    for s in record.steps[lo:hi]:
        body = (s.content or "")[:MAX_STEP_CHARS]
        out.append(f"[step {s.idx}] {s.agent}: {body}")
    return "\n".join(out)


def _header(record: Record) -> str:
    return (
        "A multi-agent system failed to solve the following task.\n\n"
        f"Task: {record.query or '(not recorded)'}\n"
        f"Ground truth: {record.ground_truth or '(not recorded)'}\n"
    )


_STEP_PAT = re.compile(r"step\s*(?:index\s*)?[:#]?\s*(\d+)", re.IGNORECASE)
_AGENT_PAT = re.compile(r"agent\s*[:#]?\s*([A-Za-z_][\w \-]*)", re.IGNORECASE)


def parse_answer(text: str) -> tuple[str | None, int | None]:
    """Pull ``(agent, step)`` out of a free-form response.

    Tries JSON first, then labelled fields. Unparseable responses return
    ``(None, None)`` and are counted as misses rather than dropped — dropping
    them would quietly inflate the baseline's accuracy.
    """
    if not text:
        return None, None
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                agent = obj.get("agent") or obj.get("mistake_agent")
                step = obj.get("step") or obj.get("mistake_step")
                try:
                    step = int(step) if step is not None else None
                except (TypeError, ValueError):
                    step = None
                return (str(agent) if agent else None), step
        except (json.JSONDecodeError, ValueError):
            pass
    a = _AGENT_PAT.search(text)
    s = _STEP_PAT.search(text)
    return (a.group(1).strip() if a else None), (int(s.group(1)) if s else None)


# --- the three methods ------------------------------------------------------


def all_at_once(record: Record, gen: TextGenerator) -> tuple[str | None, int | None, int]:
    prompt = (
        _header(record)
        + "\nFull transcript:\n"
        + _render_steps(record, 0, len(record.steps))
        + '\n\nIdentify the agent that made the decisive mistake and the step index '
        'where it happened. Reply as JSON: {"agent": "...", "step": N}\n'
    )
    return (*parse_answer(gen.generate(prompt, max_new_tokens=128)), 1)


def step_by_step(record: Record, gen: TextGenerator) -> tuple[str | None, int | None, int]:
    """Walk forward, stopping at the first step judged to be the decisive mistake."""
    calls = 0
    for i, s in enumerate(record.steps):
        prompt = (
            _header(record)
            + "\nTranscript so far:\n"
            + _render_steps(record, 0, i + 1)
            + f"\n\nIs step {i} by '{s.agent}' the decisive mistake that caused the "
            "failure? Answer Yes or No.\n"
        )
        calls += 1
        reply = gen.generate(prompt, max_new_tokens=8)
        if reply and reply.strip().lower().startswith("yes"):
            return s.agent, i, calls
    last = record.steps[-1] if record.steps else None
    return (last.agent if last else None), (last.idx if last else None), calls


def binary_search(record: Record, gen: TextGenerator) -> tuple[str | None, int | None, int]:
    """Halve the interval by asking which side contains the decisive mistake."""
    lo, hi = 0, len(record.steps)
    calls = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        prompt = (
            _header(record)
            + f"\nSegment A (steps {lo}..{mid - 1}):\n"
            + _render_steps(record, lo, mid)
            + f"\n\nSegment B (steps {mid}..{hi - 1}):\n"
            + _render_steps(record, mid, hi)
            + "\n\nWhich segment contains the decisive mistake? Answer A or B.\n"
        )
        calls += 1
        reply = (gen.generate(prompt, max_new_tokens=4) or "").strip().upper()
        if reply.startswith("B"):
            lo = mid
        else:
            hi = mid
    if lo >= len(record.steps):
        return None, None, calls
    return record.steps[lo].agent, lo, calls


BASELINES = {
    "all_at_once": all_at_once,
    "step_by_step": step_by_step,
    "binary_search": binary_search,
}


# --- driver -----------------------------------------------------------------


@dataclass
class BaselineResult:
    subset: str
    method: str
    judge: str
    n: int
    n_calls: int
    n_unparsed: int
    scores: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "subset": self.subset,
            "method": self.method,
            "judge": self.judge,
            "n": self.n,
            "llm_calls": self.n_calls,
            "n_unparsed": self.n_unparsed,
            "scores": self.scores,
        }


def run_baseline(
    records: Sequence[Record],
    gen: TextGenerator,
    *,
    method: str,
    subset: str,
    n_boot: int = 2000,
    limit: int | None = None,
) -> BaselineResult:
    fn = BASELINES[method]
    subset_records = list(records)[: limit or len(records)]
    flagged = {r.key for r in subset_records if FLAG_AGENT_STEP_MISMATCH in r.flags}

    preds: dict[str, tuple[str | None, int | None]] = {}
    total_calls = 0
    unparsed = 0
    for rec in subset_records:
        agent, step, calls = fn(rec, gen)
        total_calls += calls
        if agent is None and step is None:
            unparsed += 1
        preds[rec.key] = (agent, step)

    gold = {r.key: (r.label_mistake_agent, r.label_mistake_step) for r in subset_records}
    scores: dict[str, dict] = {}
    for scorer in ("exact", "substring"):
        for label, keys in (
            ("all", set(preds)),
            ("excl_flagged", set(preds) - flagged),
        ):
            pairs = [(preds[k], gold[k]) for k in sorted(keys)]
            if pairs:
                scores[f"{scorer}/{label}"] = score_attribution(
                    pairs, scorer=scorer, n_boot=n_boot
                ).to_dict()

    return BaselineResult(
        subset=subset,
        method=method,
        judge=gen.name,
        n=len(subset_records),
        n_calls=total_calls,
        n_unparsed=unparsed,
        scores=scores,
    )


def run(
    subsets: dict[str, Sequence[Record]],
    generators: dict[str, TextGenerator],
    *,
    methods: Sequence[str] = tuple(BASELINES),
    out_dir: str | Path | None = None,
    limit: int | None = None,
    n_boot: int = 2000,
) -> list[BaselineResult]:
    results: list[BaselineResult] = []
    for subset, records in subsets.items():
        for gen_name, gen in generators.items():
            for method in methods:
                results.append(
                    run_baseline(
                        records,
                        gen,
                        method=method,
                        subset=subset,
                        limit=limit,
                        n_boot=n_boot,
                    )
                )
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "baselines.json").write_text(
            json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8"
        )
        (out / "baselines.md").write_text(render(results), encoding="utf-8")
    return results


def render(results: Sequence[BaselineResult]) -> str:
    lines = [
        "## Who&When baseline reproduction",
        "",
        "| subset | method | judge | n | calls | unparsed | agent acc (exact) | step acc (exact) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r.scores.get("exact/all", {})
        lines.append(
            f"| {r.subset} | {r.method} | {r.judge} | {r.n} | {r.n_calls} | {r.n_unparsed} | "
            f"{s.get('agent_acc', float('nan')):.3f} | {s.get('step_acc', float('nan')):.3f} |"
        )
    lines += [
        "",
        "> Both judge arms are required: with only the gpt-4o arm, any difference "
        "between these methods and ours is confounded with judge capability.",
        "",
        "> AgenTracer and StepFinder are cited from published numbers; not reproduced here.",
    ]
    return "\n".join(lines)
