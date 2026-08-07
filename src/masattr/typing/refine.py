"""Hierarchical typing: rules for the coarse split, an LLM for plan vs delegate.

Measured on the released Hand-Crafted subset (2935 parsed steps), the rules of
:mod:`masattr.typing.normalize` split **coordination / execute / final** at
0.9935 — 9 errors — but split **plan vs delegate inside coordination** at
0.4162, which is *below* the 0.6934 majority-class baseline. The rules carry no
usable signal there.

The diagnosis is structural, not a tuning failure. In the Magentic-One idiom the
distinction lives in the role (``Orchestrator (thought)`` vs
``Orchestrator (-> WebSurfer)``), not in the text: a ledger and a handoff both
name an agent and both read like instructions. So this module keeps the rules
for what they do well and escalates only the sub-split — the ~76% of HC parsed
steps that are coordination — to an LLM classifier, per Part C §2.

Two constraints hold here:

* The LLM classifier must be **family-disjoint from the judge**
  (Part C §Validity): typing conditions the judge's evidence policy, so sharing
  a family closes a loop.
* The splitter is itself gated. It is validated on HC, where plan/delegate is
  parsed and therefore known, before it is allowed to touch AG.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from ..models import check_disjoint
from ..record import Record, Step
from .normalize import classify_steps

COORDINATION = ("plan", "delegate")

#: Gate on the sub-split, mirroring Part C §2's gate on the rules as a whole.
SPLIT_GATE = 0.90


def coarse_of(type_norm: str) -> str:
    return "coordination" if type_norm in COORDINATION else type_norm


# --- the splitter -----------------------------------------------------------

PROMPT = """You are labelling one step from a multi-agent system transcript.

The step was produced by a coordinating agent. Decide which of two acts it is:

PLAN — the coordinator is reasoning, tracking state, or laying out what to do:
inner monologue, a ledger or status update, a decomposition of the task.
DELEGATE — the coordinator is handing work to a named agent right now: an
instruction addressed to that agent, expecting them to act next.

A step that names another agent is not automatically a delegation; coordinators
name agents while planning too. Ask who is meant to act on this message.

Task: {query}

Preceding step ({prev_agent}, {prev_type}): {prev}

Step under review, by '{agent}':
{content}

Answer with one word, PLAN or DELEGATE.
Answer:"""

MAX_CONTENT_CHARS = 1500


@dataclass(slots=True)
class SplitVerdict:
    type_norm: str
    raw: str = ""
    parsed: bool = True


class CoordinationSplitter(ABC):
    """Splits a coordination step into ``plan`` or ``delegate``."""

    name: str = "abstract"

    @abstractmethod
    def split(self, record: Record, idx: int) -> SplitVerdict: ...

    def prompt_for(self, record: Record, idx: int) -> str:
        step = record.steps[idx]
        prev = record.steps[idx - 1] if idx else None
        return PROMPT.format(
            query=(record.query or "(not recorded)")[:600],
            prev_agent=prev.agent if prev else "—",
            prev_type=prev.type_norm if prev else "—",
            prev=((prev.content or "")[:400] if prev else "(none)"),
            agent=step.agent,
            content=(step.content or "")[:MAX_CONTENT_CHARS],
        )


class MockSplitter(CoordinationSplitter):
    """Deterministic stand-in so the pipeline runs without a second model.

    Never appears in a reported number; `retype` refuses to write AG types from
    it unless explicitly forced.
    """

    name = "mock"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def split(self, record: Record, idx: int) -> SplitVerdict:
        import hashlib

        h = hashlib.sha256(f"{self.seed}|{record.key}|{idx}".encode()).digest()[0]
        return SplitVerdict("delegate" if h % 2 else "plan", raw="mock")


_ANSWER = re.compile(r"\b(PLAN|DELEGATE)\b", re.IGNORECASE)


class LLMSplitter(CoordinationSplitter):
    """Generation-based splitter over any :class:`~masattr.judge.client.JudgeClient`.

    Reuses the judge client interface only as a transport — the *model* must be
    family-disjoint from the judge, which :func:`build_splitter` enforces.
    """

    def __init__(self, client, *, model_id: str = "") -> None:
        self.client = client
        self.model_id = model_id or getattr(client, "name", "?")
        self.name = f"llm:{self.model_id}"
        self.n_unparsed = 0

    def split(self, record: Record, idx: int) -> SplitVerdict:
        self.client.reset("")
        text, _ = self.client.generate(self.prompt_for(record, idx), max_new_tokens=4)
        m = _ANSWER.search(text or "")
        if not m:
            self.n_unparsed += 1
            # Unparseable answers fall back to the majority class and are
            # counted; silently dropping them would flatter the gate.
            return SplitVerdict("plan", raw=text or "", parsed=False)
        return SplitVerdict(m.group(1).lower(), raw=text or "")


def build_splitter(spec: str, *, judge_model: str, device: str | None = None, seed: int = 0):
    """``mock`` | ``hf:<model_id>``, checked disjoint from ``judge_model``."""
    if spec == "mock":
        check_disjoint("type-classifier", spec, "judge", judge_model, strict=False)
        return MockSplitter(seed=seed)
    from ..judge.client import build_client

    check_disjoint("type-classifier", spec, "judge", judge_model)
    return LLMSplitter(build_client(spec, device=device, seed=seed), model_id=spec)


# --- validation of the sub-split -------------------------------------------


@dataclass(slots=True)
class SplitReport:
    splitter: str
    n: int
    agreement: float
    majority_baseline: float
    confusion: dict[str, dict[str, int]]
    n_unparsed: int = 0
    rules_agreement: float = 0.0
    coarse_agreement: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        # Beating the gate is necessary; beating the majority class is the point.
        return self.agreement >= SPLIT_GATE and self.agreement > self.majority_baseline

    def to_dict(self) -> dict:
        return {
            "splitter": self.splitter,
            "n_coordination_steps": self.n,
            "agreement": self.agreement,
            "majority_baseline": self.majority_baseline,
            "rules_agreement_same_steps": self.rules_agreement,
            "coarse_agreement": self.coarse_agreement,
            "gate": SPLIT_GATE,
            "passes_gate": self.passes,
            "n_unparsed": self.n_unparsed,
            "confusion": self.confusion,
            "notes": self.notes,
        }

    def render(self) -> str:
        rows = [
            f"| **{a}** | " + " | ".join(str(self.confusion.get(a, {}).get(b, 0)) for b in COORDINATION) + " |"
            for a in COORDINATION
        ]
        return "\n".join(
            [
                f"### plan vs delegate within coordination — {self.splitter}",
                f"n={self.n}  agreement={self.agreement:.4f}  "
                f"majority-class baseline={self.majority_baseline:.4f}  "
                f"rules on the same steps={self.rules_agreement:.4f}  "
                f"gate={SPLIT_GATE:.2f} → **{'PASS' if self.passes else 'FAIL'}**",
                "",
                "| parsed \\ splitter | " + " | ".join(COORDINATION) + " |",
                "|---|---|---|",
                *rows,
                "",
                f"coarse split (coordination/execute/final) by rules alone: "
                f"{self.coarse_agreement:.4f}",
                f"unparsed splitter answers: {self.n_unparsed}",
            ]
        )


def validate_splitter(records: Sequence[Record], splitter: CoordinationSplitter) -> SplitReport:
    """Gate the splitter on HC, where plan/delegate is parsed and therefore known."""
    confusion = {a: {b: 0 for b in COORDINATION} for a in COORDINATION}
    n = agree = rules_agree = 0
    coarse_n = coarse_agree = 0
    counts = {a: 0 for a in COORDINATION}

    for rec in records:
        verdicts = classify_steps(rec.steps)
        for i, (step, v) in enumerate(zip(rec.steps, verdicts)):
            if step.type_source != "parsed":
                continue
            coarse_n += 1
            coarse_agree += coarse_of(step.type_norm) == coarse_of(v.type_norm)
            if step.type_norm not in COORDINATION:
                continue
            n += 1
            counts[step.type_norm] += 1
            rules_agree += step.type_norm == v.type_norm
            pred = splitter.split(rec, i).type_norm
            confusion[step.type_norm][pred] += 1
            agree += pred == step.type_norm

    return SplitReport(
        splitter=splitter.name,
        n=n,
        agreement=(agree / n) if n else 0.0,
        majority_baseline=(max(counts.values()) / n) if n else 0.0,
        confusion=confusion,
        n_unparsed=getattr(splitter, "n_unparsed", 0),
        rules_agreement=(rules_agree / n) if n else 0.0,
        coarse_agreement=(coarse_agree / coarse_n) if coarse_n else 0.0,
    )


# --- application ------------------------------------------------------------


def refine_record(record: Record, splitter: CoordinationSplitter) -> tuple[Record, int]:
    """Re-split this record's *classified* coordination steps. Parsed steps are
    never touched — the rules and the splitter both defer to read types."""
    steps: list[Step] = []
    n = 0
    for i, s in enumerate(record.steps):
        if s.type_source == "classified" and s.type_norm in COORDINATION:
            steps.append(s.typed(splitter.split(record, i).type_norm, "classified"))
            n += 1
        else:
            steps.append(s)
    return record.with_steps(steps), n


def refine_records(
    records: Sequence[Record], splitter: CoordinationSplitter
) -> tuple[list[Record], int]:
    out, total = [], 0
    for rec in records:
        refined, n = refine_record(rec, splitter)
        out.append(refined)
        total += n
    return out, total
