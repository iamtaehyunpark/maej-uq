"""Prompt scaffold (spec v3 Part C §3).

One scaffold, three readouts. The readout ablation (E2) is only meaningful if
the logit, verbalized, and binary-verdict rows see *identical* context and
question — so the preamble and the question text are shared, and only the final
instruction line differs.

Two settings run for every primary table: **without-GT** (the deployable case)
and **with-GT**, where the ground truth is appended to the preamble. With-GT is
not a better version of the method; it is the upper bound on what the judge
could know, and reporting both is what keeps the without-GT number honest.
"""

from __future__ import annotations

import hashlib

from ..record import Step

SYSTEM = (
    "You are auditing a transcript from a multi-agent system that failed to "
    "solve its task. You will see the task and the transcript up to and "
    "including one step. Judge only that step, given everything before it. "
    "Do not use hindsight you do not have."
)

#: What "correct/appropriate" means per act type. A plan and a tool call fail in
#: different ways, and a single rubric would flatten that.
RUBRIC: dict[str, str] = {
    "plan": (
        "This step is a PLAN. It is appropriate if the decomposition is sound, "
        "covers the task, and assigns work the named agents can do."
    ),
    "delegate": (
        "This step is a DELEGATION. It is appropriate if the chosen agent is "
        "right for the outstanding subtask and the handoff carries what that "
        "agent needs."
    ),
    "execute": (
        "This step is an EXECUTION. It is correct if the work performed is "
        "accurate and advances the subtask this agent was assigned."
    ),
    "final": (
        "This step is a FINAL ANSWER emission. It is correct if the answer "
        "follows from the transcript and is well-formed for the task."
    ),
    "unknown": (
        "The act type of this step could not be determined. Judge whether it is "
        "a correct and appropriate contribution given the transcript so far."
    ),
}

READOUTS = ("ptrue", "verbalized", "binary")

#: Closed, empty scratchpad appended to every readout.
#:
#: A reasoning judge does not answer at prefill+1 — it opens a scratchpad
#: first. Measured on the served model, the next-token mass after the bare
#: question is 93% on "\n\n" and ~1% on {True, False}; with this block
#: prefilled it is 54%/9% on the two answer spellings. Without it the logit
#: readout is renormalising over two tokens the model was not about to emit,
#: and the generated readouts spend hundreds of tokens reasoning before
#: reaching an answer — which is what truncated them.
#:
#: It is appended identically for all three readouts, because E2 is only an
#: ablation if the scaffold is shared.
NO_THINK = "\n<think>\n\n</think>\n\n"

_INSTRUCTION = {
    "ptrue": "Answer with a single word, True or False.\nAnswer:",
    "binary": "Answer with a single word, True or False.\nVerdict:",
    "verbalized": (
        "Reply with only your confidence that it is correct, as a probability "
        "between 0.00 and 1.00.\nConfidence:"
    ),
}


def preamble(query: str, ground_truth: str = "", *, with_gt: bool = False) -> str:
    """Shared header: system instruction, task, and — in the with-GT setting —
    the reference answer."""
    out = [SYSTEM, "", "[task]", query or "(not recorded)"]
    if with_gt:
        out += ["", "[reference answer]", ground_truth or "(not recorded)"]
    return "\n".join(out) + "\n"


def render_step(step: Step, max_chars: int = 4000) -> str:
    content = step.content or ""
    if len(content) > max_chars:
        half = max_chars // 2
        content = (
            f"{content[:half]}\n…[{len(step.content) - max_chars} chars elided]…\n"
            f"{content[-half:]}"
        )
    return f"[step {step.idx} | agent={step.agent} | type={step.type_norm}]\n{content}\n"


def readout(step: Step, kind: str = "ptrue") -> str:
    """The question, identical across readouts but for its final instruction."""
    if kind not in READOUTS:
        raise ValueError(f"unknown readout {kind!r}; known: {READOUTS}")
    return (
        f"\n{RUBRIC.get(step.type_norm, RUBRIC['unknown'])}\n"
        f"Question: Is step {step.idx} by '{step.agent}' correct and appropriate "
        f"given the context above? {_INSTRUCTION[kind]}" + NO_THINK
    )


def prompt_text() -> str:
    """Canonical serialisation of every prompt string, for the frozen hash."""
    parts = [SYSTEM]
    parts += [f"{k}::{v}" for k, v in sorted(RUBRIC.items())]
    parts += [f"{k}::{v}" for k, v in sorted(_INSTRUCTION.items())]
    parts.append(f"no_think::{NO_THINK}")
    return "\n".join(parts)


def prompt_hash() -> str:
    return hashlib.sha256(prompt_text().encode()).hexdigest()[:16]
