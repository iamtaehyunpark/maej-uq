"""Judge prompts (spec §3).

Two readouts over identical evidence, so the comparison isolates *where the
number comes from* rather than what the model was asked:

* :func:`ptrue_readout` — ends immediately before the answer token, so the score
  is ``P("True")`` at prefill+1.
* :func:`verbalized_readout` — same question, model emits a number.

The system preamble is deliberately short and type-aware: the judge is told what
kind of act it is grading, because "is this correct?" means something different
for a plan than for a tool call.
"""

from __future__ import annotations

from ..schema import Step

SYSTEM = (
    "You are auditing a multi-agent system transcript. You will see the task and "
    "the transcript up to and including one step. Judge only that step, given "
    "everything before it. Do not use hindsight you do not have."
)

#: What "correct/appropriate" means per normalised step type.
TYPE_RUBRIC: dict[str, str] = {
    "plan": (
        "This step is a PLAN. It is appropriate if the decomposition is sound, "
        "covers the task, and assigns work that the named agents can do."
    ),
    "delegate": (
        "This step is a DELEGATION. It is appropriate if the chosen agent is the "
        "right one for the outstanding subtask and the handoff carries the "
        "information that agent needs."
    ),
    "execute": (
        "This step is an EXECUTION. It is correct if the work performed is "
        "accurate and actually advances the subtask this agent was assigned."
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


def _header(step: Step) -> str:
    return (
        f"{SYSTEM}\n\n"
        f"Step under review: step {step.idx}, by agent '{step.agent}'.\n"
        f"{TYPE_RUBRIC.get(step.type_norm, TYPE_RUBRIC['unknown'])}\n"
    )


def system_preamble(step: Step) -> str:
    """Prepended to the shared prefix at :meth:`PrefixScorer.reset` time."""
    return _header(step)


def ptrue_readout(step: Step) -> str:
    """Readout whose next token is the judged verdict (single-token, prefill+1)."""
    return (
        f"\n{TYPE_RUBRIC.get(step.type_norm, TYPE_RUBRIC['unknown'])}\n"
        f"Question: Is step {step.idx} by '{step.agent}' correct and appropriate "
        f"given the context above? Answer with a single word, True or False.\n"
        f"Answer:"
    )


def verbalized_readout(step: Step) -> str:
    """Same question, numeric self-report (baseline row of spec §3)."""
    return (
        f"\n{TYPE_RUBRIC.get(step.type_norm, TYPE_RUBRIC['unknown'])}\n"
        f"Question: Is step {step.idx} by '{step.agent}' correct and appropriate "
        f"given the context above? Reply with only your confidence that it is "
        f"correct, as a probability between 0.00 and 1.00.\n"
        f"Confidence:"
    )


READOUTS = {
    "ptrue": ptrue_readout,
    "verbalized": verbalized_readout,
}
