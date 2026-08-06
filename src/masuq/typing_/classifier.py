"""Rule-based step-type classifier (spec §2, v1).

Used on the ``classified`` subsets (MATU-CAMEL, W&W-AG) where no native or
parseable type field exists. Deliberately rule-based and cheap: an LLM
classifier is only justified if rule coverage falls below 90% on the manual
audit, and the rules' agreement with the *native* (MATU-AutoGen) and *parsed*
(W&W-HC) types is itself a reportable table (see :mod:`masuq.typing_.validate`).

Each rule returns a :class:`Verdict` carrying the matched rule name, so coverage
and per-rule precision can be audited rather than asserted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..schema import Step, TypeNorm

# --- lexical resources ------------------------------------------------------

#: Agent names that, across both corpora, denote a coordinator rather than a worker.
ORCHESTRATOR_NAMES = {
    "orchestrator",
    "staragent",
    "star_agent",
    "manager",
    "chat_manager",
    "chatmanager",
    "planner",
    "admin",
    "user_proxy",
    "userproxy",
    "coordinator",
    "supervisor",
}

#: Keys that mark a JSON blob as a plan/delegation payload (MATU StarAgent style).
PLAN_JSON_KEYS = {
    "analyst_task",
    "verifier_task",
    "subtasks",
    "sub_tasks",
    "plan",
    "steps",
    "task_assignment",
    "assignments",
    "next_agent",
    "next_speaker",
}

_ANSWER_PAT = re.compile(
    r"(?:^|\b)(?:final\s+answer|the\s+answer\s+is|answer\s*[::]|"
    r"boxed\{|\\boxed\{|therefore,?\s+the\s+answer)",
    re.IGNORECASE,
)
_TERMINATE_PAT = re.compile(r"\bTERMINATE\b|\bTASK[_ ]COMPLETE\b", re.IGNORECASE)
_BARE_CHOICE_PAT = re.compile(r"^\s*[\(\[]?([A-Ea-e1-9])[\)\].]?\s*$")

#: Handoff language. Deliberately unanchored: in these corpora the delegating
#: phrase usually follows the addressee ("WebSurfer, please proceed…"), so a
#: line-start anchor would miss the most common form.
_DELEGATE_PAT = re.compile(
    r"\b(?:next\s+speaker|hand(?:s|ing)?\s*off\s+to|delegat(?:e|ing)\s+to|"
    r"assign(?:ing|ed)?\s+(?:this\s+)?to|please\s+(?:proceed|take\s+over|go\s+ahead)|"
    r"over\s+to\s+you)|@[A-Za-z_][A-Za-z0-9_]*\s*[,:]",
    re.IGNORECASE,
)
_ARROW_PAT = re.compile(r"->\s*([A-Za-z_][A-Za-z0-9_ ]*)")

#: "WebSurfer, find the capital of Peru." — a coordinator addressing a named
#: agent by name and then issuing an instruction. Common in both Magentic-One
#: and AutoGen transcripts, and not caught by handoff phrasing alone.
_ADDRESSED_PAT = re.compile(r"^\s*([A-Z][A-Za-z0-9_]{2,})\s*,\s+[a-z]")

_PLAN_PAT = re.compile(
    r"(?:^|\n)\s*(?:here(?:'s| is)\s+(?:the|my)\s+plan|plan\s*[::]|"
    r"i\s+will\s+(?:first|now)\b|step\s*1\s*[::.]|to\s+solve\s+this,?\s+we\s+(?:will|need))",
    re.IGNORECASE,
)

_TOOL_OUTPUT_PAT = re.compile(
    r"(?:^|\n)\s*(?:exitcode\s*[::]|Traceback \(most recent call last\)|"
    r"```(?:python|bash|sh|output)|Address:\s*http|Viewport position|"
    r"Here is a screenshot|The web browser is open|File .*? line \d+)",
    re.IGNORECASE,
)

#: Below this length an ``execute`` step carries essentially no evidence of its
#: own; the judge harness compensates via the type-conditioned evidence policy
#: (spec §3). Kept here because the same threshold defines the rule.
NEAR_EMPTY_CHARS = 16


@dataclass(slots=True)
class Verdict:
    type_norm: TypeNorm
    rule: str
    confident: bool = True

    @property
    def covered(self) -> bool:
        """True when a positive rule fired (as opposed to the fallback)."""
        return self.rule != "fallback"


def _looks_like_plan_json(content: str) -> bool:
    text = content.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return False
    blob = text[start : end + 1]
    try:
        obj = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        # StarAgent embeds JSON inside a string; fall back to key sniffing.
        return any(f'"{k}"' in blob for k in PLAN_JSON_KEYS)
    if isinstance(obj, dict):
        return bool(PLAN_JSON_KEYS & {k.lower() for k in obj.keys()})
    return False


def is_orchestrator(agent: str) -> bool:
    a = agent.strip().lower().replace(" ", "_")
    return a in ORCHESTRATOR_NAMES or "orchestr" in a or "manager" in a


def is_answer_emission(content: str) -> bool:
    """Does this step emit a final answer rather than work toward one?"""
    text = content.strip()
    if not text:
        return False
    if _ANSWER_PAT.search(text) or _TERMINATE_PAT.search(text):
        return True
    # A whole step consisting of a bare option letter is an answer emission in
    # both MMLU-style corpora ("B"), and is exactly the near-empty case the
    # evidence policy has to rescue.
    return bool(_BARE_CHOICE_PAT.match(text))


def classify_step(
    content: str,
    agent: str,
    *,
    is_last: bool = False,
    role_raw: str = "",
) -> Verdict:
    """Classify a single step. ``is_last`` marks the final step of its trajectory."""
    text = (content or "").strip()
    role = (role_raw or "").strip().lower()

    if not text:
        return Verdict("unknown", "empty_content", confident=False)

    # 1. Explicit final-answer emission. Checked before plan/delegate because a
    #    coordinator's last message is an answer, not a plan.
    if is_answer_emission(text) and (is_last or _TERMINATE_PAT.search(text)):
        return Verdict("final", "answer_emission_last")

    # 2. Structured plan payload (StarAgent-style JSON with task assignments).
    if _looks_like_plan_json(text):
        return Verdict("plan", "plan_json")

    # 3. Delegation: explicit handoff language, or an orchestrator arrow.
    if _ARROW_PAT.search(text) and is_orchestrator(agent):
        return Verdict("delegate", "orchestrator_arrow")
    if _DELEGATE_PAT.search(text):
        return Verdict("delegate", "handoff_language")
    if is_orchestrator(agent) and _ADDRESSED_PAT.match(text):
        return Verdict("delegate", "addressed_agent")

    # 4. Tool / environment output is execution by definition.
    if _TOOL_OUTPUT_PAT.search(text) or role in {"tool", "function", "tool_call", "observation"}:
        return Verdict("execute", "tool_output")

    # 5. Planning language from a coordinator.
    if _PLAN_PAT.search(text):
        return Verdict("plan", "plan_language")
    if is_orchestrator(agent) and not is_last:
        return Verdict("plan", "orchestrator_default", confident=False)

    # 6. Answer emission anywhere else (mid-trajectory "the answer is …") still
    #    reads as a final-type act by a worker.
    if is_answer_emission(text):
        return Verdict("final", "answer_emission")

    # 7. Everything else a worker says is execution.
    if len(text) < NEAR_EMPTY_CHARS:
        return Verdict("execute", "near_empty_worker", confident=False)
    return Verdict("execute", "fallback", confident=False)


def classify_trajectory(steps: list[Step]) -> list[Verdict]:
    """Classify every step, giving each rule the ``is_last`` context it needs."""
    n = len(steps)
    return [
        classify_step(s.content, s.agent, is_last=(i == n - 1), role_raw=s.role_raw)
        for i, s in enumerate(steps)
    ]


def apply_classifier(steps: list[Step], *, overwrite: bool = False) -> list[Verdict]:
    """Write ``type_norm``/``type_source`` onto steps whose type is not already known.

    With ``overwrite=False`` (default) steps carrying ``native`` or ``parsed``
    types are left untouched — the classifier never overrides ground truth about
    typing. Verdicts are still returned for every step so the validation table
    in :mod:`masuq.typing_.validate` can compare them.
    """
    verdicts = classify_trajectory(steps)
    for s, v in zip(steps, verdicts):
        if overwrite or s.type_source == "classified":
            s.type_norm = v.type_norm
            s.type_source = "classified"
    return verdicts


def coverage(verdicts: list[Verdict]) -> float:
    """Fraction of steps decided by a positive rule (spec §2 gate: rules stay if ≥0.90)."""
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v.covered) / len(verdicts)
