"""Step-type normalisation (spec v2 Part C §2).

Two paths, and the distinction is load-bearing:

* **HC — parsed.** The compound role encodes the act: ``Orchestrator (thought)``,
  ``Orchestrator (-> WebSurfer)``, ``WebSurfer``. Types are read, not guessed,
  so HC is the ground truth against which the AG rules are validated.
* **AG — classified.** Only ``{content, role, name}``, so types come from rules:
  JSON-plan detection, delegation verbs, answer emission, tool output.
  ``unknown`` is allowed — a wrong type is worse than an absent one, because the
  calibration maps are per-type.

This is new code, not an adaptation of paper-1's τ rules: those keyed off
environment consequences, and there is no environment here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..record import Step, TypeNorm

# --- HC: parsed from the compound role --------------------------------------

_COMPOUND = re.compile(r"^\s*([^()]+?)\s*(?:\(\s*(.*?)\s*\))?\s*$")
_ARROW = re.compile(r"^->\s*(.+)$")

_QUALIFIER: dict[str, TypeNorm] = {
    "thought": "plan",
    "thinking": "plan",
    "plan": "plan",
    "planning": "plan",
    "final answer": "final",
    "final": "final",
    "answer": "final",
    "termination condition met": "final",
}

_HC_WORKERS = {"websurfer", "assistant", "filesurfer", "coder", "computerterminal", "executor"}


def parse_hc_role(role_raw: str) -> tuple[str, TypeNorm, str | None]:
    """``(agent, type_norm, delegate_target)`` from an HC compound role.

    Delegation keeps the *orchestrator* as the acting agent — the handoff is the
    orchestrator's act, and attributing it to the receiving agent would blame
    the wrong party for a bad delegation.
    """
    m = _COMPOUND.match(role_raw or "")
    if not m:
        return (role_raw.strip() or "unknown"), "unknown", None
    head = (m.group(1) or "").strip()
    qual = (m.group(2) or "").strip()

    if qual:
        arrow = _ARROW.match(qual)
        if arrow:
            return (head or "Orchestrator"), "delegate", arrow.group(1).strip()
        mapped = _QUALIFIER.get(qual.lower())
        if mapped:
            return (head or "Orchestrator"), mapped, None
        return (head or "unknown"), "unknown", None

    low = head.lower()
    if low in _HC_WORKERS:
        return head, "execute", None
    if low.startswith("orchestr"):
        return head, "plan", None
    return (head or "unknown"), "unknown", None


def collapse_orchestrator(agent: str) -> str:
    """Normalise orchestrator naming for agent comparison.

    Annotations spell the orchestrator several ways across files
    (``Orchestrator``, ``orchestrator (thought)``, ``MagenticOneOrchestrator``).
    Collapsing them keeps the ``agent_step_mismatch`` flag and the exact-match
    scorer measuring disagreement rather than spelling.
    """
    a = re.sub(r"\s*\(.*?\)\s*$", "", (agent or "").strip().lower())
    a = a.replace("-", "_").replace(" ", "_")
    if "orchestr" in a or a in {"manager", "chat_manager", "coordinator", "supervisor"}:
        return "orchestrator"
    return a


def is_orchestrator(agent: str) -> bool:
    return collapse_orchestrator(agent) == "orchestrator"


# --- AG: classified from content --------------------------------------------

PLAN_JSON_KEYS = {
    "analyst_task", "verifier_task", "subtasks", "sub_tasks", "plan", "steps",
    "task_assignment", "assignments", "next_agent", "next_speaker",
}

_ANSWER = re.compile(
    r"(?:^|\b)(?:final\s+answer|the\s+answer\s+is|answer\s*[::]|\\?boxed\{|"
    r"therefore,?\s+the\s+answer)",
    re.IGNORECASE,
)
_TERMINATE = re.compile(r"\bTERMINATE\b|\bTASK[_ ]COMPLETE\b", re.IGNORECASE)
_BARE_CHOICE = re.compile(r"^\s*[\(\[]?([A-Ea-e1-9])[\)\].]?\s*$")

#: Unanchored on purpose: the delegating phrase usually follows the addressee
#: ("WebSurfer, please proceed…"), so a line-start anchor misses the common form.
_DELEGATE = re.compile(
    r"\b(?:next\s+speaker|hand(?:s|ing)?\s*off\s+to|delegat(?:e|ing)\s+to|"
    r"assign(?:ing|ed)?\s+(?:this\s+)?to|please\s+(?:proceed|take\s+over|go\s+ahead)|"
    r"over\s+to\s+you)|@[A-Za-z_][A-Za-z0-9_]*\s*[,:]",
    re.IGNORECASE,
)
_ARROW_IN_TEXT = re.compile(r"->\s*([A-Za-z_][A-Za-z0-9_ ]*)")
#: "WebSurfer, find the capital of Peru." — coordinator naming an agent, then an
#: instruction. Frequent in these transcripts and missed by verb patterns alone.
_ADDRESSED = re.compile(r"^\s*([A-Z][A-Za-z0-9_]{2,})\s*,\s+[a-z]")

_PLAN = re.compile(
    r"(?:^|\n)\s*(?:here(?:'s| is)\s+(?:the|my)\s+plan|plan\s*[::]|"
    r"i\s+will\s+(?:first|now)\b|step\s*1\s*[::.]|to\s+solve\s+this,?\s+we\s+(?:will|need))",
    re.IGNORECASE,
)
_TOOL_OUTPUT = re.compile(
    r"(?:^|\n)\s*(?:exitcode\s*[::]|Traceback \(most recent call last\)|"
    r"```(?:python|bash|sh|output)|Address:\s*http|Viewport position|"
    r"Here is a screenshot|The web browser is open|File .*? line \d+)",
    re.IGNORECASE,
)

#: Below this an ``execute`` step carries no assessable evidence of its own; the
#: judge's evidence policy compensates (Part C §3). Same constant defines both.
NEAR_EMPTY_CHARS = 16


@dataclass(frozen=True, slots=True)
class Verdict:
    type_norm: TypeNorm
    rule: str

    @property
    def covered(self) -> bool:
        """Did a positive rule fire, as opposed to the residual fallback?"""
        return self.rule not in ("fallback", "empty_content")


def _plan_json(content: str) -> bool:
    text = content.strip()
    lo, hi = text.find("{"), text.rfind("}")
    if lo == -1 or hi <= lo:
        return False
    blob = text[lo : hi + 1]
    try:
        obj = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        # Plans are sometimes embedded as a JSON *string*, so the quotes arrive
        # escaped; unescape before sniffing for the assignment keys.
        unescaped = blob.replace('\\"', '"')
        return any(f'"{k}"' in unescaped for k in PLAN_JSON_KEYS)
    return isinstance(obj, dict) and bool(PLAN_JSON_KEYS & {k.lower() for k in obj})


def is_answer_emission(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    if _ANSWER.search(text) or _TERMINATE.search(text):
        return True
    # A step that is nothing but an option letter is an answer emission — and is
    # exactly the near-empty case the evidence policy has to rescue.
    return bool(_BARE_CHOICE.match(text))


def classify(content: str, agent: str, *, is_last: bool = False, role_raw: str = "") -> Verdict:
    """Rule-based type for one AG step. ``unknown`` is a permitted outcome."""
    text = (content or "").strip()
    role = (role_raw or "").strip().lower()
    if not text:
        return Verdict("unknown", "empty_content")

    # Final-answer emission first: a coordinator's closing message is an answer,
    # not a plan, and ordering the other way silently retypes every trajectory's
    # last step.
    if is_answer_emission(text) and (is_last or _TERMINATE.search(text)):
        return Verdict("final", "answer_emission_last")
    if _plan_json(text):
        return Verdict("plan", "plan_json")
    if _ARROW_IN_TEXT.search(text) and is_orchestrator(agent):
        return Verdict("delegate", "orchestrator_arrow")
    if _DELEGATE.search(text):
        return Verdict("delegate", "handoff_language")
    if is_orchestrator(agent) and _ADDRESSED.match(text):
        return Verdict("delegate", "addressed_agent")
    if _TOOL_OUTPUT.search(text) or role in {"tool", "function", "tool_call", "observation"}:
        return Verdict("execute", "tool_output")
    if _PLAN.search(text):
        return Verdict("plan", "plan_language")
    if is_orchestrator(agent) and not is_last:
        return Verdict("plan", "orchestrator_default")
    if is_answer_emission(text):
        return Verdict("final", "answer_emission")
    return Verdict("execute", "fallback")


def classify_steps(steps: tuple[Step, ...] | list[Step]) -> list[Verdict]:
    n = len(steps)
    return [
        classify(s.content, s.agent, is_last=(i == n - 1), role_raw=s.role_raw)
        for i, s in enumerate(steps)
    ]


def apply_rules(steps: tuple[Step, ...] | list[Step]) -> tuple[list[Step], list[Verdict]]:
    """Type every step by rule. Returns new steps and the verdicts behind them.

    Only steps still marked ``classified`` are rewritten: parsed types are read
    from the data and the rules never override them.
    """
    verdicts = classify_steps(steps)
    out = [
        s.typed(v.type_norm, "classified") if s.type_source == "classified" else s
        for s, v in zip(steps, verdicts)
    ]
    return out, verdicts


def coverage(verdicts: list[Verdict]) -> float:
    return sum(v.covered for v in verdicts) / len(verdicts) if verdicts else 0.0


def rule_table_text() -> str:
    """Canonical serialisation of the rule table, for the frozen spec hash."""
    parts = [
        f"NEAR_EMPTY_CHARS={NEAR_EMPTY_CHARS}",
        "PLAN_JSON_KEYS=" + ",".join(sorted(PLAN_JSON_KEYS)),
        "HC_WORKERS=" + ",".join(sorted(_HC_WORKERS)),
        "QUALIFIER=" + ",".join(f"{k}->{v}" for k, v in sorted(_QUALIFIER.items())),
    ]
    for name, pat in [
        ("ANSWER", _ANSWER), ("TERMINATE", _TERMINATE), ("BARE_CHOICE", _BARE_CHOICE),
        ("DELEGATE", _DELEGATE), ("ARROW", _ARROW_IN_TEXT), ("ADDRESSED", _ADDRESSED),
        ("PLAN", _PLAN), ("TOOL_OUTPUT", _TOOL_OUTPUT), ("COMPOUND", _COMPOUND),
    ]:
        parts.append(f"{name}={pat.pattern}")
    return "\n".join(parts)
