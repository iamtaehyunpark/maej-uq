"""Model identity and family-disjointness checks (spec v3 Part C §3).

Two disjointness constraints, both enforced here rather than promised in prose:

* the judge must be family-disjoint from any labeling judge;
* the type-classifier must be family-disjoint from the judge — typing feeds the
  judge's evidence policy, so drawing both from one family closes a loop a
  reviewer will find.

``family_of`` is deliberately a lookup over substrings, not a taxonomy. It
returns ``unknown`` for anything it does not recognise, and an ``unknown``
family never counts as disjoint — an unrecognised model is a question, not a
pass.
"""

from __future__ import annotations

FAMILIES: dict[str, tuple[str, ...]] = {
    "qwen": ("qwen",),
    "llama": ("llama", "meta-llama"),
    "mistral": ("mistral", "mixtral"),
    "gemma": ("gemma",),
    "phi": ("phi-", "phi3", "phi4", "phi_"),
    "gpt": ("gpt-", "gpt3", "gpt4", "o1-", "o3-", "davinci"),
    "claude": ("claude",),
    "deepseek": ("deepseek",),
    "gemini": ("gemini", "palm"),
    "olmo": ("olmo",),
    "falcon": ("falcon",),
    "yi": ("yi-",),
    "mock": ("mock",),
}


class DisjointnessError(RuntimeError):
    """Two roles that must come from different model families do not."""


def family_of(model_id: str) -> str:
    """Best-effort model family for a spec like ``hf:Qwen/Qwen3.6-35B-A3B``."""
    if not model_id:
        return "unknown"
    text = model_id.lower()
    if ":" in text:
        text = text.split(":", 1)[1] or text
    text = text.rsplit("/", 1)[-1]
    for family, needles in FAMILIES.items():
        if any(n in text for n in needles):
            return family
    return "unknown"


def check_disjoint(role_a: str, model_a: str, role_b: str, model_b: str, *, strict: bool = True) -> str | None:
    """Return a violation message, or ``None`` when the two are disjoint."""
    fa, fb = family_of(model_a), family_of(model_b)
    problem: str | None = None
    if fa == "unknown" or fb == "unknown":
        problem = (
            f"cannot verify {role_a} ({model_a!r} → family {fa}) is disjoint from "
            f"{role_b} ({model_b!r} → family {fb}); add the family to "
            "masattr.models.FAMILIES rather than assuming"
        )
    elif fa == fb:
        problem = (
            f"{role_a} ({model_a!r}) and {role_b} ({model_b!r}) are both family "
            f"{fa!r}; spec v3 Part C §3 requires them disjoint"
        )
    if problem and strict:
        raise DisjointnessError(problem)
    return problem


def families(**roles: str) -> dict[str, str]:
    """``{role: family}`` for the run manifest, so disjointness is auditable."""
    return {role: family_of(model or "") for role, model in roles.items()}
