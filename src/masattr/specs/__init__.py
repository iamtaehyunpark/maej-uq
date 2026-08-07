"""Frozen artifacts (spec v3 Part B.5).

Prompts, the type-rule table, the registered criteria, the judge identities, and
the directive that fixes the primary attribution rule are serialised here and
hashed. Every run's manifest logs those hashes, so a run
is reproducible from ``(commit, manifest)`` alone — and a silent edit to a
prompt or a decision bound shows up as a hash mismatch instead of an unexplained
number change.

``freeze()`` writes the artifacts; ``verify()`` checks the live code still
matches them and is called at the start of every experiment.

Two owner-set files carry a ``status`` the runs check: ``criteria`` must be
``registered`` before any attribution number is computed — the changepoint
fallback condition is part of the primary rule's definition — and ``judge`` must
be ``confirmed`` before any reported run. Both start as drafts on purpose.

``rule_directive.md`` is the provenance of the primary rule itself. Its hash is
logged on every run, so a number can always be traced to the directive that
fixed the rule that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SPEC_DIR = Path(__file__).parent
PROMPTS_FILE = SPEC_DIR / "prompts.txt"
TYPE_RULES_FILE = SPEC_DIR / "type_rules.txt"
CRITERIA_FILE = SPEC_DIR / "criteria.json"
JUDGE_FILE = SPEC_DIR / "judge.json"
RULE_DIRECTIVE_FILE = SPEC_DIR / "rule_directive.md"
HASHES_FILE = SPEC_DIR / "hashes.json"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def criteria() -> dict[str, Any]:
    return _read_json(CRITERIA_FILE)


def rule_directive() -> str:
    return (
        RULE_DIRECTIVE_FILE.read_text(encoding="utf-8")
        if RULE_DIRECTIVE_FILE.exists()
        else ""
    )


def rule_provenance() -> str:
    """Hash of the directive that fixes the primary rule."""
    return sha(rule_directive())


#: Roles that must be family-disjoint from each other, and the reason.
DISJOINT_PAIRS = (
    ("type_classifier", "judge_primary"),
    ("type_classifier", "judge_secondary"),
    ("judge_primary", "judge_secondary"),
)

ROLES = ("judge_primary", "judge_secondary", "type_classifier")


def judge_spec() -> dict[str, Any]:
    return _read_json(JUDGE_FILE)


def role(name: str) -> dict[str, Any]:
    """One role's ``{id, family, status}``, or empty if undeclared."""
    entry = judge_spec().get(name)
    return dict(entry) if isinstance(entry, dict) else {}


def role_id(name: str) -> str:
    return str(role(name).get("id", ""))


def client_spec(name: str) -> str:
    """The role's id as a client spec. Bare HF ids are prefixed."""
    ident = role_id(name)
    if not ident:
        raise RuntimeError(f"specs/judge.json declares no id for role {name!r}")
    return ident if ":" in ident else f"hf:{ident}"


def require_role(name: str) -> dict[str, Any]:
    """Refuse to use a role that has not been confirmed.

    Status is per role on purpose: the secondary judge and the type classifier
    are decided independently of the primary, and a run should be blocked only
    by the roles it actually uses.
    """
    entry = role(name)
    if not entry:
        raise RuntimeError(f"specs/judge.json declares no role {name!r}; known: {ROLES}")
    status = str(entry.get("status", "draft")).lower()
    if status != "confirmed":
        raise RuntimeError(
            f"role {name!r} ({entry.get('id')!r}) is marked {status!r}, not "
            "'confirmed'. Set its status in specs/judge.json once the checkpoint "
            "is decided; a reported number should not rest on a draft identity."
        )
    return entry


def check_families(*, strict: bool = True) -> list[str]:
    """Verify declared families against the resolver, then the disjointness pairs.

    The declared ``family`` is a claim, not evidence. If it disagrees with what
    the id actually resolves to, every disjointness check downstream is being
    made against a fiction.
    """
    from ..models import check_disjoint, family_of

    problems: list[str] = []
    for name in ROLES:
        entry = role(name)
        ident = entry.get("id")
        if not ident:
            continue
        declared = str(entry.get("family", "")).lower()
        resolved = family_of(str(ident))
        if declared and declared != resolved:
            problems.append(
                f"{name}: declares family {declared!r} but {ident!r} resolves to "
                f"{resolved!r}"
            )
    for a, b in DISJOINT_PAIRS:
        ia, ib = role_id(a), role_id(b)
        if ia and ib:
            problem = check_disjoint(a, ia, b, ib, strict=False)
            if problem:
                problems.append(problem)
    if problems and strict:
        raise RuntimeError("specs/judge.json: " + "; ".join(problems))
    return problems


def require_status(name: str, blob: dict[str, Any], wanted: str, why: str) -> None:
    """Refuse to proceed while an owner-set artifact is still a draft."""
    status = str(blob.get("status", "draft")).lower()
    if status != wanted:
        raise RuntimeError(
            f"specs/{name}.json is marked {status!r}, not {wanted!r}. {why} "
            f"Set \"status\": \"{wanted}\" once the values have been decided."
        )


def live_artifacts() -> dict[str, str]:
    """The artifacts as the code and the spec files currently define them."""
    from ..judge.prompts import prompt_text
    from ..typing.normalize import rule_table_text

    return {
        "prompts": prompt_text(),
        "type_rules": rule_table_text(),
        "criteria": json.dumps(criteria(), indent=2, sort_keys=True),
        "judge": json.dumps(judge_spec(), indent=2, sort_keys=True),
        "rule_directive": rule_directive(),
    }


def hashes() -> dict[str, str]:
    return {k: sha(v) for k, v in live_artifacts().items()}


def freeze() -> dict[str, str]:
    """Write the code-derived artifacts and hash everything under ``specs/``."""
    arts = live_artifacts()
    PROMPTS_FILE.write_text(arts["prompts"], encoding="utf-8")
    TYPE_RULES_FILE.write_text(arts["type_rules"], encoding="utf-8")
    h = {k: sha(v) for k, v in arts.items()}
    HASHES_FILE.write_text(json.dumps(h, indent=2), encoding="utf-8")
    return h


def verify(*, strict: bool = True) -> list[str]:
    """Compare live artifacts to the frozen ones. Returns the list of drifts."""
    problems = check_families(strict=strict)
    if not HASHES_FILE.exists():
        msg = f"no frozen hashes at {HASHES_FILE}; run `masattr freeze` before any experiment"
        if strict:
            raise RuntimeError(msg)
        return problems + [msg]
    frozen = json.loads(HASHES_FILE.read_text(encoding="utf-8"))
    live = hashes()
    drift = problems + [
        f"{k}: frozen {frozen.get(k)} != live {v}" for k, v in live.items() if frozen.get(k) != v
    ]
    if drift and strict:
        raise RuntimeError(
            "frozen artifacts drifted from the code: "
            + "; ".join(drift)
            + " — re-freeze deliberately (`masattr freeze`) and note it in the run log"
        )
    return drift
