"""Frozen artifacts (spec v2 Part B rule 6, v2.1 §1.5).

Prompts, the type-rule table, the E0 decision criteria, and the judge identities
are serialised here and hashed. Every run's manifest logs those hashes, so a run
is reproducible from ``(commit, manifest)`` alone — and a silent edit to a
prompt or a decision bound shows up as a hash mismatch instead of an unexplained
number change.

``freeze()`` writes the artifacts; ``verify()`` checks the live code still
matches them and is called at the start of every experiment.

The two owner-set files carry a ``status`` that the runs check: ``e0_criteria``
must be ``registered`` before E0 runs, and ``judge`` must be ``confirmed``
before any reported run. Both start as drafts on purpose.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SPEC_DIR = Path(__file__).parent
PROMPTS_FILE = SPEC_DIR / "prompts.txt"
TYPE_RULES_FILE = SPEC_DIR / "type_rules.txt"
E0_CRITERIA_FILE = SPEC_DIR / "e0_criteria.json"
JUDGE_FILE = SPEC_DIR / "judge.json"
HASHES_FILE = SPEC_DIR / "hashes.json"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def e0_criteria() -> dict[str, Any]:
    return _read_json(E0_CRITERIA_FILE)


def judge_spec() -> dict[str, Any]:
    return _read_json(JUDGE_FILE)


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
        "e0_criteria": json.dumps(e0_criteria(), indent=2, sort_keys=True),
        "judge": json.dumps(judge_spec(), indent=2, sort_keys=True),
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
    if not HASHES_FILE.exists():
        msg = f"no frozen hashes at {HASHES_FILE}; run `masattr freeze` before any experiment"
        if strict:
            raise RuntimeError(msg)
        return [msg]
    frozen = json.loads(HASHES_FILE.read_text(encoding="utf-8"))
    live = hashes()
    drift = [
        f"{k}: frozen {frozen.get(k)} != live {v}" for k, v in live.items() if frozen.get(k) != v
    ]
    if drift and strict:
        raise RuntimeError(
            "frozen artifacts drifted from the code: "
            + "; ".join(drift)
            + " — re-freeze deliberately (`masattr freeze`) and note it in the run log"
        )
    return drift
