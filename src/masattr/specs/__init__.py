"""Frozen artifacts (spec v2 Part B rule 6).

Prompts, the type-rule table, and the paper-1 type map are serialised here and
hashed. Every run's manifest logs those hashes, so a run is reproducible from
``(commit, manifest)`` alone — and a silent edit to a prompt shows up as a hash
mismatch instead of an unexplained number change.

``freeze()`` writes the artifacts; ``verify()`` checks the live code still
matches them and is called at the start of every experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPEC_DIR = Path(__file__).parent
PROMPTS_FILE = SPEC_DIR / "prompts.txt"
TYPE_RULES_FILE = SPEC_DIR / "type_rules.txt"
TYPE_MAP_FILE = SPEC_DIR / "paper1_type_map.json"
HASHES_FILE = SPEC_DIR / "hashes.json"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def live_artifacts() -> dict[str, str]:
    """The artifacts as the code currently defines them."""
    from ..calib.fit import DEFAULT_TYPE_MAP, load_type_map
    from ..judge.prompts import prompt_text
    from ..typing.normalize import rule_table_text

    type_map = load_type_map(TYPE_MAP_FILE) if TYPE_MAP_FILE.exists() else dict(DEFAULT_TYPE_MAP)
    return {
        "prompts": prompt_text(),
        "type_rules": rule_table_text(),
        "paper1_type_map": json.dumps(type_map, indent=2, sort_keys=True),
    }


def hashes() -> dict[str, str]:
    return {k: sha(v) for k, v in live_artifacts().items()}


def freeze() -> dict[str, str]:
    """Write the artifacts and their hashes to ``specs/``."""
    arts = live_artifacts()
    PROMPTS_FILE.write_text(arts["prompts"], encoding="utf-8")
    TYPE_RULES_FILE.write_text(arts["type_rules"], encoding="utf-8")
    TYPE_MAP_FILE.write_text(arts["paper1_type_map"], encoding="utf-8")
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
