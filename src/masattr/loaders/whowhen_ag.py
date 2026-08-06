"""Who&When Algorithm-Generated loader (subset ``alg``).

Steps are ``{content, role, name}``; ``agent := name``. No act type is present
or parseable, so every step arrives ``classified`` and the rules of
:mod:`masattr.typing.normalize` supply the type.
"""

from __future__ import annotations

from pathlib import Path

from ..record import Record, Step
from ..typing.normalize import apply_rules
from ._common import (
    LoadReport,
    _QUERY_KEYS,
    _TRUTH_KEYS,
    as_text,
    cast_step,
    finish,
    flag_mismatch,
    history,
    json_files,
    pick,
    read_json,
    require,
)

SUBSET = "alg"


def load_file(path: Path, report: LoadReport) -> Record:
    blob = read_json(path)
    where = f"{SUBSET}/{path.stem}"

    steps = []
    for i, raw in enumerate(history(blob, where)):
        role = as_text(require(raw, "role", f"{where} step {i}", "name"))
        name = raw.get("name")
        agent = as_text(name).strip() if name else role.strip()
        if not name:
            report.bump("missing_name_fallback_role")
        steps.append(
            Step(
                idx=i,
                agent=agent,
                role_raw=role,
                content=as_text(require(raw, "content", f"{where} step {i}", "output", "message")),
                type_norm="unknown",
                type_source="classified",
            )
        )

    typed, verdicts = apply_rules(steps)
    for v in verdicts:
        report.bump(f"rule_{v.rule}")

    rec = Record(
        subset=SUBSET,
        file_id=path.stem,
        query=as_text(pick(blob, _QUERY_KEYS)),
        ground_truth=as_text(pick(blob, _TRUTH_KEYS)),
        steps=tuple(typed),
        label_mistake_agent=as_text(pick(blob, ("mistake_agent", "agent"))),
        label_mistake_step=cast_step(blob.get("mistake_step", blob.get("step")), where),
        label_mistake_reason=as_text(pick(blob, ("mistake_reason", "reason"))),
        source_file=str(path),
    ).validate()
    return flag_mismatch(rec, report)


def load(root: str | Path) -> tuple[list[Record], LoadReport]:
    root = Path(root)
    report = LoadReport(subset=SUBSET, source=str(root))
    records = [load_file(p, report) for p in json_files(root)]
    return records, finish(records, report)
