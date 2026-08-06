"""Who&When Hand-Crafted loader (subset ``hc``).

Steps are ``{content, role}`` where the role is *compound* and encodes both the
agent and the act — ``Orchestrator (thought)``, ``Orchestrator (-> WebSurfer)``,
``WebSurfer``. Types are therefore ``parsed``, which makes HC the reference
corpus for the rule-validation gate.

Trajectories here reach ~130 steps, which is what forces KV prefix sharing in
the judge.
"""

from __future__ import annotations

from pathlib import Path

from ..record import Record, Step
from ..typing.normalize import apply_rules, is_answer_emission, parse_hc_role
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

SUBSET = "hc"


def load_file(path: Path, report: LoadReport) -> Record:
    blob = read_json(path)
    where = f"{SUBSET}/{path.stem}"

    steps = []
    for i, raw in enumerate(history(blob, where)):
        role = as_text(require(raw, "role", f"{where} step {i}", "name"))
        agent, type_norm, target = parse_hc_role(role)
        if target:
            report.bump("delegate_parsed")
        if type_norm == "unknown":
            report.bump("role_unparsed")
            report.note(f"{path.stem} step {i}: unparsed compound role {role!r}")
        steps.append(
            Step(
                idx=i,
                agent=agent,
                role_raw=role,
                content=as_text(require(raw, "content", f"{where} step {i}", "output", "message")),
                type_norm=type_norm,
                type_source="parsed" if type_norm != "unknown" else "classified",
            )
        )

    # The trajectory's last step is `final` when it actually emits an answer;
    # otherwise the parsed type stands.
    if steps and is_answer_emission(steps[-1].content):
        steps[-1] = steps[-1].typed("final", "parsed")
        report.bump("final_promoted")

    typed, _ = apply_rules(steps)  # fills only the steps still `classified`

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
