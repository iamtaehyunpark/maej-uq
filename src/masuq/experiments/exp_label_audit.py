"""MATU label spot-audit (spec §7.6) — provenance mitigation.

The MATU per-run correctness labels are inherited, not produced here, and every
calibration map and trajectory AUROC in the pilot rests on them. So we re-label
a deterministic sample of 100 runs with a 3-judge pipeline and report agreement
with the shipped labels. This does not correct the labels; it bounds how much
they can be trusted, which is the only honest thing to do with a label set you
did not create.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..judge.backends import TextGenerator
from ..schema import Record

MAX_STEP_CHARS = 1200


@dataclass
class AuditRow:
    key: str
    shipped: bool
    votes: list[bool] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)

    @property
    def majority(self) -> bool:
        return sum(self.votes) > len(self.votes) / 2

    @property
    def unanimous(self) -> bool:
        return len(set(self.votes)) == 1

    @property
    def agrees(self) -> bool:
        return self.majority == self.shipped

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "shipped": self.shipped,
            "votes": self.votes,
            "majority": self.majority,
            "unanimous": self.unanimous,
            "agrees": self.agrees,
        }


@dataclass
class AuditReport:
    subset: str
    n: int
    agreement: float
    unanimity: float
    n_shipped_true: int
    n_audit_true: int
    rows: list[AuditRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "subset": self.subset,
            "n": self.n,
            "agreement_with_shipped_labels": self.agreement,
            "judge_unanimity": self.unanimity,
            "n_shipped_true": self.n_shipped_true,
            "n_audit_true": self.n_audit_true,
            "rows": [r.to_dict() for r in self.rows],
        }

    def render(self) -> str:
        return "\n".join(
            [
                f"## MATU label spot-audit — {self.subset}",
                "",
                f"n={self.n} sampled runs, 3-judge majority vote",
                "",
                "| quantity | value |",
                "|---|---|",
                f"| agreement with shipped labels | {self.agreement:.3f} |",
                f"| judge unanimity | {self.unanimity:.3f} |",
                f"| shipped label positive rate | {self.n_shipped_true / self.n:.3f} |",
                f"| audit positive rate | {self.n_audit_true / self.n:.3f} |",
                "",
                "> This bounds label trust; it does not replace the labels. Every "
                "calibration map and trajectory metric in this pilot inherits whatever "
                "error remains here.",
            ]
        )


def _prompt(record: Record) -> str:
    body = "\n".join(
        f"[step {s.idx}] {s.agent}: {(s.content or '')[:MAX_STEP_CHARS]}" for s in record.steps
    )
    return (
        "Judge whether this multi-agent run solved its task correctly.\n\n"
        f"Task: {record.query or '(not recorded in this subset)'}\n"
        f"Reference answer: {record.ground_truth or '(not recorded)'}\n\n"
        f"Transcript:\n{body}\n\n"
        "Did the run reach the correct final answer? Answer Yes or No.\n"
    )


def sample_runs(records: Sequence[Record], n: int = 100, seed: int = 0) -> list[Record]:
    labelled = [r for r in records if r.label_correct is not None]
    rng = random.Random(seed)
    idx = list(range(len(labelled)))
    rng.shuffle(idx)
    return [labelled[i] for i in idx[:n]]


def run(
    records: Sequence[Record],
    judges: Sequence[TextGenerator],
    *,
    subset: str,
    n: int = 100,
    seed: int = 0,
    out_dir: str | Path | None = None,
) -> AuditReport:
    if len(judges) != 3:
        raise ValueError(f"spec §7.6 specifies a 3-judge pipeline, got {len(judges)}")
    sample = sample_runs(records, n=n, seed=seed)
    rows: list[AuditRow] = []
    for rec in sample:
        prompt = _prompt(rec)
        votes, raw = [], []
        for j in judges:
            reply = j.generate(prompt, max_new_tokens=8) or ""
            raw.append(reply)
            votes.append(reply.strip().lower().startswith("yes"))
        rows.append(AuditRow(key=rec.key, shipped=bool(rec.label_correct), votes=votes, raw=raw))

    report = AuditReport(
        subset=subset,
        n=len(rows),
        agreement=sum(r.agrees for r in rows) / max(len(rows), 1),
        unanimity=sum(r.unanimous for r in rows) / max(len(rows), 1),
        n_shipped_true=sum(r.shipped for r in rows),
        n_audit_true=sum(r.majority for r in rows),
        rows=rows,
    )
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"label_audit_{subset}.json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        (out / f"label_audit_{subset}.md").write_text(report.render() + "\n", encoding="utf-8")
    return report
