"""Scoring (spec v3 Part C §6, Part E).

Primary: **exact match** on agent and step. Comparability row: the Who&When
substring scorer, re-implemented here from the four lines in their
``evaluate.py`` — their eval path is not imported, so the artifact is
reproduced deliberately and labelled, never inherited by accident.

The artifact: ``str(pred) in str(gold)`` makes a predicted step ``1`` a hit
against gold ``12``. That regime is what the published numbers were computed in,
so it is reported as a comparability row and never as primary.

Every table is dual-reported with and without the 6 ``agent_step_mismatch``
files and, in Exp-0's aftermath, with and without the held-aside 20. CIs
bootstrap over files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..record import FLAG_AGENT_STEP_MISMATCH, Record
from ..typing.normalize import collapse_orchestrator
from .ci import CI, bootstrap_ci

Pair = tuple[tuple[str | None, int | None], tuple[str | None, int | None]]


def exact_agent(pred: str | None, gold: str | None) -> bool:
    """Exact match after orchestrator-name collapse.

    The collapse is not leniency: the annotations spell one agent several ways,
    and without it the scorer would measure spelling rather than attribution.
    """
    if pred is None or gold is None:
        return False
    return collapse_orchestrator(pred) == collapse_orchestrator(gold)


def exact_step(pred: int | None, gold: int | None) -> bool:
    if pred is None or gold is None:
        return False
    return int(pred) == int(gold)


def substring_agent(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    p, g = pred.strip().lower(), gold.strip().lower()
    return p in g or g in p


def substring_step(pred: int | None, gold: int | None) -> bool:
    if pred is None or gold is None:
        return False
    return str(pred) in str(gold) or str(gold) in str(pred)


SCORERS = {
    "exact": (exact_agent, exact_step),
    "substring": (substring_agent, substring_step),
}


@dataclass(slots=True)
class Score:
    scorer: str
    slice_name: str
    n: int
    agent_acc: float
    step_acc: float
    both_acc: float
    agent_ci: CI | None = None
    step_ci: CI | None = None

    def to_dict(self) -> dict:
        return {
            "scorer": self.scorer,
            "slice": self.slice_name,
            "n": self.n,
            "agent_acc": self.agent_acc,
            "step_acc": self.step_acc,
            "both_acc": self.both_acc,
            "agent_ci": self.agent_ci.to_dict() if self.agent_ci else None,
            "step_ci": self.step_ci.to_dict() if self.step_ci else None,
        }


def score_pairs(
    pairs: Sequence[Pair],
    *,
    scorer: str = "exact",
    slice_name: str = "all",
    n_boot: int = 2000,
    seed: int = 0,
) -> Score:
    agent_fn, step_fn = SCORERS[scorer]
    units = [(agent_fn(p[0], g[0]), step_fn(p[1], g[1])) for p, g in pairs]
    if not units:
        return Score(scorer, slice_name, 0, float("nan"), float("nan"), float("nan"))
    n = len(units)
    return Score(
        scorer=scorer,
        slice_name=slice_name,
        n=n,
        agent_acc=sum(a for a, _ in units) / n,
        step_acc=sum(s for _, s in units) / n,
        both_acc=sum(a and s for a, s in units) / n,
        agent_ci=bootstrap_ci(units, lambda u: sum(a for a, _ in u) / len(u), n_boot=n_boot, seed=seed),
        step_ci=bootstrap_ci(units, lambda u: sum(s for _, s in u) / len(u), n_boot=n_boot, seed=seed + 1),
    )


def slices(records: Sequence[Record], held_aside: Iterable[str] = ()) -> dict[str, set[str]]:
    """The pre-registered dual-reporting slices (Part E).

    Three exclusion axes, kept separate because they are different objections:
    the 6 ``agent_step_mismatch`` files (the annotation names an agent that does
    not act at the annotated step), the 5 record-level anomalies the release
    contains (out-of-range ``mistake_step``, empty-content step), and the 20
    held-aside files E0 checked transfer on.
    """
    keys = {r.key for r in records}
    flagged = {r.key for r in records if FLAG_AGENT_STEP_MISMATCH in r.flags}
    anomalous = {r.key for r in records if r.is_anomalous}
    held = set(held_aside) & keys
    out = {"all": keys, "excl_flagged": keys - flagged}
    if anomalous:
        out["excl_anomalous"] = keys - anomalous
    if held:
        out["excl_held_aside"] = keys - held
    if anomalous or held:
        out["excl_all_excluded"] = keys - flagged - anomalous - held
    return out


def score_all(
    preds: Mapping[str, object],
    gold: Mapping[str, tuple[str, int]],
    records: Sequence[Record],
    *,
    held_aside: Iterable[str] = (),
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, dict]:
    """Every scorer × every slice, for one attribution method."""
    out: dict[str, dict] = {}
    for slice_name, keys in slices(records, held_aside).items():
        usable = sorted(k for k in keys if k in preds and k in gold)
        pairs = [(preds[k].as_pair(), gold[k]) for k in usable]  # type: ignore[union-attr]
        for scorer in ("exact", "substring"):
            s = score_pairs(
                pairs, scorer=scorer, slice_name=slice_name, n_boot=n_boot, seed=seed
            )
            out[f"{scorer}/{slice_name}"] = s.to_dict()
    return out


def gold_map(records: Sequence[Record]) -> dict[str, tuple[str, int]]:
    return {r.key: r.gold for r in records}


def render(rows: Mapping[str, Mapping[str, dict]], title: str = "") -> str:
    """``{method: {scorer/slice: score_dict}}`` → a markdown table."""
    lines = [
        f"## Attribution {title}".rstrip(),
        "",
        "| method | scorer | slice | n | agent acc | step acc | both |",
        "|---|---|---|---|---|---|---|",
    ]
    for method, variants in rows.items():
        for name, s in variants.items():
            scorer, _, slice_name = name.partition("/")
            a, st = f"{s['agent_acc']:.3f}", f"{s['step_acc']:.3f}"
            if s.get("agent_ci"):
                a += f" [{s['agent_ci']['lo']:.3f}, {s['agent_ci']['hi']:.3f}]"
            if s.get("step_ci"):
                st += f" [{s['step_ci']['lo']:.3f}, {s['step_ci']['hi']:.3f}]"
            lines.append(
                f"| {method} | {scorer} | {slice_name} | {s['n']} | {a} | {st} | "
                f"{s['both_acc']:.3f} |"
            )
    lines += [
        "",
        "> Exact match is primary. The substring row reproduces the published-number "
        'regime and carries its artifact: predicted step `1` scores as a hit against '
        "gold `12`.",
    ]
    return "\n".join(lines)
