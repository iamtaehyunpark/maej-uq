"""Attribution track — first-crossing on Who&When (spec §5, §6, §8).

Primary: exact-match agent-accuracy and step-accuracy, per subset, with
bootstrap CIs over files. Reported four ways because the spec pre-registers all
four: primary scorer *and* the published substring scorer; with *and* without
the ``agent_step_mismatch``-flagged files.

Ablations: argmin, changepoint, agent-first. The step-first vs agent-first
disagreement is reported stratified by predicted step type and, for HC, by
orchestrator vs worker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..attribution import (
    METHODS,
    PRIMARY,
    attribute,
    disagreement,
    orchestrator_strata,
    render_disagreement,
    type_strata,
)
from ..metrics import AttributionScore, score_attribution
from ..schema import FLAG_AGENT_STEP_MISMATCH, Record


@dataclass
class AttributionResult:
    subset: str
    n_files: int
    n_flagged: int
    scores: dict[str, dict] = field(default_factory=dict)  # method → variant → score
    disagreement_type: list = field(default_factory=list)
    disagreement_role: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "subset": self.subset,
            "n_files": self.n_files,
            "n_flagged_agent_step_mismatch": self.n_flagged,
            "primary_method": PRIMARY,
            "scores": self.scores,
            "disagreement_by_type": [
                {"stratum": r.stratum, "n": r.n, "step": r.step_rate, "agent": r.agent_rate}
                for r in self.disagreement_type
            ],
            "disagreement_by_role": [
                {"stratum": r.stratum, "n": r.n, "step": r.step_rate, "agent": r.agent_rate}
                for r in self.disagreement_role
            ],
        }

    def render(self) -> str:
        lines = [
            f"## Attribution — {self.subset}",
            "",
            f"n={self.n_files} files ({self.n_flagged} flagged agent_step_mismatch)",
            "",
            "| method | scorer | files | agent acc | step acc | both |",
            "|---|---|---|---|---|---|",
        ]
        for method, variants in self.scores.items():
            mark = " **(primary)**" if method == PRIMARY else ""
            for variant, s in variants.items():
                a_ci = s.get("agent_ci")
                s_ci = s.get("step_ci")
                a = f"{s['agent_acc']:.3f}"
                st = f"{s['step_acc']:.3f}"
                if a_ci:
                    a += f" [{a_ci['lo']:.3f}, {a_ci['hi']:.3f}]"
                if s_ci:
                    st += f" [{s_ci['lo']:.3f}, {s_ci['hi']:.3f}]"
                lines.append(
                    f"| {method}{mark} | {variant} | {s['n']} | {a} | {st} | {s['both_acc']:.3f} |"
                )
        lines += [
            "",
            "> The substring scorer reproduces the published-number regime and carries "
            'its artifact with it: `"1" in "12"` scores as a hit. Exact match is primary.',
            "",
            render_disagreement(self.disagreement_type, "(by predicted step type)"),
        ]
        if self.disagreement_role:
            lines += ["", render_disagreement(self.disagreement_role, "(orchestrator vs worker)")]
        return "\n".join(lines)


def _gold(records: Sequence[Record]) -> dict[str, tuple[str | None, int | None]]:
    return {r.key: (r.label_mistake_agent, r.label_mistake_step) for r in records}


def run_subset(
    records: Sequence[Record],
    scores_by_key: dict[str, list],
    *,
    subset: str,
    threshold: float,
    use_calibrated: bool = True,
    n_boot: int = 2000,
    seed: int = 0,
    include_hc_roles: bool = False,
) -> AttributionResult:
    gold = _gold(records)
    flagged = {r.key for r in records if FLAG_AGENT_STEP_MISMATCH in r.flags}
    grouped = {k: v for k, v in scores_by_key.items() if k in gold and v}

    all_preds: dict[str, dict] = {}
    scores: dict[str, dict] = {}

    for method in METHODS:
        preds = attribute(
            grouped, threshold=threshold, method=method, use_calibrated=use_calibrated
        )
        all_preds[method] = preds
        variants: dict[str, dict] = {}
        for scorer in ("exact", "substring"):
            for label, keys in (
                ("all", set(preds)),
                ("excl_flagged", set(preds) - flagged),
            ):
                pairs = [(preds[k].as_pair(), gold[k]) for k in sorted(keys)]
                if not pairs:
                    continue
                s: AttributionScore = score_attribution(
                    pairs, scorer=scorer, n_boot=n_boot, seed=seed
                )
                variants[f"{scorer}/{label}"] = s.to_dict()
        scores[method] = variants

    step_first = all_preds[PRIMARY]
    agent_first = all_preds["agent_first"]
    dis_type = disagreement(
        step_first, agent_first, strata=type_strata(grouped, step_first)
    )
    dis_role = (
        disagreement(step_first, agent_first, strata=orchestrator_strata(grouped, step_first))
        if include_hc_roles
        else []
    )

    return AttributionResult(
        subset=subset,
        n_files=len(grouped),
        n_flagged=len(flagged & set(grouped)),
        scores=scores,
        disagreement_type=dis_type,
        disagreement_role=dis_role,
    )


def run(
    subsets: dict[str, tuple[Sequence[Record], dict[str, list]]],
    *,
    threshold: float,
    use_calibrated: bool = True,
    out_dir: str | Path | None = None,
    n_boot: int = 2000,
) -> dict[str, AttributionResult]:
    results = {
        name: run_subset(
            recs,
            scores,
            subset=name,
            threshold=threshold,
            use_calibrated=use_calibrated,
            n_boot=n_boot,
            include_hc_roles=(name == "hc"),
        )
        for name, (recs, scores) in subsets.items()
    }
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "attribution.json").write_text(
            json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2), encoding="utf-8"
        )
        (out / "attribution.md").write_text(
            "\n\n".join(r.render() for r in results.values()) + "\n", encoding="utf-8"
        )
    return results
