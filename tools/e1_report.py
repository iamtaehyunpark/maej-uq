"""Regenerate the E0+E1 reference report from current artifacts.

The hand-written version of this report went stale the moment the readout
scaffold changed, and a stale table with a plausible shape is worse than no
table. Everything here is read from ``runs/main`` — the same artifacts the
master table reads — so the document cannot disagree with the run that produced
it.

E0 is sanity-only and gates nothing; the primary rule is fixed by
``specs/rule_directive.md``. E1 applies the rules and scores them.

Usage: python tools/e1_report.py <runs-root> <data-root> [> docs/E1_report.md]
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")

from masattr.eval.ci import bootstrap_ci  # noqa: E402
from masattr.eval.scorers import exact_agent, exact_step  # noqa: E402
from masattr.loaders.whowhen_ag import load as load_ag  # noqa: E402
from masattr.loaders.whowhen_hc import load as load_hc  # noqa: E402
from masattr.typing.normalize import is_orchestrator  # noqa: E402

RULES = (
    "changepoint_single",
    "first_crossing",
    "argmin",
    "changepoint",
    "agent_first",
    "relative_crossing@1.5",
    "relative_crossing@2.0",
    "relative_crossing@2.5",
)
PRIMARY = "changepoint_single"
SLICES = ("all", "excl_flagged", "excl_anomalous", "excl_all_excluded")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def corpus(data_root: Path):
    ag = load_ag(data_root / "who_and_when/Algorithm-Generated.parquet", anomaly_policy="flag")[0]
    hc = load_hc(data_root / "who_and_when/Hand-Crafted.parquet", anomaly_policy="flag")[0]
    return {r.key: r for r in ag + hc}


def _subset(label: str) -> str:
    return [p for p in label.split(" · ") if p.startswith("subset=")][0].split("=")[1]


def cfgs(res: dict) -> dict[str, dict]:
    return {_subset(l): c for l, c in res["configs"].items()}


def sec_scoring(root: Path) -> list[str]:
    out = ["## 1. Scoring pass", ""]
    rows, partial = [], False
    for gt in ("nogt", "gt"):
        m = load(root / "main/judge/manifest.json") if gt == "gt" else None
        m = m or load(root / f"main/judge_{gt}/manifest.json")
        if not m:
            continue
        for subset, blk in (m.get("results") or {}).items():
            c = blk.get("cost", {})
            if not c.get("n_assessments"):
                partial = True
                continue
            rows.append(
                f"| {subset} | {gt} | {c['n_assessments']:,} | {c.get('seconds_total', 0):.0f} s | "
                f"{c.get('fraction_assessments_truncated', 0):.1%} | "
                f"{c.get('fraction_trajectories_truncated', 0):.1%} | "
                f"{c.get('prefix_tokens_max', 0):,} tok | {c.get('n_parse_failures', 0)} |"
            )
    if rows:
        out += [
            "| subset | GT | assessments | wall-clock | truncated assessments | "
            "truncated trajectories | max prefix | parse failures |",
            "|---|---|---|---|---|---|---|---|",
            *rows,
        ]
    out += [
        "",
        "> Cost figures cover only the trajectories judged in each run's **final**"
        " invocation. The scoring pass was resumable and was resumed, so"
        " trajectories served from cache report no timing. Row counts on disk are"
        " complete (1,099 alg / 2,993 hc per setting); the wall-clock column is not"
        " a total for the whole corpus." if partial else "",
    ]
    return out


def sec_e0_field(root: Path) -> list[str]:
    res = load(root / "main/e0_nogt/results.json")
    if not res:
        return []
    out = [
        "",
        "## 2. E0 — score field (GT off)",
        "",
        "| cell | n | mean | sd | p05 | median | p95 | saturated | distinct | AUROC vs derived labels |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    cells = res["field_sanity"]["cells"]
    for k, v in sorted(cells.items()):
        au = v.get("auroc_vs_derived_labels")
        out.append(
            f"| {k} | {v['n']:,} | {v['mean']:.3f} | {v['sd']:.3f} | {v['p05']:.3f} | "
            f"{v['median']:.3f} | {v['p95']:.3f} | {v['saturated_fraction']:.1%} | "
            f"{v['n_distinct']} | {'—' if au is None else f'{au:.3f}'} |"
        )
    deg = res["field_sanity"].get("degenerate") or []
    out += [
        "",
        f"**Degenerate cells: {'none' if not deg else '; '.join(deg)}.**",
        "",
        "Cells with n < 20 are flagged undersized by the fitter and are not read as "
        "evidence: " + ", ".join(f"`{k}` (n={v['n']})" for k, v in sorted(cells.items())
                                 if v.get("undersized")) + ".",
    ]
    return out


def sec_stability(root: Path) -> list[str]:
    out = [
        "",
        "## 3. E0 — cross-fold threshold stability",
        "",
        "| GT | subset | folds | global threshold mean | sd | CV | range |",
        "|---|---|---|---|---|---|---|",
    ]
    worst = []
    for gt in ("nogt", "gt"):
        res = load(root / f"main/e0_{gt}/results.json")
        if not res:
            continue
        for subset, s in sorted(res["stability"].items()):
            g = s["global_threshold"]
            out.append(
                f"| {gt} | {subset} | {s['n_folds']} | {g['mean']:+.3f} | {g['sd']:.3f} | "
                f"**{g['cv']:.3f}** | [{g['min']:+.3f}, {g['max']:+.3f}] |"
            )
        worst.append(f"{gt}: `{res.get('worst_threshold_cell')}` {res.get('worst_threshold_cv', 0):.3f}")
    out += ["", "Worst per-type threshold CV — " + "; ".join(worst) + "."]
    return out


def sec_fallback(root: Path, meta: dict) -> list[str]:
    out = [
        "",
        "## 4. E1 — primary rule fallback",
        "",
        f"`{PRIMARY}` falls back to argmin on boundary splits, contrast below the "
        "registered bound (z units), or trajectories too short to split.",
        "",
        "| subset | GT off | GT on | reasons (GT off) |",
        "|---|---|---|---|",
    ]
    fb: dict[str, dict] = {}
    for gt in ("nogt", "gt"):
        res = load(root / f"main/e1_{gt}/results.json")
        if res:
            for subset, c in cfgs(res).items():
                fb.setdefault(subset, {})[gt] = c.get("primary_fallback", {})
    for subset, byg in sorted(fb.items()):
        off, on = byg.get("nogt", {}), byg.get("gt", {})
        reasons = ", ".join(f"{k} {v}" for k, v in sorted((off.get("reasons") or {}).items()))
        out.append(
            f"| {subset} (n={off.get('n', '—')}) | **{off.get('rate', 0):.1%}** | "
            f"{on.get('rate', 0):.1%} | {reasons} |"
        )

    # fallback against trajectory length, pooled across subsets, GT off
    res = load(root / "main/e1_nogt/results.json")
    if res:
        buckets: dict[str, list[int]] = {}
        for subset, c in cfgs(res).items():
            for key, p in c["predictions"][PRIMARY].items():
                rec = meta.get(key)
                if rec is None:
                    continue
                n = rec.n_steps
                b = "< 10" if n < 10 else "10–20" if n < 20 else "20–50" if n < 50 else "50+"
                buckets.setdefault(b, []).append(1 if p.get("fallback") else 0)
        if any(buckets.values()):
            out += [
                "",
                "Fallback against trajectory length, pooled across subsets (GT off):",
                "",
                "| steps | n | fallback |",
                "|---|---|---|",
            ]
            for b in ("< 10", "10–20", "20–50", "50+"):
                v = buckets.get(b)
                if v:
                    out.append(f"| {b} | {len(v)} | {st.fmean(v):.1%} |")
    return out


def sec_accuracy(root: Path) -> list[str]:
    out = [
        "",
        "## 5. E1 — attribution accuracy, exact scorer, all files",
        "",
        "Bootstrap CIs over files, 2,000 resamples. **Bold** = registered primary.",
    ]
    for gt, title in (("nogt", "GT off"), ("gt", "GT on")):
        res = load(root / f"main/e1_{gt}/results.json")
        if not res:
            continue
        by = cfgs(res)
        out += [
            "",
            f"### {title}",
            "",
            "| rule | alg agent | alg step | hc agent | hc step |",
            "|---|---|---|---|---|",
        ]
        for rule in RULES:
            cells = []
            for subset in ("alg", "hc"):
                sc = by.get(subset, {}).get("scores", {}).get(rule, {}).get("exact/all")
                if not sc:
                    cells += ["—", "—"]
                    continue
                a, s = sc["agent_ci"], sc["step_ci"]
                cells += [
                    f"{sc['agent_acc']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]",
                    f"{sc['step_acc']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]",
                ]
            if all(c == "—" for c in cells):
                continue
            b = "**" if rule == PRIMARY else ""
            out.append(f"| {b}{rule}{b} | " + " | ".join(f"{b}{c}{b}" for c in cells) + " |")
    return out


def sec_fault_split(root: Path, meta: dict) -> list[str]:
    out = [
        "",
        f"## 6. E1 — orchestrator vs worker fault ({PRIMARY}, exact scorer)",
        "",
        "| GT | subset | fault | n | agent | step |",
        "|---|---|---|---|---|---|",
    ]
    for gt in ("nogt", "gt"):
        res = load(root / f"main/e1_{gt}/results.json")
        if not res:
            continue
        for subset, c in sorted(cfgs(res).items()):
            preds = c["predictions"][PRIMARY]
            for grp in ("orchestrator", "worker"):
                units = []
                for key, p in preds.items():
                    rec = meta.get(key)
                    if rec is None:
                        continue
                    if is_orchestrator(rec.label_mistake_agent) != (grp == "orchestrator"):
                        continue
                    units.append(
                        (
                            exact_agent(p.get("pred_agent"), rec.label_mistake_agent),
                            exact_step(p.get("pred_step"), rec.label_mistake_step),
                        )
                    )
                if len(units) < 2:
                    continue
                a = bootstrap_ci(units, lambda x: sum(i for i, _ in x) / len(x), n_boot=2000)
                s = bootstrap_ci(units, lambda x: sum(j for _, j in x) / len(x), n_boot=2000)
                out.append(
                    f"| {gt} | {subset} | {grp} | {len(units)} | "
                    f"{a.point:.3f} [{a.lo:.3f}, {a.hi:.3f}] | "
                    f"{s.point:.3f} [{s.lo:.3f}, {s.hi:.3f}] |"
                )
    return out


def sec_slices(root: Path) -> list[str]:
    out = [
        "",
        f"## 7. E1 — dual reporting across slices ({PRIMARY})",
        "",
        "| GT | subset | column | " + " | ".join(SLICES) + " |",
        "|---|---|---|" + "---|" * len(SLICES),
    ]
    for gt in ("nogt", "gt"):
        res = load(root / f"main/e1_{gt}/results.json")
        if not res:
            continue
        for subset, c in sorted(cfgs(res).items()):
            sc = c["scores"].get(PRIMARY, {})
            for col in ("agent_acc", "step_acc"):
                cells = []
                for sl in SLICES:
                    v = sc.get(f"exact/{sl}")
                    cells.append("—" if not v else f"{v[col]:.3f} ({v['n']})")
                out.append(f"| {gt} | {subset} | {col.replace('_acc', '')} | " + " | ".join(cells) + " |")
    return out


def sec_position(root: Path, meta: dict) -> list[str]:
    out = [
        "",
        f"## 8. E1 — normalized position of the attributed step ({PRIMARY})",
        "",
        "Where the rule places its pick, against where gold sits, as a fraction of "
        "trajectory length.",
        "",
        "| GT | subset | mean predicted | mean gold | pred in [0,0.2) | gold in [0,0.2) |",
        "|---|---|---|---|---|---|",
    ]
    for gt in ("nogt", "gt"):
        res = load(root / f"main/e1_{gt}/results.json")
        if not res:
            continue
        for subset, c in sorted(cfgs(res).items()):
            pp, gg = [], []
            for key, p in c["predictions"][PRIMARY].items():
                rec = meta.get(key)
                if rec is None or rec.n_steps < 2:
                    continue
                if p.get("pred_step") is not None:
                    pp.append(p["pred_step"] / (rec.n_steps - 1))
                if 0 <= rec.label_mistake_step < rec.n_steps:
                    gg.append(rec.label_mistake_step / (rec.n_steps - 1))
            if not pp or not gg:
                continue
            out.append(
                f"| {gt} | {subset} | {st.fmean(pp):.3f} | {st.fmean(gg):.3f} | "
                f"{sum(1 for x in pp if x < 0.2) / len(pp):.1%} | "
                f"{sum(1 for x in gg if x < 0.2) / len(gg):.1%} |"
            )
    return out


def header(root: Path) -> list[str]:
    m = load(root / "main/e1_nogt/manifest.json") or {}
    h = m.get("spec_hashes", {})
    return [
        "# E0 + E1 results — reference row",
        "",
        "> **Generated file.** Rebuild with `python tools/e1_report.py runs <data-root>`. "
        "Do not hand-edit: the previous hand-written version went stale when the "
        "readout scaffold changed and had to be superseded.",
        "",
        "The B0 reference row of the pilot suite, reported in full: all eight rules "
        "(the four in the master table plus `changepoint`, `agent_first`, and the "
        "`relative_crossing` k-sweep), both GT settings, every pre-registered slice.",
        "",
        f"Run provenance: commit `{m.get('commit', '—')}`, rule directive hash "
        f"`{h.get('rule_directive', '—')}`, prompts `{h.get('prompts', '—')}`, "
        f"type rules `{h.get('type_rules', '—')}`, criteria `{h.get('criteria', '—')}`, "
        f"judge `{h.get('judge', '—')}`.",
        "",
        "Evidence arm **W0** (prefix-conditional, no lookahead). Anomaly policy "
        "`flag`. Normalization: per-type leave-one-file-out CV, fit separately per "
        "subset and per GT setting. E0 is sanity-only and gates nothing — the "
        "primary rule is fixed by `specs/rule_directive.md`.",
    ]


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "runs")
    meta = corpus(Path(argv[2] if len(argv) > 2 else "data"))
    lines = header(root)
    lines += sec_scoring(root)
    lines += sec_e0_field(root)
    lines += sec_stability(root)
    lines += sec_fallback(root, meta)
    lines += sec_accuracy(root)
    lines += sec_fault_split(root, meta)
    lines += sec_slices(root)
    lines += sec_position(root, meta)
    lines += [
        "",
        "## 9. Related documents",
        "",
        "| doc | scope |",
        "|---|---|",
        "| `pilot_baseline_report.md` | the B0–B4 suite; this row is its B0 |",
        "| `delta_field_report.md` | D1, the lookahead-shift (δ) fields |",
        "| `runs/base/MASTER.md` | all rows, four rules, one table |",
        "| `runs/base/TOPK.md` | recall@k over the score ranking |",
        "",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
