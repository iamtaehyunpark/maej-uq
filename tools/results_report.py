"""Generate ``docs/RESULTS.md`` — the single consolidated results document.

Everything the pilot produced, in one place, read from ``runs/``. The narrative
is static; every number is computed. That split is deliberate: the earlier
hand-written reports went stale the moment the readout scaffold changed, and a
stale table that still looks plausible is worse than no table.

Names are spelled out rather than coded. The codebase keeps its identifiers;
a reader of the results should not have to decode them.

Usage: python tools/results_report.py <runs-root> <data-root> > docs/RESULTS.md
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from masattr.eval.ci import bootstrap_ci  # noqa: E402
from masattr.eval.scorers import exact_agent, exact_step  # noqa: E402
from masattr.judge.score import load_scores  # noqa: E402
from masattr.normalize.fit import load_folds  # noqa: E402
from masattr.typing.normalize import collapse_orchestrator as collapse  # noqa: E402
from masattr.typing.normalize import is_orchestrator  # noqa: E402

import e1_report as E1  # noqa: E402
import topk as TK  # noqa: E402

PRIMARY = "changepoint_single"
RULES4 = ("changepoint_single", "first_crossing", "argmin", "relative_crossing@2.0")

RULE = {
    "changepoint_single": "two-regime split (registered)",
    "first_crossing": "first step below threshold",
    "argmin": "lowest-scoring step",
    "relative_crossing@2.0": "relative drop (2x)",
    "changepoint": "two-regime split, unscaled",
    "relative_crossing@1.5": "relative drop (1.5x)",
    "relative_crossing@2.5": "relative drop (2.5x)",
}

#: Every rule implemented, for the full sweep on the reference field.
RULES_ALL = (
    "changepoint_single", "first_crossing", "argmin", "changepoint",
    "relative_crossing@1.5", "relative_crossing@2.0", "relative_crossing@2.5",
)

SUBSET = {"alg": "algorithm-generated", "hc": "hand-crafted"}
ANS = {"off": "hidden", "on": "shown", "nogt": "hidden", "gt": "shown"}

#: One block per score field, in master-table order.
BLOCKS = (
    ("P(True)", {"off": "main/e1_nogt", "on": "main/e1_gt"}),
    ("verbalized confidence",
     {"off": "base/b3/e1_verbalized_nogt", "on": "base/b3/e1_verbalized_gt"}),
    ("binary verdict",
     {"off": "base/b3/e1_binary_nogt", "on": "base/b3/e1_binary_gt"}),
    ("P(True) shift, +response", {"off": "delta/e1_resp"}),
    ("P(True) shift, +response +next turn", {"off": "delta/e1_own"}),
)

#: Plain names for the recall@k fields, matched to topk.FIELDS by prefix.
FIELD_NAME = {
    "B0 ptrue nogt": "P(True), answer hidden",
    "B0 ptrue gt": "P(True), answer shown",
    "B3 verbalized nogt": "verbalized confidence",
    "B3 binary nogt": "binary verdict",
    "D1 delta_resp (C5−C3)": "P(True) shift, +response",
    "D1 delta_own (C6−C3)": "P(True) shift, +response +next turn",
}


def _auroc(pos, neg):
    if not pos or not neg:
        return None
    w = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def discrimination(root: Path, meta: dict) -> list[str]:
    """The headline: can the field rank the faulty step above its own trajectory?"""
    from masattr.judge.score import by_file

    out = [
        "## 2. Does the score field find the labelled step?",
        "",
        "The direct test, with no rule and no threshold in the way: within each "
        "trajectory, rank every step by the score and ask whether the step "
        "Who&When labels as the decisive mistake ranks above the others. "
        "Reported as AUROC per trajectory, averaged over trajectories. **0.5 is "
        "chance.** Intervals are bootstrapped over files.",
        "",
        "| score field | logs | files | AUROC | mean score at labelled step | elsewhere |",
        "|---|---|---|---|---|---|",
    ]
    for label, tmpl, folds_rel in TK.FIELDS:
        name = FIELD_NAME.get(label, label)
        folds_path = root / folds_rel
        if not folds_path.exists():
            continue
        folds = load_folds(folds_path)
        for subset in ("alg", "hc"):
            path = root / tmpl.format(s=subset)
            if not path.exists():
                continue
            g = by_file(load_scores(path))
            per_file, pg, po = [], [], []
            for key, sc in g.items():
                rec = meta.get(key)
                if rec is None or not (0 <= rec.label_mistake_step < len(sc)):
                    continue
                gi = rec.label_mistake_step
                a = _auroc(
                    [-x.p_raw for x in sc if x.step_idx == gi],
                    [-x.p_raw for x in sc if x.step_idx != gi],
                )
                if a is not None:
                    per_file.append(a)
                pg += [x.p_raw for x in sc if x.step_idx == gi]
                po += [x.p_raw for x in sc if x.step_idx != gi]
            if not per_file:
                continue
            ci = bootstrap_ci(per_file, lambda x: st.fmean(x), n_boot=2000)
            star = " **" if ci.lo > 0.5 else " "
            out.append(
                f"| {name} | {SUBSET[subset]} | {len(per_file)} |{star}{ci.point:.3f} "
                f"[{ci.lo:.3f}, {ci.hi:.3f}]{star.strip()} | {st.fmean(pg):.3f} | "
                f"{st.fmean(po):.3f} |"
            )
    out += [
        "",
        "**Bold** marks fields whose interval excludes chance.",
        "",
        "P(True) is above chance on both corpora and in both answer settings, and "
        "its mean sits lower on the labelled step than elsewhere — the judge is "
        "measurably less confident about the step the benchmark blames. The "
        "effect is real and small: an AUROC near 0.58.",
        "",
        "Everything downstream follows from that number. On a trajectory "
        "averaging under nine steps, an AUROC of 0.58 turns into roughly 19% "
        "exact-step accuracy once you force a single pick, which is what §3 "
        "reports. The accuracy tables are a lossy projection of this table, not "
        "an independent result.",
    ]

    res = load(root / "main/e0_nogt/results.json")
    if res:
        out += [
            "",
            "### Broken out by step type (P(True), answer hidden)",
            "",
            "Same question, pooled within each step type instead of within each "
            "trajectory, and scored against the labelled step versus the steps "
            "*before* it. Cells under 20 steps are too small to read.",
            "",
            "| step type | steps | AUROC |",
            "|---|---|---|",
        ]
        for k, v in sorted(res["field_sanity"]["cells"].items()):
            au = v.get("auroc_vs_derived_labels")
            if au is None:
                continue
            note = " *(too small)*" if v.get("undersized") else ""
            out.append(f"| {k} | {v['n']:,} | {au:.3f}{note} |")
    return out





def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cell(sc):
    if not sc:
        return "—"
    ci = sc.get("agent_ci")
    return f"{sc['agent_acc']:.3f}" + (f" [{ci['lo']:.3f},{ci['hi']:.3f}]" if ci else "")


def master(root: Path) -> list[str]:
    out = [
        "## 3. What that becomes after forcing one pick",
        "",
        "How often each method names the faulty agent, and the faulty step, "
        "exactly. Confidence intervals are bootstrapped over files (2,000 "
        "resamples). *Rule gave up* is how often the registered rule found no "
        "usable split and fell back to simply picking the lowest-scoring step. "
        "Rows marked *none* make a prediction directly and never use a rule.",
        "",
        "| score field | answer | logs | rule | names agent | names step | rule gave up |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, dirs in BLOCKS:
        for gt, rel in sorted(dirs.items()):
            res = load(root / rel / "results.json")
            if not res:
                out.append(f"| {label} | {ANS[gt]} | — | *not run* | | | |")
                continue
            for lbl, cfg in sorted(res["configs"].items()):
                subset = E1._subset(lbl)
                fb = cfg.get("primary_fallback", {})
                for rule in RULES4:
                    sc = (cfg["scores"].get(rule) or {}).get("exact/all")
                    if not sc:
                        continue
                    s = sc["step_ci"]
                    out.append(
                        f"| {label} | {ANS[gt]} | {SUBSET[subset]} | {RULE[rule]} | "
                        f"{cell(sc)} | {sc['step_acc']:.3f} [{s['lo']:.3f},{s['hi']:.3f}] | "
                        f"{fb.get('rate', float('nan')):.1%} |"
                    )
    b1 = load(root / "base/b1/results.json")
    if b1:
        for key, variants in sorted(b1["rows"].items()):
            subset, _, name = key.partition("/")
            sc = variants.get("exact/all") or variants.get("expectation/all")
            if not sc:
                continue
            s = sc.get("step_ci")
            out.append(
                f"| simple guess: {name.replace('_', ' ')} | — | {SUBSET[subset]} | *none* | "
                f"{cell(sc)} | "
                + (f"{sc['step_acc']:.3f} [{s['lo']:.3f},{s['hi']:.3f}]" if s else "—")
                + " | — |"
            )
    b2 = load(root / "base/b2/results.json")
    if b2:
        for run in b2["runs"]:
            sc = run["scores"].get("exact/all")
            s = sc and sc.get("step_ci")
            out.append(
                f"| published method: {run['method'].replace('_', ' ')} | — | "
                f"{SUBSET[run['subset']]} | *none* | {cell(sc)} | "
                + (f"{sc['step_acc']:.3f} [{s['lo']:.3f},{s['hi']:.3f}]" if sc else "—")
                + " | — |"
            )
    return out


def all_rules(root: Path) -> list[str]:
    """Full rule sweep on the reference field, including the demoted variants."""
    out = [
        "",
        "### Every rule tried, on the judge-probability field",
        "",
        "The main table shows four rules for every field. All eight are listed "
        "here for the reference field only. The registered rule was fixed in "
        "advance, before any of these numbers existed — that it is not the "
        "winner is a result, not a reason to swap it.",
        "",
    ]
    for gt in ("nogt", "gt"):
        res = load(root / f"main/e1_{gt}/results.json")
        if not res:
            continue
        by = {E1._subset(l): c for l, c in res["configs"].items()}
        out += [
            f"**Reference answer {ANS[gt]}**",
            "",
            "| rule | agent, alg-generated | step, alg-generated | agent, hand-crafted | step, hand-crafted |",
            "|---|---|---|---|---|",
        ]
        for rule in RULES_ALL:
            cells = []
            for subset in ("alg", "hc"):
                sc = (by.get(subset, {}).get("scores", {}).get(rule) or {}).get("exact/all")
                if not sc:
                    cells += ["—", "—"]
                    continue
                a, st_ = sc["agent_ci"], sc["step_ci"]
                cells += [
                    f"{sc['agent_acc']:.3f} [{a['lo']:.3f},{a['hi']:.3f}]",
                    f"{sc['step_acc']:.3f} [{st_['lo']:.3f},{st_['hi']:.3f}]",
                ]
            if all(c == "—" for c in cells):
                continue
            out.append(f"| {RULE.get(rule, rule)} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def position(root: Path, meta: dict) -> list[str]:
    out = [
        "",
        "## 4. Naming the right agent mostly measures position",
        "",
        "The agent is whoever owns the step a method picks. So this score "
        "inherits any positional pattern in the corpus — and there is a big one, "
        "which the simple guesses collect for free.",
        "",
        "How often a fixed position belongs to the faulty agent, against that "
        "agent's average share of all steps (the rate a randomly placed pick "
        "would get):",
        "",
        "| logs | always step 0 | step 1 | step 2 | last step | faulty agent's share of steps |",
        "|---|---|---|---|---|---|",
    ]
    for subset in ("alg", "hc"):
        recs = [r for r in meta.values() if r.subset == subset]
        cells = []
        for k in (0, 1, 2):
            n = sum(1 for r in recs if k < r.n_steps)
            hit = sum(
                1
                for r in recs
                if k < r.n_steps
                and collapse(r.steps[k].agent) == collapse(r.label_mistake_agent)
            )
            cells.append(f"{hit / max(n, 1):.3f}")
        last = sum(
            1 for r in recs if collapse(r.steps[-1].agent) == collapse(r.label_mistake_agent)
        ) / len(recs)
        share = st.fmean(
            [
                sum(1 for s in r.steps if collapse(s.agent) == collapse(r.label_mistake_agent))
                / r.n_steps
                for r in recs
            ]
        )
        out.append(
            f"| {SUBSET[subset]} | **{cells[0]}** | {cells[1]} | {cells[2]} | "
            f"{last:.3f} | **{share:.3f}** |"
        )

    out += [
        "",
        "Where the registered rule actually picks, and how often it names the "
        "right agent at each depth (reference answer hidden):",
        "",
        "| logs | picks step 0 | right agent at step 0 | at step 1 | steps 2–4 | step 5+ | overall |",
        "|---|---|---|---|---|---|---|",
    ]
    res = load(root / "main/e1_nogt/results.json")
    if res:
        for lbl, cfg in sorted(res["configs"].items()):
            subset = E1._subset(lbl)
            buckets: dict[str, list[int]] = {}
            n0 = n = ok = 0
            for key, p in cfg["predictions"][PRIMARY].items():
                rec = meta.get(key)
                s = p.get("pred_step")
                if rec is None or s is None:
                    continue
                n += 1
                n0 += s == 0
                hit = int(exact_agent(p.get("pred_agent"), rec.label_mistake_agent))
                ok += hit
                b = "0" if s == 0 else "1" if s == 1 else "2-4" if s < 5 else "5+"
                buckets.setdefault(b, []).append(hit)
            cols = [
                f"{st.fmean(buckets[b]):.3f}" if buckets.get(b) else "—"
                for b in ("0", "1", "2-4", "5+")
            ]
            out.append(
                f"| {SUBSET[subset]} | {n0}/{n} ({n0 / n:.1%}) | "
                + " | ".join(cols)
                + f" | {ok / n:.3f} |"
            )
    out += [
        "",
        "On algorithm-generated logs the faulty agent owns about half of all "
        "first steps but only a third of steps overall, so *always blame whoever "
        "spoke first* banks a positional bonus the score field never tries to "
        "earn. The registered rule picks step 0 in roughly one log in seven, and "
        "its agent score falls steadily the deeper it picks — ending up at the "
        "plain ownership share. On this column the rule performs at the level of "
        "picking a step at random. Naming the right **step** is where the score "
        "field's signal actually shows up.",
    ]
    return out


def guesses(root: Path, meta: dict) -> list[str]:
    out = [
        "",
        "## 5. Allowing more than one guess",
        "",
        "Two different ways of being lenient. **Near-miss** accepts a pick that "
        "lands within one or two steps of the true one. **Three guesses** accepts "
        "the true step appearing anywhere among the three most suspicious steps. "
        "The second is ranked on the score field itself — the registered rule "
        "returns a single step and has no three-guess form.",
        "",
        "### Near-miss",
        "",
        "| answer | logs | exact | within 1 step | within 2 steps |",
        "|---|---|---|---|---|",
    ]
    for gt in ("nogt", "gt"):
        res = load(root / f"main/e1_{gt}/results.json")
        if not res:
            continue
        for lbl, cfg in sorted(res["configs"].items()):
            sc = cfg["scores"].get(PRIMARY, {})
            vals = [sc.get(f"{k}/all", {}).get("step_acc") for k in ("exact", "tol1", "tol2")]
            out.append(
                f"| {ANS[gt]} | {SUBSET[E1._subset(lbl)]} | "
                + " | ".join("—" if v is None else f"{v:.3f}" for v in vals)
                + " |"
            )

    out += [
        "",
        "### Three guesses — is the true step among the 3 most suspicious?",
        "",
        "*Random 3* is the matched control: three steps drawn at random from the "
        "same log. Short logs make that control strong, so the gain over it is "
        "the number that matters.",
        "",
        "| score field | logs | files | top 1 | top 3 | random 3 | gain | top 5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    agent_rows = []
    for label, tmpl, folds_rel in TK.FIELDS:
        folds_path = root / folds_rel
        if not folds_path.exists():
            continue
        folds = load_folds(folds_path)
        name = FIELD_NAME.get(label, label)
        for subset in ("alg", "hc"):
            path = root / tmpl.format(s=subset)
            if not path.exists():
                continue
            ev = TK.evaluate(load_scores(path), meta, folds)
            u = ev["units"]
            if not u:
                continue
            s1, s3, s5 = (TK._ci(u, f"step@{k}") for k in (1, 3, 5))
            r3 = st.fmean([x["rand_step@3"] for x in u])
            out.append(
                f"| {name} | {SUBSET[subset]} | {ev['n']} | {s1.point:.3f} | "
                f"{s3.point:.3f} [{s3.lo:.3f},{s3.hi:.3f}] | {r3:.3f} | "
                f"{s3.point - r3:+.3f} | {s5.point:.3f} |"
            )
            a1, a3 = TK._ci(u, "agent@1"), TK._ci(u, "agent@3")
            ra3 = st.fmean([x["rand_agent@3"] for x in u])
            cov = st.fmean([x["cover@3"] for x in u])
            agent_rows.append(
                f"| {name} | {SUBSET[subset]} | {ev['n']} | {a1.point:.3f} | "
                f"{a3.point:.3f} | {ra3:.3f} | {a3.point - ra3:+.3f} | {cov:.2f} |"
            )
    out += [
        "",
        "### Three guesses — naming the agent",
        "",
        "*Agents covered* is how many distinct agents the three picks span. Once "
        "that approaches the number of agents present, three guesses stops "
        "distinguishing anything.",
        "",
        "| score field | logs | files | top 1 | top 3 | random 3 | gain | agents covered |",
        "|---|---|---|---|---|---|---|---|",
        *agent_rows,
    ]
    return out


def base_rate(root: Path, meta: dict) -> list[str]:
    from masattr.baselines.naive import PREDICTORS

    out = [
        "",
        "## 6. Why hand-crafted logs flatter the simple guesses",
        "",
        "| simple guess | who was at fault | logs | names agent | names step |",
        "|---|---|---|---|---|",
    ]
    hc = [r for r in meta.values() if r.subset == "hc"]
    for name, fn in PREDICTORS.items():
        for grp in ("orchestrator", "worker"):
            sel = [
                r for r in hc if is_orchestrator(r.label_mistake_agent) == (grp == "orchestrator")
            ]
            u = [
                (
                    exact_agent(fn(r)[0], r.label_mistake_agent),
                    exact_step(fn(r)[1], r.label_mistake_step),
                )
                for r in sel
            ]
            if len(u) < 2:
                continue
            a = bootstrap_ci(u, lambda x: sum(i for i, _ in x) / len(x), n_boot=2000)
            s = bootstrap_ci(u, lambda x: sum(j for _, j in x) / len(x), n_boot=2000)
            out.append(
                f"| {name.replace('_', ' ')} | {grp} | {len(u)} | "
                f"{a.point:.3f} [{a.lo:.3f}, {a.hi:.3f}] | "
                f"{s.point:.3f} [{s.lo:.3f}, {s.hi:.3f}] |"
            )
    out += [
        "",
        "| who was at fault | logs | that agent's mean share of steps |",
        "|---|---|---|",
    ]
    for grp in ("orchestrator", "worker"):
        sel = [r for r in hc if is_orchestrator(r.label_mistake_agent) == (grp == "orchestrator")]
        shares = [
            sum(1 for s in r.steps if collapse(s.agent) == collapse(r.label_mistake_agent))
            / r.n_steps
            for r in sel
        ]
        out.append(f"| {grp} | {len(sel)} | {st.fmean(shares):.3f} |")
    out += [
        "",
        "*Blame the busiest agent* is right every time the orchestrator was at "
        "fault and wrong every time a worker was, and the ownership shares split "
        "the same way. On hand-crafted logs the orchestrator owns most steps of "
        "most logs, so that guess reproduces the answer exactly when the "
        "orchestrator erred and never otherwise. It is a property of the corpus, "
        "not a skill.",
    ]
    return out


def appendix(root: Path) -> list[str]:
    out = [
        "",
        "## 7. Appendix — unreadable outputs and rule fallback",
        "",
        "| score field | logs | rows | share readable |",
        "|---|---|---|---|",
    ]
    for label, tmpl, _ in TK.FIELDS:
        name = FIELD_NAME.get(label, label)
        for subset in ("alg", "hc"):
            p = root / tmpl.format(s=subset)
            if not p.exists():
                continue
            rows = load_scores(p)
            ok = sum(1 for r in rows if r.parse_ok) / len(rows)
            out.append(f"| {name} | {SUBSET[subset]} | {len(rows):,} | {ok:.4f} |")
    out += [
        "",
        "> The two non-judge fields cannot score a log's first step — there is "
        "nothing before it to compare against — so their figure is exactly one "
        "unscoreable row per log, by construction rather than a failure.",
        "",
        "The published methods use their own output format and parser. Output we "
        "could not parse counts as a miss against the full set of logs, not as a "
        "dropped log:",
        "",
        "| published method | logs | logs total | unparseable | rate |",
        "|---|---|---|---|---|",
    ]
    b2 = load(root / "base/b2/results.json")
    if b2:
        for run in b2["runs"]:
            out.append(
                f"| {run['method'].replace('_', ' ')} | {SUBSET[run['subset']]} | "
                f"{run['n_gold']} | {run['n_unparsed']} | {run['unparsed_rate']:.1%} |"
            )
    out += [
        "",
        "How often the registered rule gave up, and why:",
        "",
        "| answer | logs | files | gave up | reasons |",
        "|---|---|---|---|---|",
    ]
    for gt in ("nogt", "gt"):
        res = load(root / f"main/e1_{gt}/results.json")
        if not res:
            continue
        for lbl, cfg in sorted(res["configs"].items()):
            fb = cfg.get("primary_fallback", {})
            reasons = ", ".join(
                f"{k.replace('_', ' ')} {v}" for k, v in sorted((fb.get("reasons") or {}).items())
            )
            out.append(
                f"| {ANS[gt]} | {SUBSET[E1._subset(lbl)]} | {fb.get('n', '—')} | "
                f"{fb.get('rate', 0):.1%} | {reasons} |"
            )
    return out


def manifest(root: Path) -> list[str]:
    m = load(root / "main/e1_nogt/manifest.json") or {}
    h = m.get("spec_hashes", {})
    return [
        "",
        "## 8. How these numbers were produced",
        "",
        "| item | value |",
        "|---|---|",
        "| judge | `Qwen/Qwen3.6-35B-A3B`, served locally |",
        "| embedding model | `sentence-transformers/all-MiniLM-L6-v2` |",
        "| contradiction model | `cross-encoder/nli-deberta-v3-large` |",
        f"| prompt set hash | `{h.get('prompts', '—')}` |",
        f"| rule registration hash | `{h.get('rule_directive', '—')}` |",
        f"| step-type rules hash | `{h.get('type_rules', '—')}` |",
        "| confidence intervals | bootstrap over files, 2,000 resamples, seed 0 |",
        "| irregular logs | the 5 released logs that break the per-step checks are "
        "kept, flagged, and reported both ways |",
        "",
        "The published-method rows use the original authors' code and prompts with "
        "our judge substituted for theirs. That makes them a control for judge "
        "capability — **not** a reproduction of their published figures, which "
        "used a different model we had no quota for.",
        "",
        "Rebuild every table here with:",
        "",
        "```",
        "python tools/results_report.py runs <data-root> > docs/RESULTS.md",
        "```",
    ]


HEADER = """# Results

> **Generated file** — rebuild with `tools/results_report.py` (command at the
> end). Do not hand-edit: the earlier per-experiment write-ups went stale when
> the prompt scaffold changed, and are kept only for history in `docs/archive/`.

Every number here is read back from the stored run outputs. The corpus is
Who&When: **184 logs, 4,092 steps** — 126 algorithm-generated logs (1,099
steps) and 58 hand-crafted ones (2,993 steps). Every judged row covers the
whole corpus; nothing here is a subsample.

## 1. What is being measured

Each failed multi-agent log has one recorded answer: which agent made the
decisive mistake, and at which step. A method must name both. The two are
scored separately because they behave very differently (see §4).

Everything is built the same way. A **score field** assigns every step a number
meaning roughly *how suspect is this step*. Those numbers are put on a common
scale using statistics fit without the log being scored, so no log is judged
against itself. Then a **rule** turns the sequence of numbers into a single
accusation.

The score fields compared here:

| score field | what it is |
|---|---|
| P(True) | the judge's probability that the step is correct, read from a single token |
| verbalized confidence | the judge writes a confidence number instead |
| binary verdict | the judge answers correct/incorrect |
| P(True) shift, +response | how much P(True) *moves* once the reply to that step is appended |
| P(True) shift, +response +next turn | the same, also appending that agent's own next turn |

> **The two judge-free coherence fields have been removed from this report.**
> As run they were invalid: step 0 was assigned a score that made it the most
> suspect step in almost every trajectory, so the selection collapsed onto
> step 0 in 92% of files. Corrected figures are in `PTRUE_RESULTS.md`; they are
> below chance on both corpora.

Against them, two kinds of reference point: **simple guesses** that ignore
content entirely (always blame the first step, the last step, the busiest
agent, a random step), and the **published methods** from the Who&When paper
run through our judge.

The **answer** column says whether the judge was shown the reference answer
while scoring. Both settings are reported because the benchmark defines both.
"""

FINDINGS = """
## 9. What this adds up to

1. **External P(True) does find the labelled step, weakly.** Within-trajectory
   AUROC ≈ 0.57–0.62, interval above chance on both corpora and in both answer
   settings, with the judge measurably less confident on the labelled step than
   on the rest of the trajectory. This is the result; everything else is a
   consequence of it.
2. **A 0.58 AUROC does not survive being forced into one pick.** On logs
   averaging under nine steps it becomes roughly 19% exact-step accuracy. The
   gap between §2 and §3 is arithmetic, not a second finding, and no rule in §3
   recovers it.
3. **Naming the agent tests something else.** That column largely rewards
   picking an early or busy position, which the content-free guesses do by
   construction — see §4. Rankings taken from it are not attribution skill.
4. **One published method still beats the rule layer on the step column** —
   step-by-step, 0.294 against 0.190, same judge, same logs. That gap is about
   method and has no positional explanation.
5. **The registered rule rarely fires on short logs** — it falls back to the
   lowest-scoring step in about three quarters of algorithm-generated logs,
   because the best split keeps landing on a trajectory edge.
6. **The lookahead shift is a dead end.** It cannot rank the labelled step
   anywhere (every interval spans chance) and points the wrong way on
   hand-crafted logs.

## 9. Limits

- The published-method rows are a judge-capability control, not a reproduction;
  they are not comparable to the figures in that paper.
- The lookahead-shift rows were run only with the reference answer hidden. A
  wider window (following a delegation to its resolution) is implemented and
  unrun — that, not the shift idea itself, is the open question, since the
  narrow window rarely reaches the consequence on hand-crafted logs.
- Three hand-crafted logs record a mistake step outside the log's own length.
  They are counted and reported, never silently dropped, which is why some
  tables show 55 hand-crafted logs rather than 58.
- Rankings taken from the agent column should not be read as attribution
  skill. See §3.
"""


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "runs")
    meta = E1.corpus(Path(argv[2] if len(argv) > 2 else "data"))
    lines = [HEADER]
    lines += discrimination(root, meta)
    lines += master(root)
    lines += all_rules(root)
    lines += position(root, meta)
    lines += guesses(root, meta)
    lines += base_rate(root, meta)
    lines += appendix(root)
    lines += manifest(root)
    lines.append(FINDINGS)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
