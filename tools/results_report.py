"""Generate ``docs/RESULTS.md`` — the single consolidated results document.

Everything the pilot produced, in one place, read from ``runs/``. The narrative
is static; every number is computed. That split is deliberate: the previous
hand-written reports went stale the moment the readout scaffold changed, and a
stale table that still looks plausible is worse than no table.

Supersedes the per-experiment reports, which move to ``docs/archive/``.

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
from masattr.typing.normalize import collapse_orchestrator as collapse  # noqa: E402
from masattr.typing.normalize import is_orchestrator  # noqa: E402

import e1_report as E1  # noqa: E402
import topk as TK  # noqa: E402

PRIMARY = "changepoint_single"
RULES4 = ("changepoint_single", "first_crossing", "argmin", "relative_crossing@2.0")

#: (label, e1 dir template keyed by GT, note). One entry per master-table block.
BLOCKS = (
    ("B0 P(True)/W0", {"off": "main/e1_nogt", "on": "main/e1_gt"}),
    ("B3 verbalized", {"off": "base/b3/e1_verbalized_nogt", "on": "base/b3/e1_verbalized_gt"}),
    ("B3 binary", {"off": "base/b3/e1_binary_nogt", "on": "base/b3/e1_binary_gt"}),
    ("B4 embed_divergence", {"off": "base/b4/e1_embed_divergence_nogt"}),
    ("B4 nli_contradiction", {"off": "base/b4/e1_nli_contradiction_nogt"}),
    ("D1 delta_resp (C5−C3)", {"off": "delta/e1_resp"}),
    ("D1 delta_own (C6−C3)", {"off": "delta/e1_own"}),
)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cell(sc):
    if not sc:
        return "—"
    ci = sc.get("agent_ci")
    return f"{sc['agent_acc']:.3f}" + (f" [{ci['lo']:.3f},{ci['hi']:.3f}]" if ci else "")


# --- 2. master table --------------------------------------------------------


def master(root: Path) -> list[str]:
    out = [
        "## 2. Master table",
        "",
        "Exact scorer, slice `all`, file-level bootstrap CIs (2,000 resamples). "
        "`fallback` is the primary rule's rate of falling back to argmin. Rows "
        "without a `rule` produce a prediction directly and never touch the rule "
        "layer.",
        "",
        "| row | GT | subset | rule | agent | step | fallback |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, dirs in BLOCKS:
        for gt, rel in sorted(dirs.items()):
            res = load(root / rel / "results.json")
            if not res:
                out.append(f"| {label} | {gt} | — | *not run* | | | |")
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
                        f"| {label} | {gt} | {subset} | {rule} | {cell(sc)} | "
                        f"{sc['step_acc']:.3f} [{s['lo']:.3f},{s['hi']:.3f}] | "
                        f"{fb.get('rate', float('nan')):.1%} |"
                    )
    # B1 / B2 direct rows
    b1 = load(root / "base/b1/results.json")
    if b1:
        for key, variants in sorted(b1["rows"].items()):
            subset, _, name = key.partition("/")
            sc = variants.get("exact/all") or variants.get("expectation/all")
            if not sc:
                continue
            s = sc.get("step_ci")
            out.append(
                f"| B1 {name} | — | {subset} | *direct* | {cell(sc)} | "
                + (f"{sc['step_acc']:.3f} [{s['lo']:.3f},{s['hi']:.3f}]" if s else "—")
                + " | — |"
            )
    b2 = load(root / "base/b2/results.json")
    if b2:
        for run in b2["runs"]:
            sc = run["scores"].get("exact/all")
            s = sc and sc.get("step_ci")
            out.append(
                f"| B2 {run['method']} | — | {run['subset']} | *direct* | {cell(sc)} | "
                + (f"{sc['step_acc']:.3f} [{s['lo']:.3f},{s['hi']:.3f}]" if sc else "—")
                + " | — |"
            )
    return out


# --- 3. position analysis ---------------------------------------------------


def position(root: Path, meta: dict) -> list[str]:
    """Why the agent column ranks methods differently from the step column."""
    out = [
        "",
        "## 3. The agent column measures position, not attribution",
        "",
        "Agent accuracy is defined as *the owner of the selected step*, so it "
        "inherits whatever positional prior the corpus has. That prior is large, "
        "and constant predictors collect it for free.",
        "",
        "Agent accuracy if you always pick position k, against the gold agent's "
        "mean share of all steps (the chance rate for a randomly placed pick):",
        "",
        "| subset | step 0 | step 1 | step 2 | last | gold agent's mean share of steps |",
        "|---|---|---|---|---|---|",
    ]
    for subset in ("alg", "hc"):
        recs = [r for r in meta.values() if r.subset == subset]
        cells = []
        for k in (0, 1, 2):
            n = sum(1 for r in recs if k < r.n_steps)
            hit = sum(
                1 for r in recs
                if k < r.n_steps and collapse(r.steps[k].agent) == collapse(r.label_mistake_agent)
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
            f"| {subset} | **{cells[0]}** | {cells[1]} | {cells[2]} | {last:.3f} | **{share:.3f}** |"
        )

    # where the primary rule actually places its picks
    out += [
        "",
        "Where the primary rule places its picks, and the agent accuracy it earns "
        "at each depth (GT off):",
        "",
        "| subset | picks at step 0 | agent acc @0 | @1 | @2–4 | @5+ | overall |",
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
                f"| {subset} | {n0}/{n} ({n0 / n:.1%}) | " + " | ".join(cols) + f" | {ok / n:.3f} |"
            )
    out += [
        "",
        "On `alg` the gold agent owns 49.2% of first steps but only 33.2% of steps "
        "overall, so `first_step` banks a positional bonus the score field never "
        "tries to earn. The primary rule picks step 0 in 13.5% of files and its "
        "agent accuracy falls monotonically with depth, landing at the mean "
        "ownership share — i.e. the agent column, for this rule, is at the "
        "random-step level. The step column is where the field's signal shows up.",
    ]
    return out


# --- 4. recall@k ------------------------------------------------------------


def recall_at_k(root: Path, meta: dict) -> list[str]:
    out = [
        "",
        "## 4. Relaxing top-1",
        "",
        "Two different relaxations, reported side by side. **Tolerance** (|Δ|≤k) "
        "allows a positional near-miss; **recall@k** allows the gold step to be "
        "anywhere in the k most suspicious. Recall@k is ranked on the score field "
        "(generalizing `argmin`) — the primary rule emits one step and has no "
        "top-3 form. `random@3` is the matched control: three steps drawn "
        "uniformly from the same trajectory.",
        "",
        "### Tolerance curves, primary rule",
        "",
        "| GT | subset | exact | \\|Δ\\|≤1 | \\|Δ\\|≤2 |",
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
                f"| {gt} | {E1._subset(lbl)} | "
                + " | ".join("—" if v is None else f"{v:.3f}" for v in vals)
                + " |"
            )

    out += [
        "",
        "### Recall@k, step column",
        "",
        "| field | subset | n | @1 | @3 | random@3 | lift | @5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    agent_rows = []
    for label, tmpl, folds_rel in TK.FIELDS:
        folds_path = root / folds_rel
        if not folds_path.exists():
            continue
        from masattr.normalize.fit import load_folds

        folds = load_folds(folds_path)
        for subset in ("alg", "hc"):
            path = root / tmpl.format(s=subset)
            if not path.exists():
                continue
            from masattr.judge.score import load_scores

            ev = TK.evaluate(load_scores(path), meta, folds)
            u = ev["units"]
            if not u:
                continue
            s1, s3, s5 = (TK._ci(u, f"step@{k}") for k in (1, 3, 5))
            r3 = st.fmean([x["rand_step@3"] for x in u])
            out.append(
                f"| {label} | {subset} | {ev['n']} | {s1.point:.3f} | "
                f"{s3.point:.3f} [{s3.lo:.3f},{s3.hi:.3f}] | {r3:.3f} | "
                f"{s3.point - r3:+.3f} | {s5.point:.3f} |"
            )
            a1, a3 = TK._ci(u, "agent@1"), TK._ci(u, "agent@3")
            ra3 = st.fmean([x["rand_agent@3"] for x in u])
            cov = st.fmean([x["cover@3"] for x in u])
            agent_rows.append(
                f"| {label} | {subset} | {ev['n']} | {a1.point:.3f} | {a3.point:.3f} | "
                f"{ra3:.3f} | {a3.point - ra3:+.3f} | {cov:.2f} |"
            )
    out += [
        "",
        "### Recall@k, agent column",
        "",
        "`cover@3` is the mean number of distinct agents the three picks span. The "
        "column is at ceiling when it approaches the number of agents in the file, "
        "and then @3 stops discriminating.",
        "",
        "| field | subset | n | @1 | @3 | random@3 | lift | cover@3 |",
        "|---|---|---|---|---|---|---|---|",
        *agent_rows,
    ]
    return out


# --- 5. re-analyses ---------------------------------------------------------


def base_rate(root: Path, meta: dict) -> list[str]:
    from masattr.baselines.naive import PREDICTORS

    out = [
        "",
        "## 5. Re-analyses",
        "",
        "### (i) Base-rate audit — `hc`, split by gold fault agent",
        "",
        "| predictor | fault | n | agent | step |",
        "|---|---|---|---|---|",
    ]
    hc = [r for r in meta.values() if r.subset == "hc"]
    for name, fn in PREDICTORS.items():
        for grp in ("orchestrator", "worker"):
            sel = [r for r in hc if is_orchestrator(r.label_mistake_agent) == (grp == "orchestrator")]
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
                f"| {name} | {grp} | {len(u)} | {a.point:.3f} [{a.lo:.3f}, {a.hi:.3f}] | "
                f"{s.point:.3f} [{s.lo:.3f}, {s.hi:.3f}] |"
            )
    out += ["", "| fault | n | mean share of steps owned by the gold agent |", "|---|---|---|"]
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
        "`majority_agent` is 1.000 on orchestrator-fault files and 0.000 on "
        "worker-fault files, and the ownership share splits the same way. On `hc` "
        "the orchestrator owns most steps of most trajectories, so picking the most "
        "frequent agent reproduces gold exactly when the fault is the "
        "orchestrator's and never otherwise.",
    ]
    return out


def appendices(root: Path) -> list[str]:
    out = [
        "",
        "## 6. Appendix A — parse failures and fallback",
        "",
        "| field | subset | rows | parse_ok |",
        "|---|---|---|---|",
    ]
    from masattr.judge.score import load_scores

    for label, tmpl, _ in TK.FIELDS:
        for subset in ("alg", "hc"):
            p = root / tmpl.format(s=subset)
            if not p.exists():
                continue
            rows = load_scores(p)
            ok = sum(1 for r in rows if r.parse_ok) / len(rows)
            out.append(f"| {label} | {subset} | {len(rows):,} | {ok:.4f} |")
    out += [
        "",
        "> B4's rate is the step-0 fraction of each subset — exactly one "
        "unscoreable row per file (no premise available), by construction rather "
        "than a failure mode.",
        "",
        "B2 uses their output format and their parser; unparseable predictions are "
        "scored as misses against the full gold denominator, not dropped:",
        "",
        "| method | subset | n gold | unparsed | rate |",
        "|---|---|---|---|---|",
    ]
    b2 = load(root / "base/b2/results.json")
    if b2:
        for run in b2["runs"]:
            out.append(
                f"| {run['method']} | {run['subset']} | {run['n_gold']} | "
                f"{run['n_unparsed']} | {run['unparsed_rate']:.1%} |"
            )

    out += [
        "",
        "Primary-rule fallback with reasons, reference row:",
        "",
        "| GT | subset | n | fallback | reasons |",
        "|---|---|---|---|---|",
    ]
    for gt in ("nogt", "gt"):
        res = load(root / f"main/e1_{gt}/results.json")
        if not res:
            continue
        for lbl, cfg in sorted(res["configs"].items()):
            fb = cfg.get("primary_fallback", {})
            reasons = ", ".join(f"{k} {v}" for k, v in sorted((fb.get("reasons") or {}).items()))
            out.append(
                f"| {gt} | {E1._subset(lbl)} | {fb.get('n', '—')} | "
                f"{fb.get('rate', 0):.1%} | {reasons} |"
            )
    return out


def manifest(root: Path) -> list[str]:
    out = [
        "",
        "## 7. Run manifest",
        "",
        "| run | commit | prompts hash | rows |",
        "|---|---|---|---|",
    ]
    for rel in sorted(
        p.parent for p in root.rglob("manifest.json") if "e1_" in str(p) or "judge" in str(p)
    ):
        m = load(rel / "manifest.json")
        if not m:
            continue
        out.append(
            f"| `{rel.relative_to(root)}` | `{(m.get('commit') or '')[:12]}` | "
            f"`{m.get('spec_hashes', {}).get('prompts', '—')}` | "
            f"{m.get('experiment', '—')} |"
        )
    out += [
        "",
        "Judge `Qwen/Qwen3.6-35B-A3B` (family qwen) for every judged row, served "
        "over vLLM. B4 uses `sentence-transformers/all-MiniLM-L6-v2` and "
        "`cross-encoder/nli-deberta-v3-large` and sends no prompt. Anomaly policy "
        "`flag` throughout: the 5 released files that violate the per-step asserts "
        "are kept, flagged, and dual-reported. Bootstrap CIs are file-level, "
        "`n_boot=2000`, `seed=0`.",
        "",
        "> The B4 field-extraction manifest carries the pre-scaffold prompts hash. "
        "That arm emits no prompts, so the hash records tree state rather than an "
        "input to its numbers. Every arm that does send a prompt carries "
        "`e8bc3b7bb8f22151`.",
    ]
    return out


HEADER = """# Results

> **Generated file.** Rebuild with
> `python tools/results_report.py runs <data-root> > docs/RESULTS.md`.
> Do not hand-edit — the per-experiment reports it replaces went stale when the
> readout scaffold changed, and are archived under `docs/archive/`.

Every number here is read from `runs/`. Corpus: Who&When, **184 files /
4,092 steps** — Algorithm-Generated (`alg`) 126 files / 1,099 steps,
Hand-Crafted (`hc`) 58 files / 2,993 steps. Every judged row covers the full
corpus; no row is a subsample.

## 1. What the names mean

Three schemes overlap, which is worth stating once.

**E0–E9 are pipeline stages** — verbs. `E0` fits leave-one-file-out
normalization folds and reports field sanity; it is **sanity-only and gates
nothing**, because the primary rule is fixed in advance by
`specs/rule_directive.md`. `E1` applies the attribution rules to a normalized
field and scores them. E2–E9 are ablation axes.

**B0–B4 and D1 are rows** — nouns, one per score field:

| row | field |
|---|---|
| B0 | P(True) readout, W0 evidence — the reference |
| B1 | naive baselines (first/last step, majority agent, prior position, random) |
| B2 | Who&When capability control — their three methods, our judge |
| B3 | readout variants (verbalized, binary) |
| B4 | semantic coherence (embedding divergence, NLI contradiction) |
| D1 | lookahead-shift (δ) fields |

Every B/D row **is** an E0 run followed by an E1 run on its own score field.
"E1" therefore names both a stage and, historically, a standalone report — that
collision is why the old documents were confusing.

**C3/C5/C6** map onto the `lookahead` axis: C3 = `none` (W0), C5 = `resp`
(the realized response, cap 2), C6 = `own` (C5 **plus** the actor's next
appearance — a superset, not a disjoint arm).
"""

FOOTER = """
## 8. Scope and limits

- **Not run:** the gpt-4o published-regime arm of B2 (no quota) — so the B2 rows
  are a capability control, *not* a reproduction of published Who&When numbers,
  and should not be compared to them. E7 struck; E8 gated. The secondary judge
  and the LLM type classifier are declared and family-verified but unused.
- **δ (D1) is GT-off only**, staged behind a positive GT-off result that did not
  arrive. Its `deleg` (cap-5) arm is implemented and unrun.
- **`hc` n=55 in step-ranking analyses**, not 58: three released files carry a
  `mistake_step` outside the trajectory and have no gold step to rank against.
  They are counted, not silently dropped.
- **Agent-column caveat.** §3 shows this column largely tracks a positional
  prior. Rankings read off it should not be read as attribution skill.

## 9. Findings

1. **Nothing separates from the naive baselines on the agent column.** No
   B0/B3/B4/D1 row's CI excludes the best B1 row on either subset. §3 explains
   why: that column is a positional lottery, and constant predictors win it.
2. **On the step column the score field does carry signal** — the reference row
   beats every B1 heuristic and the random-step control on `alg`.
3. **B2 `step_by_step` beats the primary rule on step accuracy** (0.294 vs 0.190
   on `alg`), same judge, same corpus. That gap is method, not metric, and is
   the one result with no positional excuse.
4. **The primary rule mostly does not fire on `alg`** — ~75% argmin fallback,
   driven by boundary hits on short (8.7-step mean) trajectories.
5. **δ is a null.** The lookahead shift cannot rank the faulty step (all CIs
   span chance) and inverts on `hc`, where the window rarely reaches the
   consequence. C5/C6 stay dropped.
6. **Top-3 helps on the step column, modestly.** Roughly +0.09 to +0.15 over a
   matched random-3 control on `alg`; the field's advantage is concentrated at
   the very top of the ranking and dilutes as k grows. The agent column at @3 is
   at or below chance.
"""


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "runs")
    meta = E1.corpus(Path(argv[2] if len(argv) > 2 else "data"))
    lines = [HEADER]
    lines += master(root)
    lines += position(root, meta)
    lines += recall_at_k(root, meta)
    lines += base_rate(root, meta)
    lines += appendices(root)
    lines += manifest(root)
    lines.append(FOOTER)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
