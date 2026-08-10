"""Recall@k for every score field: is the gold step (agent) in the k most
suspicious steps, rather than the single most suspicious one?

Top-k is defined over the **score ranking** — the k lowest normalized scores —
which generalizes ``argmin``. ``changepoint_single`` selects one step by
construction and has no top-3 analogue, so the primary rule is not the thing
being relaxed here; the field's ranking is.

Two controls are mandatory and are printed beside every row, because @3 is
trivially better than @1 and the question is whether it is better than getting
three guesses is worth:

* **random@k** — per file, the expectation of hitting gold with k steps drawn
  uniformly without replacement: ``k / n_steps`` for the step column, and for
  the agent column the exact expectation over which agents k random steps cover.
* **agent coverage@k** — the mean number of distinct agents the k picks span. A
  trajectory with 3.6 agents where the top-3 covers 3.0 of them has not
  identified anyone; the agent column is at ceiling and any gain is arithmetic.

Usage: python tools/topk.py <runs-root> <data-root> [> report.md]
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "src")

from masattr.eval.ci import bootstrap_ci  # noqa: E402
from masattr.judge.score import by_file, load_scores  # noqa: E402
from masattr.loaders.whowhen_ag import load as load_ag  # noqa: E402
from masattr.loaders.whowhen_hc import load as load_hc  # noqa: E402
from masattr.typing.normalize import collapse_orchestrator as collapse  # noqa: E402

KS = (1, 2, 3, 5)

#: Every field that produced a row in the master table, plus the delta arms.
FIELDS: tuple[tuple[str, str], ...] = (
    ("B0 ptrue nogt", "main/scores/{s}__served_Qwen-Qwen3.6-35B-A3B_ptrue_typed_nogt.jsonl"),
    ("B0 ptrue gt", "main/scores/{s}__served_Qwen-Qwen3.6-35B-A3B_ptrue_typed_gt.jsonl"),
    ("B3 verbalized nogt", "base/b3/scores/{s}__served_Qwen-Qwen3.6-35B-A3B_verbalized_typed_nogt.jsonl"),
    ("B3 binary nogt", "base/b3/scores/{s}__served_Qwen-Qwen3.6-35B-A3B_binary_typed_nogt.jsonl"),
    ("B4 embed_divergence", "base/b4/scores/{s}__embed_divergence_nogt.jsonl"),
    ("B4 nli_contradiction", "base/b4/scores/{s}__nli_contradiction_nogt.jsonl"),
    ("D1 delta_resp (C5−C3)", "delta/fields/{s}__delta_resp_nogt.jsonl"),
    ("D1 delta_own (C6−C3)", "delta/fields/{s}__delta_own_nogt.jsonl"),
)


def corpus(data_root: Path) -> dict:
    ag = load_ag(data_root / "who_and_when/Algorithm-Generated.parquet", anomaly_policy="flag")[0]
    hc = load_hc(data_root / "who_and_when/Hand-Crafted.parquet", anomaly_policy="flag")[0]
    return {r.key: r for r in ag + hc}


def _random_agent_hit(steps, gold_agent: str, k: int) -> float:
    """P(at least one of k uniformly drawn steps is owned by the gold agent).

    Exact hypergeometric complement — the gold agent owns ``m`` of ``n`` steps,
    so a miss means all k draws come from the other ``n - m``.
    """
    n = len(steps)
    m = sum(1 for s in steps if collapse(s.agent) == collapse(gold_agent))
    if m == 0:
        return 0.0
    k = min(k, n)
    miss = 1.0
    for i in range(k):
        avail = n - m - i
        if avail < 0:
            return 1.0
        miss *= avail / (n - i)
    return 1.0 - miss


def evaluate(rows, meta) -> dict:
    """Per-file hit indicators at each k, plus the matched random controls."""
    units: list[dict] = []
    for key, scores in by_file(rows).items():
        rec = meta.get(key)
        if rec is None or rec.n_steps < 2:
            continue
        gold_step, gold_agent = rec.label_mistake_step, rec.label_mistake_agent
        if not 0 <= gold_step < len(scores):
            continue  # released files with an out-of-range mistake_step
        ranked = sorted(scores, key=lambda s: s.p)  # low = suspicious
        u: dict = {"n_steps": len(scores)}
        for k in KS:
            top = ranked[:k]
            u[f"step@{k}"] = any(s.step_idx == gold_step for s in top)
            u[f"agent@{k}"] = any(collapse(s.agent) == collapse(gold_agent) for s in top)
            u[f"cover@{k}"] = len({collapse(s.agent) for s in top})
            u[f"rand_step@{k}"] = min(k, len(scores)) / len(scores)
            u[f"rand_agent@{k}"] = _random_agent_hit(scores, gold_agent, k)
        units.append(u)
    return {"n": len(units), "units": units}


def _ci(units: list[dict], field: str):
    if not units:
        return None
    return bootstrap_ci(units, lambda x: st.fmean([float(u[field]) for u in x]), n_boot=2000)


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "runs")
    meta = corpus(Path(argv[2] if len(argv) > 2 else "data"))

    out = [
        "# Recall@k — does relaxing top-1 to top-3 find the fault?",
        "",
        "Top-k is the k lowest-scoring (most suspicious) steps of a trajectory. "
        "`random@k` is the matched control: k steps drawn uniformly from the same "
        "trajectory. `cover@3` is the mean number of distinct agents the three "
        "picks span — the agent column is at ceiling when this approaches the "
        "number of agents in the file.",
        "",
        "## Step column",
        "",
        "| field | subset | n | @1 | @3 | random@3 | @3 − random@3 | @5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    agent_rows: list[str] = []

    def f(ci, p=3):
        return "—" if ci is None else f"{ci.point:.{p}f}"

    for label, tmpl in FIELDS:
        for subset in ("alg", "hc"):
            path = root / tmpl.format(s=subset)
            if not path.exists():
                continue
            ev = evaluate(load_scores(path), meta)
            u = ev["units"]
            if not u:
                continue
            s1, s3, s5 = (_ci(u, f"step@{k}") for k in (1, 3, 5))
            r3 = st.fmean([x["rand_step@3"] for x in u])
            gap = s3.point - r3
            out.append(
                f"| {label} | {subset} | {ev['n']} | {f(s1)} | "
                f"{s3.point:.3f} [{s3.lo:.3f},{s3.hi:.3f}] | {r3:.3f} | "
                f"{gap:+.3f} | {f(s5)} |"
            )
            a1, a3 = _ci(u, "agent@1"), _ci(u, "agent@3")
            ra3 = st.fmean([x["rand_agent@3"] for x in u])
            cov = st.fmean([x["cover@3"] for x in u])
            agent_rows.append(
                f"| {label} | {subset} | {ev['n']} | {f(a1)} | "
                f"{a3.point:.3f} [{a3.lo:.3f},{a3.hi:.3f}] | {ra3:.3f} | "
                f"{a3.point - ra3:+.3f} | {cov:.2f} |"
            )

    out += [
        "",
        "## Agent column",
        "",
        "| field | subset | n | @1 | @3 | random@3 | @3 − random@3 | cover@3 |",
        "|---|---|---|---|---|---|---|---|",
        *agent_rows,
    ]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
