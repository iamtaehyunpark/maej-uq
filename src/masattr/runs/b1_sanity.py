"""B1 — judge-free sanity rows (pilot baseline spec).

Five predictors that use no model: ``prior_position``, ``majority_agent``,
``first_step``, ``last_step``, and ``uniform_random_step``. Every field-based
row has to clear these.

The two to read carefully are ``prior_position`` — a constant guess at the
label distribution's known early skew — and ``majority_agent`` on HC, where the
orchestrator owns most steps and therefore wins the agent column for free. E1's
HC orchestrator cell should be read against that number before the field is
credited with it.

Scored through the same grid as every other row: all four scorers, every
pre-registered slice, file-level bootstrap CIs, AG and HC separately.
"""

from __future__ import annotations

import argparse
import json

from ..baselines.naive import PREDICTORS, uniform_random_expectations
from ..eval.ci import bootstrap_ci
from ..eval.scorers import SCORERS, gold_map, score_pairs, slices
from ._shared import add_common, emit, load_records, open_manifest


class _Pred:
    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def as_pair(self):
        return self.pair


def build_parser() -> argparse.ArgumentParser:
    p = add_common(argparse.ArgumentParser(prog="masattr b1", description=__doc__))
    p.add_argument("--draws", type=int, default=100, help="uniform_random_step draws")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = open_manifest("b1_sanity", args)
    by_subset = load_records(args)
    manifest.record_anomalies([r for rs in by_subset.values() for r in rs])

    results: dict = {"draws": args.draws, "rows": {}}
    lines = [
        "# B1 — judge-free sanity rows",
        "",
        "| subset | row | scorer | slice | n | agent | step |",
        "|---|---|---|---|---|---|---|",
    ]

    for subset, records in sorted(by_subset.items()):
        gold = gold_map(records)
        for name, fn in PREDICTORS.items():
            preds = {r.key: _Pred(fn(r)) for r in records}
            for slice_name, keys in slices(records).items():
                usable = sorted(k for k in keys if k in preds)
                pairs = [(preds[k].as_pair(), gold[k]) for k in usable]
                for scorer in SCORERS:
                    sc = score_pairs(
                        pairs,
                        scorer=scorer,
                        slice_name=slice_name,
                        n_boot=args.n_boot,
                        seed=args.seed,
                    )
                    results["rows"].setdefault(f"{subset}/{name}", {})[
                        f"{scorer}/{slice_name}"
                    ] = sc.to_dict()
                    if slice_name == "all":
                        lines.append(
                            f"| {subset} | {name} | {scorer} | {slice_name} | {sc.n} | "
                            f"{sc.agent_acc:.3f} [{sc.agent_ci.lo:.3f}, {sc.agent_ci.hi:.3f}] | "
                            f"{sc.step_acc:.3f} [{sc.step_ci.lo:.3f}, {sc.step_ci.hi:.3f}] |"
                        )

        # uniform_random is an expectation per file, not a single prediction, so
        # the randomness is averaged inside each file and the bootstrap is left
        # to measure file-to-file variation.
        exp = uniform_random_expectations(records, draws=args.draws, seed=args.seed)
        units = [exp[r.key] for r in records if r.key in exp]
        a = bootstrap_ci(units, lambda u: sum(x for x, _ in u) / len(u), n_boot=args.n_boot)
        s = bootstrap_ci(units, lambda u: sum(y for _, y in u) / len(u), n_boot=args.n_boot)
        results["rows"][f"{subset}/uniform_random_step"] = {
            "expectation/all": {
                "n": len(units),
                "agent_acc": a.point,
                "step_acc": s.point,
                "agent_ci": a.to_dict(),
                "step_ci": s.to_dict(),
            }
        }
        lines.append(
            f"| {subset} | uniform_random_step | expectation | all | {len(units)} | "
            f"{a.point:.3f} [{a.lo:.3f}, {a.hi:.3f}] | {s.point:.3f} [{s.lo:.3f}, {s.hi:.3f}] |"
        )

    lines += [
        "",
        "> `uniform_random_step` is the per-file expected accuracy over "
        f"{args.draws} seeded draws, so its interval reflects file-to-file "
        "variation rather than sampling noise.",
        "",
        "> Agent matching collapses orchestrator spellings, so `majority_agent` "
        "on HC measures how far trajectory composition alone carries the agent "
        "column.",
    ]
    emit(manifest, results, "\n".join(lines), args.out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
