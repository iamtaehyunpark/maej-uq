"""E0 — single-agent → MAS calibration transfer (Part C §4). Runs first.

Fit per-type maps once on paper 1's scored single-agent corpus, freeze them,
then check on a seeded 20-file held-aside slice (10 AG + 10 HC) whether they
still say something true about multi-agent steps. The gates are pre-registered
here, and the outcome decides the calibration fallback **before any attribution
number is seen** (Part E).

Fail ⇒ disclosed fallback: leave-one-out CV on Who&When, and the uniformity
claim in the paper text is weakened accordingly.

Without ``--paper1-scores`` there is nothing to fit: the run stops and says so
rather than quietly calibrating on Who&When, which is the very thing the
fallback is supposed to disclose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..calib.apply import HELD_ASIDE_PER_SUBSET, held_aside, labelled_rows
from ..calib.fit import fit, load_paper1_scores, load_type_map
from ..eval.ci import auroc, reliability
from ..specs import TYPE_MAP_FILE
from ._shared import add_common, emit, flatten, load_records, open_manifest, read_scores

#: Pre-registered gates. Transfer holds if the frozen maps improve (or at least
#: do not worsen) calibration error without degrading ranking.
MAX_AUROC_DROP = 0.02
MAX_ECE_INCREASE = 0.02


def build_parser() -> argparse.ArgumentParser:
    p = add_common(argparse.ArgumentParser(prog="masattr e0", description=__doc__))
    p.add_argument(
        "--paper1-scores",
        required=True,
        help="JSONL from paper 1: {step_id, arm, model, benchmark, step_kind, tau, "
        "p_raw, judge_model, label_correct} — raw single-probe P(True), not a "
        "combined score",
    )
    p.add_argument(
        "--judge-model",
        help="keep only rows whose judge_model matches, so the maps describe the "
        "judge that will actually be applied to Who&When",
    )
    p.add_argument(
        "--allow-draft-type-map",
        action="store_true",
        help="run against an unfrozen specs/paper1_type_map.json (exploration only)",
    )
    p.add_argument("--scores", nargs="+", required=True, help="Who&When step-score JSONL(s)")
    p.add_argument("--method", default="percentile", choices=("percentile", "platt", "isotonic"))
    p.add_argument("--label-policy", default="prefix", choices=("prefix", "point"))
    p.add_argument("--frozen-out", default="src/masattr/calib/frozen/calibration.json")
    p.add_argument(
        "--held-aside-per-subset",
        type=int,
        default=HELD_ASIDE_PER_SUBSET,
        help="pre-registered at 10 (Part C §4); any other value is recorded in the manifest",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = open_manifest("e0_transfer", args)

    type_map = load_type_map(TYPE_MAP_FILE)
    ps, types, ys, intake = load_paper1_scores(
        args.paper1_scores,
        type_map,
        judge_model=args.judge_model,
        require_frozen=not args.allow_draft_type_map,
    )
    if args.allow_draft_type_map:
        manifest.note(
            "type map was NOT frozen for this run — exploratory only, not an E0 result"
        )
    manifest.record_models(judge=args.judge_model or "", type_classifier="")
    cal = fit(
        ps,
        types,
        ys,
        method=args.method,
        fit_on=str(args.paper1_scores),
        type_map_hash=manifest.spec_hashes.get("paper1_type_map", ""),
        extra={"intake": intake},
    )
    frozen_path = cal.save(args.frozen_out)
    manifest.calibration_hash = cal.content_hash()

    records = flatten(load_records(args))
    scores: dict = {}
    for path in args.scores:
        scores.update(read_scores(path))

    manifest.record_anomalies(records)
    held = held_aside(records, seed=args.seed, per_subset=args.held_aside_per_subset)
    if args.held_aside_per_subset != HELD_ASIDE_PER_SUBSET:
        manifest.note(
            f"held-aside slice is {args.held_aside_per_subset} per subset, not the "
            f"pre-registered {HELD_ASIDE_PER_SUBSET}"
        )
    held_records = [r for r in records if r.key in held]
    p_raw, t_held, y_held = labelled_rows(
        held_records, scores, policy=args.label_policy, keys=held
    )
    if p_raw.size == 0:
        raise SystemExit(
            "no labelled steps on the held-aside slice — score those 20 files before running E0"
        )
    p_cal = cal.apply_many(p_raw.tolist(), t_held)

    rel_raw = reliability(p_raw.tolist(), y_held.tolist())
    rel_cal = reliability(p_cal, y_held.tolist())
    a_raw, a_cal = auroc(p_raw.tolist(), y_held.tolist()), auroc(p_cal, y_held.tolist())
    transfers = (a_cal >= a_raw - MAX_AUROC_DROP) and (rel_cal.ece <= rel_raw.ece + MAX_ECE_INCREASE)

    results = {
        "fit_on": str(args.paper1_scores),
        "n_fit": int(ps.size),
        "intake": intake,
        "held_aside_files": sorted(held),
        "held_aside_per_subset": args.held_aside_per_subset,
        "label_policy": args.label_policy,
        "n_held_steps": int(p_raw.size),
        "auroc_raw": a_raw,
        "auroc_cal": a_cal,
        "auroc_delta": a_cal - a_raw,
        "ece_raw": rel_raw.ece,
        "ece_cal": rel_cal.ece,
        "ece_delta": rel_cal.ece - rel_raw.ece,
        "brier_raw": rel_raw.brier,
        "brier_cal": rel_cal.brier,
        "threshold": cal.threshold,
        "gates": {"max_auroc_drop": MAX_AUROC_DROP, "max_ece_increase": MAX_ECE_INCREASE},
        "transfers": transfers,
        "decision": (
            "apply the frozen single-agent calibration unchanged to Who&When"
            if transfers
            else "FALLBACK: leave-one-out CV on Who&When (disclosed); weaken the "
            "uniformity claim in the paper text"
        ),
        "frozen_calibration": str(frozen_path),
        "calibration_hash": cal.content_hash(),
        "reliability_raw": rel_raw.to_dict(),
        "reliability_cal": rel_cal.to_dict(),
        "provenance": cal.provenance,
    }

    md = "\n".join(
        [
            "## E0 — single-agent → MAS calibration transfer",
            "",
            f"fit on {args.paper1_scores} (n={ps.size}); checked on the seeded "
            f"held-aside 20 (n={p_raw.size} labelled steps, policy={args.label_policy})",
            "",
            "| | raw | calibrated | delta |",
            "|---|---|---|---|",
            f"| AUROC | {a_raw:.4f} | {a_cal:.4f} | {a_cal - a_raw:+.4f} |",
            f"| ECE | {rel_raw.ece:.4f} | {rel_cal.ece:.4f} | {rel_cal.ece - rel_raw.ece:+.4f} |",
            f"| Brier | {rel_raw.brier:.4f} | {rel_cal.brier:.4f} | "
            f"{rel_cal.brier - rel_raw.brier:+.4f} |",
            "",
            f"threshold={cal.threshold:.3f}  calibration_hash={cal.content_hash()}",
            "",
            f"**{'TRANSFERS' if transfers else 'DOES NOT TRANSFER'}** → {results['decision']}",
            "",
            rel_raw.render("(raw)"),
            "",
            rel_cal.render("(calibrated)"),
        ]
    )

    emit(manifest, results, md, args.out_dir)
    Path(args.out_dir, "threshold.json").write_text(
        json.dumps(
            {
                "threshold": cal.threshold,
                "thresholds": cal.thresholds,
                "chosen_on": str(args.paper1_scores),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if transfers else 2  # 2 = falsified; the fallback is now in force


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
