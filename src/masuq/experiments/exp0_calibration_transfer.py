"""Experiment 0 — does frozen typed calibration transfer? (spec §4)

This is the falsifier and it runs **first**. Calibration is fit once on
MATU-AutoGen and frozen; Exp-0 asks whether that frozen map still says something
true on MATU-CAMEL, the other label-available cell. If it does not transfer
across two cells of the *same* benchmark, there is no basis for asserting it
transfers to Who&When, and the pre-registered fallback (leave-one-out CV on W&W,
disclosed) takes over for the attribution track. The trajectory track is
unaffected either way because it is calibrated in-corpus.

The outcome is decided here, before any attribution number is seen (spec §8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..calibration import TypedCalibrator, choose_threshold
from ..metrics import auroc, reliability
from ..schema import Record

#: Transfer is declared to hold if the frozen map does not *degrade* ranking on
#: the held-out cell beyond this, and improves (or at least does not worsen)
#: calibration error. Both are pre-registered here rather than chosen on sight.
MAX_AUROC_DROP = 0.02
MAX_ECE_INCREASE = 0.02


@dataclass
class Exp0Result:
    fit_subset: str
    test_subset: str
    n_fit: int
    n_test: int
    auroc_raw: float
    auroc_cal: float
    ece_raw: float
    ece_cal: float
    brier_raw: float
    brier_cal: float
    threshold: float
    transfers: bool = False
    reliability_raw: dict = field(default_factory=dict)
    reliability_cal: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    @property
    def auroc_delta(self) -> float:
        return self.auroc_cal - self.auroc_raw

    @property
    def ece_delta(self) -> float:
        return self.ece_cal - self.ece_raw

    @property
    def fallback(self) -> str:
        return (
            "frozen typed calibration applied unchanged to W&W"
            if self.transfers
            else "leave-one-out CV on W&W (disclosed); trajectory track unaffected"
        )

    def to_dict(self) -> dict:
        return {
            "fit_subset": self.fit_subset,
            "test_subset": self.test_subset,
            "n_fit": self.n_fit,
            "n_test": self.n_test,
            "auroc_raw": self.auroc_raw,
            "auroc_cal": self.auroc_cal,
            "auroc_delta": self.auroc_delta,
            "ece_raw": self.ece_raw,
            "ece_cal": self.ece_cal,
            "ece_delta": self.ece_delta,
            "brier_raw": self.brier_raw,
            "brier_cal": self.brier_cal,
            "threshold": self.threshold,
            "transfers": self.transfers,
            "decision": self.fallback,
            "gates": {"max_auroc_drop": MAX_AUROC_DROP, "max_ece_increase": MAX_ECE_INCREASE},
            "reliability_raw": self.reliability_raw,
            "reliability_cal": self.reliability_cal,
            "provenance": self.provenance,
        }

    def render(self) -> str:
        verdict = "TRANSFERS" if self.transfers else "DOES NOT TRANSFER"
        return "\n".join(
            [
                f"## Experiment 0 — frozen typed calibration, {self.fit_subset} → {self.test_subset}",
                "",
                f"| | raw | calibrated | delta |",
                "|---|---|---|---|",
                f"| AUROC | {self.auroc_raw:.4f} | {self.auroc_cal:.4f} | {self.auroc_delta:+.4f} |",
                f"| ECE | {self.ece_raw:.4f} | {self.ece_cal:.4f} | {self.ece_delta:+.4f} |",
                f"| Brier | {self.brier_raw:.4f} | {self.brier_cal:.4f} | "
                f"{self.brier_cal - self.brier_raw:+.4f} |",
                "",
                f"n_fit={self.n_fit}  n_test={self.n_test}  threshold={self.threshold:.3f}",
                f"**Verdict: {verdict}** → {self.fallback}",
            ]
        )


def _step_targets(
    records: Sequence[Record], scores_by_key: dict[str, list]
) -> tuple[list[float], list[str], list[bool]]:
    """Flatten step scores with the run label propagated to each step.

    The propagation is the stated assumption of :mod:`masuq.calibration`; it is
    repeated here so a reader of the experiment does not have to go looking.
    """
    p: list[float] = []
    types: list[str] = []
    y: list[bool] = []
    for rec in records:
        if rec.label_correct is None:
            continue
        for s in scores_by_key.get(rec.key, []):
            p.append(s.p_raw)
            types.append(s.type_norm)
            y.append(bool(rec.label_correct))
    return p, types, y


def run(
    fit_records: Sequence[Record],
    fit_scores: dict[str, list],
    test_records: Sequence[Record],
    test_scores: dict[str, list],
    *,
    method: str = "percentile",
    fit_subset: str = "autogen_mmlu",
    test_subset: str = "camel_math",
    out_dir: str | Path | None = None,
) -> tuple[Exp0Result, TypedCalibrator]:
    """Fit → freeze → test. Returns the result and the frozen calibrator."""
    p_fit, t_fit, y_fit = _step_targets(fit_records, fit_scores)
    if not p_fit:
        raise ValueError("no labelled fit data; join MATU labels before running Exp-0")

    cal = TypedCalibrator(method=method).fit(p_fit, t_fit, y_fit, fit_on=fit_subset)
    threshold = choose_threshold(cal.transform(p_fit, t_fit), y_fit)
    cal.freeze()

    p_test, t_test, y_test = _step_targets(test_records, test_scores)
    if not p_test:
        raise ValueError("no labelled test data; Exp-0 needs the label-available CAMEL cell")
    p_cal = cal.transform(p_test, t_test)

    rel_raw = reliability(p_test, y_test)
    rel_cal = reliability(p_cal, y_test)
    a_raw = auroc(p_test, y_test)
    a_cal = auroc(p_cal, y_test)

    res = Exp0Result(
        fit_subset=fit_subset,
        test_subset=test_subset,
        n_fit=len(p_fit),
        n_test=len(p_test),
        auroc_raw=a_raw,
        auroc_cal=a_cal,
        ece_raw=rel_raw.ece,
        ece_cal=rel_cal.ece,
        brier_raw=rel_raw.brier,
        brier_cal=rel_cal.brier,
        threshold=threshold,
        reliability_raw=rel_raw.to_dict(),
        reliability_cal=rel_cal.to_dict(),
        provenance=cal.provenance,
    )
    res.transfers = (a_cal >= a_raw - MAX_AUROC_DROP) and (
        rel_cal.ece <= rel_raw.ece + MAX_ECE_INCREASE
    )

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cal.save(out / "calibrator_frozen.json")
        (out / "exp0.json").write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
        (out / "exp0.md").write_text(
            res.render()
            + "\n\n"
            + rel_raw.render("(raw)")
            + "\n\n"
            + rel_cal.render("(calibrated)")
            + "\n",
            encoding="utf-8",
        )
        (out / "threshold.json").write_text(
            json.dumps({"threshold": threshold, "chosen_on": fit_subset}, indent=2),
            encoding="utf-8",
        )
    return res, cal


def loo_calibrate(
    scores_by_key: dict[str, list],
    labels_by_key: dict[str, bool],
    *,
    method: str = "percentile",
) -> dict[str, list[float]]:
    """Pre-registered fallback: leave-one-file-out calibration on W&W.

    Used only if Exp-0 fails. Each file's steps are calibrated by a map fit on
    every *other* file, so no file contributes to its own calibration.
    """
    keys = list(scores_by_key)
    out: dict[str, list[float]] = {}
    for held in keys:
        p, t, y = [], [], []
        for k in keys:
            if k == held or k not in labels_by_key:
                continue
            for s in scores_by_key[k]:
                p.append(s.p_raw)
                t.append(s.type_norm)
                y.append(labels_by_key[k])
        if not p:
            out[held] = [s.p_raw for s in scores_by_key[held]]
            continue
        cal = TypedCalibrator(method=method).fit(p, t, y, fit_on="whowhen_loo")
        out[held] = cal.transform(
            [s.p_raw for s in scores_by_key[held]],
            [s.type_norm for s in scores_by_key[held]],
        )
    return out
