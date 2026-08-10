"""Assemble the pilot baseline master table from the individual run outputs.

Reads the artifacts each row already wrote — it recomputes nothing — so the
table cannot disagree with the runs that produced it. Rows that did not run are
printed as such rather than omitted, because a missing row and a zero row are
different claims.

Usage: python tools/master_table.py <runs-root> [> report.md]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RULES = ("changepoint_single", "first_crossing", "argmin", "relative_crossing@2.0")
SCORERS = ("exact", "tol1", "tol2", "substring")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cell(sc: dict | None) -> str:
    if not sc:
        return "—"
    ci = sc.get("agent_ci")
    return f"{sc['agent_acc']:.3f}" + (f" [{ci['lo']:.3f},{ci['hi']:.3f}]" if ci else "")


def field_rows(root: Path, name: str, e1_dirs: dict[str, Path]) -> list[str]:
    """One block of rows for a scored field, per GT arm and subset."""
    out = []
    for gt, d in sorted(e1_dirs.items()):
        res = load(d / "results.json")
        if not res:
            out.append(f"| {name} | {gt} | — | *not run* | | | |")
            continue
        for label, cfg in sorted(res["configs"].items()):
            subset = [p for p in label.split(" · ") if p.startswith("subset=")][0].split("=")[1]
            fb = cfg.get("primary_fallback", {})
            for rule in RULES:
                sc = cfg["scores"].get(rule)
                if not sc:
                    continue
                a = sc.get("exact/all")
                s_step = a and a.get("step_ci")
                out.append(
                    f"| {name} | {gt} | {subset} | {rule} | {cell(a)} | "
                    + (f"{a['step_acc']:.3f} [{s_step['lo']:.3f},{s_step['hi']:.3f}]" if a else "—")
                    + f" | {fb.get('rate', float('nan')):.1%} |"
                )
    return out


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "runs")
    lines: list[str] = [
        "# Pilot baseline suite — master table",
        "",
        "Agent and step accuracy, exact scorer, slice = all, file-level bootstrap CIs.",
        "Fallback is the primary rule's rate of falling back to argmin.",
        "",
        "| row | GT | subset | rule | agent | step | fallback |",
        "|---|---|---|---|---|---|---|",
    ]

    # B0 reference row
    lines += field_rows(
        root, "B0 P(True)/W0", {"off": root / "main/e1_nogt", "on": root / "main/e1_gt"}
    )

    # B1 direct rows
    b1 = load(root / "base/b1/results.json")
    if b1:
        for key, variants in sorted(b1["rows"].items()):
            subset, _, name = key.partition("/")
            sc = variants.get("exact/all") or variants.get("expectation/all")
            if not sc:
                continue
            st = sc.get("step_ci")
            lines.append(
                f"| B1 {name} | — | {subset} | *direct* | {cell(sc)} | "
                + (f"{sc['step_acc']:.3f} [{st['lo']:.3f},{st['hi']:.3f}]" if st else "—")
                + " | — |"
            )
    else:
        lines.append("| B1 | — | — | *not run* | | | |")

    # B2 capability control
    b2 = load(root / "base/b2/results.json")
    if b2:
        for run in b2["runs"]:
            sc = run["scores"].get("exact/all")
            st = sc and sc.get("step_ci")
            lines.append(
                f"| B2 {run['method']} | — | {run['subset']} | *direct* | {cell(sc)} | "
                + (f"{sc['step_acc']:.3f} [{st['lo']:.3f},{st['hi']:.3f}]" if sc else "—")
                + " | — |"
            )
    else:
        lines.append("| B2 | — | — | *not run* | | | |")

    # B3 readout variants
    for ro in ("verbalized", "binary"):
        lines += field_rows(
            root,
            f"B3 {ro}",
            {"off": root / f"base/b3/e1_{ro}_nogt", "on": root / f"base/b3/e1_{ro}_gt"},
        )

    # B4 coherence fields
    for fld in ("embed_divergence", "nli_contradiction"):
        lines += field_rows(root, f"B4 {fld}", {"off": root / f"base/b4/e1_{fld}_nogt"})

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
