# PORT_REPORT — masattr

Deletion receipts for the v1 → v2 rescope (spec v2 Part B rule 7).

## What happened

Spec v2 rescopes the project to **attribution only**. The v1 package (`masuq`)
implemented both tracks; it was removed wholesale rather than trimmed, and
`masattr` was started empty and filled by allowlist. The v1 code remains in git
history at commit `fea1a58` if any of it is ever wanted back.

## Port source

Spec v2 Part B names paper 1's harness as the port source. That checkout was not
available in this environment, so the allowlisted components were re-derived
from the v1 package written against pilot spec v1 — which is itself the paper-1
harness re-templated for MAS steps. **This is a substitution, not the port the
spec describes.** If the paper-1 repo differs materially in judge-client
internals, prompt assembly, or the calibration serialisation format, those three
modules should be re-checked against it.

## Pulled in (file → origin)

| File | Origin | Notes |
|---|---|---|
| `judge/client.py` | paper-1 judge client (via `masuq/judge/backends.py`) | model load, logit extraction for P(True), KV prefix-sharing path |
| `judge/prompts.py` | paper-1 prompt assembly (via `masuq/judge/prompts.py`) | re-templated for MAS steps; with-GT setting and three readouts added |
| `judge/score.py` | paper-1 scoring loop (via `masuq/judge/harness.py`, `evidence.py`) | prefix-conditional loop, evidence policies, cost logging |
| `calib/fit.py` | paper-1 calibration module (via `masuq/calibration.py`) | per-type fit/apply, frozen-map serialisation + content hash |
| `calib/apply.py` | new | held-aside slice, derived step labels, LOO fallback |
| `eval/ci.py` | paper-1 bootstrap-CI + reliability-diagram utilities (via `masuq/metrics.py`) | AUROC kept for E0 only |
| `eval/scorers.py` | re-implemented | substring scorer written from the four lines in their `evaluate.py`; their eval path is **not** imported |
| `loaders/whowhen_{ag,hc}.py`, `loaders/_common.py` | v1 `masuq/loaders/whowhen.py`, split by subset | hard-fail replaces v1's flag-and-continue on `mistake_step` cast |
| `typing/normalize.py`, `typing/validate.py` | **new code** | MAS act-typing; paper-1's τ rules were environment-consequence based and were not adapted |
| `attribute/rules.py` | v1 `masuq/attribution.py` | `agent_first` re-specified to v2's per-agent-max selector |
| `baselines/whowhen_repo.py` | new wrapper | imports their `inference.py`; nothing patched but credentials |
| `baselines/surrogate.py` | v1 `masuq/judge/surrogate.py` | now feeds the same attribution rules (E7) |
| `record.py`, `paths.py`, `manifest.py`, `cli.py`, `runs/*` | new | frozen record, run manifest, one file per experiment |

## Explicitly NOT ported

From the v2 delete-list (Part A) — all of these existed in v1 and were removed:

- ✗ MATU loaders, both cells (`matu_camel.py`, `matu_autogen.py`, `matu_common.py`)
- ✗ `accuracy_dict_*.pkl` label joins and per-run correctness plumbing (`loaders/labels.py`)
- ✗ noisy-OR aggregation, trajectory-level `U`, AUROC/AUARC-vs-completion (`aggregate.py`, `experiments/exp_trajectory.py`)
- ✗ MATU published-numbers comparability tables
- ✗ all cross-step aggregation beyond what the attribution rules consume pointwise
- ✗ the MATU label spot-audit (`experiments/exp_label_audit.py`) — it audited labels that no longer exist here
- ✗ the success-control adapter (Part D) — **gated**, builds only on an explicit go

From the paper-1 exclusion list (Part B rule 2): environment wrappers, agent
generation, trajectory regeneration, τ action-typing rules, evidence-grid decoy
apparatus, loop-stratification analysis, noisy-OR combiner, judge-size curve
scaffolding, and the hindsight harness beyond the single context-swap flag —
none were pulled in. The one retained context swap is `policy="hindsight"` in
`judge/score.py`, used for the E5 ceiling figure.

Also cut during the diet, as abstraction rather than feature: every
`__init__.py` re-export wall. Import from the concrete module.

## LOC

Measured over `src/masattr/**.py` (AST-based; docstring lines counted separately).

| | lines |
|---|---|
| code | **3124** |
| docstrings | 576 |
| blank + comment | 883 |
| physical total | 4583 |
| tests (not counted toward the target) | 897 |

**Over the ~2.5k target by ~620 code lines.** Cutting the re-export walls
removed ~210. What remains is features, not abstraction, so per Part B rule 7 it
was left in place. The honest options if the target is hard:

- `baselines/whowhen_repo.py` `--impl local` (~110 lines): the three strategies
  re-prompted so the pipeline runs without their checkout. Not the reproduction;
  deletable if you always have the repo.
- `calib/fit.py` `platt` + `isotonic` (~60 lines): `percentile` is the default
  and the other two are never on the primary path.
- `paths.py` `ALTERNATES` (~15 lines): tolerance for renamed dumps.
- Mock client / mock proxy LM (~60 lines): required for the dependency-free test
  suite and dry runs; deleting them makes CI need a GPU.

None of these is an abstraction; each is a capability. Say the word on which to
drop.

## Reproducibility

`specs/` holds the serialised prompts, type-rule table, and paper-1 type map,
each hashed. Every run writes `manifest.json` with those hashes, the calibration
content hash, the git commit, and the full argument set — and every experiment
verifies the live code still matches the frozen artifacts before it starts. A
run is reproducible from `(commit, manifest)`.
