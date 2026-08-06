# masuq — MAS-UQ pilot

Uncertainty quantification over **frozen multi-agent-system trajectories**.

Four log formats load into one record schema; every step gets a normalised act
type; a judge scores each step prefix-conditionally with KV prefix sharing; one
per-type calibration map is fit once and frozen; the calibrated scores are then
read two ways — as **trajectory-level uncertainty** (noisy-OR) and as **failure
attribution** (first crossing).

Implements [`mas_uq_pilot_spec_v1.md`](mas_uq_pilot_spec_v1.md). Every module
docstring cites the spec section it implements.

> **Status: pilot / feasibility.** No data is committed and no numbers are
> claimed here. The repository is the harness; the numbers come from running it
> on data you supply.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest        # 80 tests, no data or GPU required
```

Optional extras: `.[judge]` for the local-LM judge (torch + transformers),
`.[openai]` for the gpt-4o baseline arm.

## Data

Nothing is bundled. Point the package at your copies of
[MAS-Trajectory-Understanding](https://github.com/) logs and the
[Who&When](https://github.com/) annotations — see [`data/README.md`](data/README.md)
for the expected layout — then:

```bash
export MASUQ_DATA_ROOT=/path/to/data
masuq paths          # resolve and verify every expected file
```

`masuq paths` exits non-zero and names what is missing, so it is the first thing
to run after downloading.

## The pipeline

```bash
masuq load --assert                       # loaders + flags + pre-registered counts
masuq typecheck --audit-out audit.json    # classifier vs native/parsed types
masuq smoke --n-steps 50                  # judge harness sanity check
masuq judge --subset autogen_mmlu --backend hf --model <id> --out-scores runs/autogen.jsonl
masuq judge --subset camel_math  --backend hf --model <id> --out-scores runs/camel.jsonl
masuq exp0 --fit-scores runs/autogen.jsonl --test-scores runs/camel.jsonl
masuq trajectory --scores runs/autogen.jsonl runs/camel.jsonl --calibrator runs/exp0/calibrator_frozen.json
masuq judge --subset alg --backend hf --model <id> --out-scores runs/alg.jsonl
masuq judge --subset hc  --backend hf --model <id> --out-scores runs/hc.jsonl
masuq attribution --scores runs/alg.jsonl runs/hc.jsonl \
    --calibrator runs/exp0/calibrator_frozen.json --threshold-file runs/exp0/threshold.json
masuq baselines --generators openai:gpt-4o hf:<id> --api-key sk-...
masuq audit --judges <j1> <j2> <j3>
```

Every command runs with `--backend mock` / `--generators mock` for a
dependency-free dry run of the whole pipeline.

## What each piece does

| Module | Spec | Purpose |
|---|---|---|
| `schema.py` | §0 | The unified record; validation hard-fails rather than coercing |
| `loaders/` | §1 | Four adapters, the label join, the pre-registered counts |
| `typing_/` | §2 | Rule classifier + its validation table against known types |
| `judge/` | §3 | Prefix-conditional P(True), evidence policy, surrogate baseline |
| `calibration.py` | §4 | Per-type maps, fit once, frozen, persisted |
| `aggregate.py` | §5 | Noisy-OR trajectory `U` and its ablations |
| `attribution.py` | §5 | First crossing, argmin, changepoint, agent-first |
| `metrics.py` | §6, §8 | AUROC / AUARC / reliability, bootstrap CIs, dual scorer |
| `experiments/` | §4–§7 | Exp-0 falsifier, both tracks, baselines, label audit |

## Design commitments worth knowing before you read the code

**Loaders hard-fail.** A silently repaired record is a provenance hole, and the
pilot's claims rest entirely on provenance. The MATU label join asserts
`task_id × run` alignment and raises on any unmatched key in either direction —
the spec says *asserted, not assumed*, and `--lenient-labels` exists only for
exploring a new dump.

**The classifier never overrides a known type.** Steps carrying `native` or
`parsed` types are untouched; the classifier fills in `classified` steps only.
Its agreement with the known types is a reportable table with a 90% gate, not an
assumption.

**Prefix sharing is structural, not an optimisation.** The judge interface is a
stateful `PrefixScorer` whose shared prefix only grows, so a `T`-step trajectory
costs `O(T)` prefix tokens. On Who&When's 130-step traces the quadratic version
is simply not runnable.

**Evidence never looks ahead.** The augmentation that rescues near-empty
`execute` steps pulls from the assigned subtask and *earlier* peer steps in the
same turn block. Letting a future step in would make the score non-causal and
inflate attribution accuracy for free.

**Calibration is fit once, then frozen.** `TypedCalibrator.freeze()` makes
refitting raise. The attribution threshold is chosen on the calibration corpus
and persisted alongside the maps; `masuq attribution` refuses to run without one
rather than picking a threshold on the corpus it is scoring.

**Exp-0 is a real falsifier.** It runs before any attribution number is seen,
with its pass gates pre-registered in the module. `masuq exp0` exits **2** when
calibration fails to transfer — a legitimate outcome that switches the
attribution track to the disclosed leave-one-out fallback.

**Both scorers, both flag sets, always.** Who&When numbers are reported four
ways: exact-match (primary) and the published substring scorer (comparability,
carrying its `"1" in "12"` artifact), each with and without the
`agent_step_mismatch`-flagged files. CIs bootstrap over *files*, since the file
is the sampling unit.

**Baselines get two judge arms.** Reproducing all_at_once / step_by_step /
binary_search with gpt-4o *and* with our judge model is what separates "their
method is weaker" from "their judge was stronger".

## Known limits

- The calibration fit target is per-*run* correctness propagated to each step. A
  correct run can contain a bad step, so the label is biased; this is why the v1
  default is a rank-preserving monotone map rather than Platt. Stated at the top
  of `calibration.py`.
- Plain noisy-OR saturates as `T` grows, so on 130-step HC traces it is near-1 by
  construction. Length-normalised and max variants are reported alongside it, and
  the trajectory track is validated on MATU, where `T` is short.
- The "intrinsic" baseline is a surrogate: frozen logs do not carry the
  generating model's distributions. It is reported as the only intrinsic-flavoured
  signal computable here and is expected to be weak.
- MATU labels are inherited. `masuq audit` re-labels 100 sampled runs with a
  3-judge pipeline to bound how far they can be trusted; it does not correct them.
- MATU-AutoGen `query` is null by construction (the log is keyed by task id). No
  v1 policy depends on it.
- StarAgent plan steps embed delegation payloads; v1 types the whole step as
  `plan`. Splitting plan from delegate is deferred.

## Layout

```
src/masuq/
  schema.py  config.py  cli.py
  loaders/   base.py matu_common.py matu_camel.py matu_autogen.py
             whowhen.py labels.py expectations.py
  typing_/   classifier.py validate.py
  judge/     backends.py prompts.py evidence.py harness.py surrogate.py
  calibration.py  aggregate.py  attribution.py  metrics.py
  experiments/ exp0_calibration_transfer.py exp_trajectory.py
               exp_attribution.py exp_baselines.py exp_label_audit.py
tests/         synthetic fixtures in all four source formats
```
