# masattr — MAS attribution harness

Failure attribution over **frozen multi-agent trajectories**. Given a Who&When
transcript of a multi-agent system that failed, locate the decisive mistake:
which agent, which step.

The method is a typed, externally-estimated per-step error field. A judge scores
every step prefix-conditionally; per-type calibration maps — fit once on a
single-agent corpus and frozen — turn those scores into probabilities; the
earliest step whose probability crosses the threshold is the attribution.

Implements [`docs/mas_attr_harness_spec_v2.md`](docs/mas_attr_harness_spec_v2.md).
Module docstrings cite the section they implement. The port receipts are in
[`PORT_REPORT.md`](PORT_REPORT.md).

> **Status: pilot.** No data is committed and no numbers are claimed here. The
> repository is the harness; the numbers come from running it on data you supply.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest        # 84 tests, no data / GPU / network required
```

Extras: `.[judge]` for the local-LM judge (torch + transformers), `.[openai]`
for the gpt-4o baseline arm.

## Data

Nothing is bundled. The release ships one parquet per subset:

```
<root>/who_and_when/Algorithm-Generated.parquet   # subset "alg", 126 rows
<root>/who_and_when/Hand-Crafted.parquet          # subset "hc",   58 rows
```

then `export MASATTR_DATA_ROOT=<root>` (or pass `--data-root`). A directory of
per-trajectory JSON is also accepted.

Verified against the release: 126 / 58 files, **4092 steps**, 3 + 3
`agent_step_mismatch` files, HC reaching 130 steps — every pre-registered count
in Part C §1 holds.

**One conflict the release forces.** Part C §1 asserts both "126 / 58 files" and
"`mistake_step` within bounds / every step has content", but 5 released files
violate the second: 3 HC files point past the end of their trajectory
(step 51 of 28, 8 of 5, 24 of 19) and 2 AG files contain an empty-content step.
Both asserts cannot hold. `--anomaly-policy` makes the choice explicit rather
than silent:

| policy | effect |
|---|---|
| `fail` (default) | refuses to load, naming the files — spec-literal |
| `flag` | keeps them, flags them, dual-reports them; counts hold |
| `drop` | excludes them; the 126/58 assert then fails |

`fail` is the default because the resolution is a pre-registration decision, not
a loader default.

## Running the manifest

```bash
masattr freeze                                     # hash prompts / type rules / type map
masattr load --assert                              # 126 / 58 / 4092 steps / 3+3 flagged
masattr typecheck --audit-out audit.json           # rules vs HC parsed types, ≥90% gate
masattr judge --judge hf:<id>                      # × readout × policy × GT setting
masattr e0 --paper1-scores p1.jsonl --scores runs/scores/*.jsonl
masattr e1 --scores runs/scores/*.jsonl --calibration src/masattr/calib/frozen/calibration.json
masattr baselines --generators openai:gpt-4o judge:hf:<id> --impl repo --repo-path <checkout>
masattr e2 / e3 / e4 / e5 / e6 / e7                # ablations
masattr e9 --e1-results runs/out/results.json      # stratification, no new runs
```

Everything runs end to end with `--judge mock` / `--generators mock` for a
dependency-free dry run. `masattr <cmd> --help` shows that experiment's flags
and nothing else.

## Layout

```
src/masattr/
  record.py       frozen record — every stage consumes and returns this
  paths.py manifest.py cli.py
  loaders/        whowhen_ag.py whowhen_hc.py _common.py
  typing/         normalize.py (HC parse + AG rules)  validate.py (the gate)
  judge/          client.py prompts.py score.py
  calib/          fit.py apply.py frozen/
  attribute/      rules.py
  eval/           scorers.py ci.py
  baselines/      whowhen_repo.py surrogate.py
  specs/          frozen prompts, type rules, type map, hashes
  runs/           one file per experiment, argparse only
```

## Design commitments

**Loaders hard-fail.** An uncastable `mistake_step`, an out-of-range one, an
empty step — all raise. v1 flagged and carried on; v2 does not, and that is
right for an attribution-only harness: the annotated step *is* the label, so a
file whose label cannot be read is not a datapoint.

**Records are frozen.** A record is a loaded fact about a trajectory. Scores
live in separate arrays keyed by `(file_id, step_idx)`; no stage edits a record.

**Types are parsed where possible, classified only where necessary.** HC's
compound role encodes the act, so HC types are read, not guessed — which is what
makes HC the reference corpus for validating the AG rules. The rules never
override a parsed type, and `masattr typecheck` exits non-zero below 90%
agreement.

**Prefix sharing is structural.** The judge client's shared prefix only grows,
so a `T`-step trajectory costs `O(T)` prefix tokens. HC reaches 130 steps; the
quadratic path is not runnable. `score_record` raises if handed a client that
does not expose the shared-prefix path.

**Evidence never looks ahead.** The rescue for near-empty `execute` steps pulls
the assigned subtask and *earlier* same-turn peers. A future step in the
evidence would make the score non-causal and inflate attribution for free. The
one deliberate exception is `--policy hindsight`, which is the E5 ceiling, not a
method.

**One prompt scaffold, three readouts.** Logit P(True), verbalized number, and
binary verdict share the preamble and the question; only the final instruction
differs. That is what makes E2 an ablation rather than three methods.

**Calibration is fit once, frozen, and hash-checked.** `FrozenCalibration.load`
refuses a file whose `content_hash` no longer matches its contents. The
first-crossing threshold is chosen on the *fitting* corpus and travels with the
maps; `masattr e1` refuses to run without one rather than picking a threshold on
the corpus it is scoring.

**E0 is a real falsifier.** It runs first, its gates are pre-registered in the
module, and it exits **2** when transfer fails — a legitimate outcome that puts
the disclosed leave-one-out fallback in force and weakens the uniformity claim
in the paper text.

**Ablations refuse single arms.** `masattr e2` stops if every input score file
has the same readout. A one-row ablation is not an ablation.

**Both scorers, every slice, always.** Exact match is primary; the substring
scorer reproduces the published regime and carries its artifact (predicted step
`1` scores as a hit against gold `12`). Tables are dual-reported with and
without the 6 flagged files and the held-aside 20. CIs bootstrap over *files*.

**Runs are reproducible from `(commit, manifest)`.** Every experiment verifies
the live prompts and type rules still match `specs/` before it starts.

## Known limits and stated assumptions

- **Who&When has no per-step labels.** It annotates one decisive mistake per
  trajectory. Reliability diagrams need per-step correctness, so it is *derived*:
  the default `prefix` policy uses steps `0..mistake_step` with
  `correct = idx < mistake_step`, excluding the post-mistake tail rather than
  guessing it. The `point` policy keeps every step and asserts the tail is fine.
  Documented at the top of `calib/apply.py`; both are pre-registered.
- **E0 needs paper 1's corpus scored by this judge** — a JSONL of
  `{p_raw, type, correct}`. Without it `masattr e0` stops. It will not quietly
  calibrate on Who&When, which is the very thing the fallback is meant to
  disclose.
- **The paper-1 type map is data, not code.** It lives in
  `specs/paper1_type_map.json`, is hashed into every manifest, and unmapped
  source types become `unknown` and are counted — never silently dropped.
- **The surrogate baseline is a surrogate.** Frozen logs do not carry the
  generating model's distributions. A proxy LM's logprob and entropy are the
  closest computable thing, they are uncalibrated, and they are expected to be
  weak. Read the argmin row.
- **`--impl local` for the baselines is not the reproduction.** It re-prompts the
  three strategies so the pipeline runs without their checkout; every row is
  stamped `impl=local` and the manifest says so.
- **E8 (success-control) is not built.** Part D gates it behind an explicit owner
  decision.
- **The port source was substituted.** Paper 1's harness was unavailable; the
  allowlisted components were re-derived from the v1 package. See PORT_REPORT.
