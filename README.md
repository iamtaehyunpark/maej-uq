# masattr — MAS attribution harness

Failure attribution over **frozen multi-agent trajectories**. Given a Who&When
transcript of a multi-agent system that failed, locate the decisive mistake:
which agent, which step.

The method is a typed, externally-estimated per-step error field. A judge scores
every step prefix-conditionally; per-type statistics — fit by leave-one-file-out
CV within each subset — put those scores on one scale; the decisive step is
localised by reading the normalized field pointwise. Nothing is aggregated
across steps.

Implements [`docs/mas_attr_harness_spec_v2.md`](docs/mas_attr_harness_spec_v2.md)
as amended by
[`docs/mas_attr_harness_spec_v2_1_severance.md`](docs/mas_attr_harness_spec_v2_1_severance.md).
Module docstrings cite the section they implement. Inventory and provenance are
in [`BUILD_REPORT.md`](BUILD_REPORT.md).

> **Status: pilot.** No data is committed and no numbers are claimed here. The
> repository is the harness; the numbers come from running it on data you supply.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest        # 144 tests, no data / GPU / network required
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
| `flag` (default) | keeps them, flags them with a class distinct from the six `agent_step_mismatch` files, dual-reports them; counts hold |
| `fail` | refuses to load, naming the files |
| `drop` | excludes them; the 126/58 assert then fails |

Part C §1 is amended accordingly: hard-fail on count mismatch, flag on
record-level anomalies. The five file ids are logged in every run manifest.

## Running the manifest

```bash
masattr freeze                                     # hash prompts, type rules, and both spec files
masattr load --assert                              # 126 / 58 / 4092 steps / 3+3 flagged
masattr typecheck --audit-out audit.json           # rules vs HC parsed types, ≥90% gate
masattr retype --splitter hf:<id> --judge hf:<id>  # gate + apply the plan/delegate splitter
masattr judge --judge hf:<id>                      # × readout × policy × GT setting
masattr e0 --scores runs/scores/*.jsonl            # field sanity, threshold stability, primary rule
masattr e1 --scores runs/scores/*.jsonl --folds runs/normalize/folds.json \
           --decision runs/out/e0_decision.json
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
  typing/         normalize.py (HC parse + AG rules)  validate.py (gate)  refine.py (splitter)
  judge/          client.py prompts.py score.py
  normalize/      fit.py (leave-one-out CV)  apply.py (z-scoring, field sanity)
  attribute/      rules.py
  eval/           scorers.py ci.py
  baselines/      whowhen_repo.py surrogate.py
  specs/          frozen prompts, type rules, E0 criteria, judge ids, hashes
  runs/           one file per experiment, argparse only
```

## Design commitments

**Loaders hard-fail.** An uncastable `mistake_step`, an out-of-range one, an
empty step — all raise. v1 flagged and carried on; v2 does not, and that is
right for an attribution-only harness: the annotated step *is* the label, so a
file whose label cannot be read is not a datapoint.

**Records are frozen.** A record is a loaded fact about a trajectory. Scores
live in separate arrays keyed by `(file_id, step_idx)`; no stage edits a record.

**Typing is hierarchical, because the release says it has to be.** HC's compound
role encodes the act, so HC types are read, not guessed — which makes HC the
reference for validating the AG rules. Measured there, the rules split
coordination / execute / final at **0.9935** but plan vs delegate at **0.4162**,
below the 0.6934 majority-class baseline: in the Magentic-One idiom that
distinction lives in the role, not the text. So the rules keep the coarse split
and an LLM classifier takes only the plan/delegate sub-split
(`masattr retype`), gated on HC and required to beat both 0.90 *and* the
majority baseline before it may touch AG. Collapsing plan+delegate would have
scored 0.994 by deleting the delegation-error prediction; tuning the rules on
HC's ledger markers would license AG typing on HC's idiom, which is the
circularity the gate exists to prevent.

**Two family-disjointness constraints, checked in code.** The judge must be
disjoint from any labeling judge, and the type-classifier must be disjoint from
the judge — typing conditions the judge's evidence policy. `masattr retype`
refuses a same-family splitter, and every manifest logs the resolved families.

**Prefix sharing is structural.** The judge client's shared prefix only grows,
so a `T`-step trajectory costs `O(T)` prefix tokens. HC reaches 130 steps; the
quadratic path is not runnable. `score_record` raises if handed a client that
does not expose the shared-prefix path.

**Evidence never looks ahead.** The rescue for near-empty `execute` steps pulls
the assigned subtask and *earlier* same-turn peers. A future step in the
evidence would make the score non-causal and inflate attribution for free. The
one deliberate exception is `--policy hindsight`, which is the E5 ceiling, not a
method. The subtask pointer, peer corroboration, and prefix window are
separately switchable, because §7(iv) ablates them separately.

**Long logs truncate under a pre-registered policy, not under context pressure.**
Type-aware retention: query, ground truth, and every plan/delegate step stay
verbatim; the newest execute steps stay verbatim; over budget, the *oldest*
execute steps are demoted to a one-line header — the row survives, the detail
does not. On the real HC subset this fires on **24.1% of trajectories and 13.8%
of assessments**, which is a limitations sentence, not a surprise, because
`cost_summary` reports it every run.

**One prompt scaffold, three readouts.** Logit P(True), verbalized number, and
binary verdict share the preamble and the question; only the final instruction
differs. That is what makes E2 an ablation rather than three methods.

**Normalization is leave-one-file-out, so nothing is scored under statistics
that saw it.** Per-type mean/sd and the crossing threshold are fit on every
*other* file in the subset; each file is z-scored under its own fold. Subsets
are normalized independently. Thresholds are fit **per type**, with the pooled
threshold as fallback and as E4's global-threshold arm. `masattr e1` refuses to
run without folds rather than picking a threshold on the corpus it is scoring,
and the raw score stays on every row beside `p_norm`.

**E0 fixes the primary rule before it can see the outcome.** It asks two
questions — is there a field to localize (per-type distributions, plus
constant / saturated / near-binary checks), and is the threshold stable across
folds — and reads its decision bound from `specs/e0_criteria.json`, which must
be marked `registered` before it will run. If the worst cross-fold threshold CV
exceeds the bound, the primary rule switches to the threshold-free set
(`relative_crossing`, argmin, changepoint) and first-crossing demotes to an
ablation. The decision and the criterion hash go in the manifest, and `masattr
e1 --decision` reads the rule from the file rather than defaulting.

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
  Documented at the top of `normalize/fit.py`; both are pre-registered.
- **Two spec files start as drafts on purpose.** `specs/e0_criteria.json` must
  be `registered` before E0 runs, and `specs/judge.json` must be `confirmed`
  before any reported run. Both are owner-set; the code refuses rather than
  defaults.
- **The surrogate baseline is a surrogate.** Frozen logs do not carry the
  generating model's distributions. A proxy LM's logprob and entropy are the
  closest computable thing, they are uncalibrated, and they are expected to be
  weak. Read the argmin row.
- **`--impl local` for the baselines is not the reproduction.** It re-prompts the
  three strategies so the pipeline runs without their checkout; every row is
  stamped `impl=local` and the manifest says so.
- **E8 (success-control) is not built.** Part D gates it behind an explicit owner
  decision.
- **The package is over the ≤2.5k LOC target** at ~3.9k code lines. BUILD_REPORT
  lists what would come out first if the target is hard; the overage is
  experiment surface, not abstraction.
