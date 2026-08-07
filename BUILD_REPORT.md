# BUILD_REPORT — masattr

Inventory and provenance statement for the package (spec v2.1 §1.6).

## Provenance

**No code was vendored from any prior project tree.** Every module here was
written for this repository. The judge client, prompt assembly, bootstrap-CI and
reliability utilities are generic implementations written plainly against their
own requirements; no external project's paths, corpora, results, or internal
names appear in code, comments, specs, or reports.

The one external dependency of record is the Who&When baseline repository, which
is used as a **dependency, not a fork** (`baselines/whowhen_repo.py`): its
`inference.py` is imported and its three functions called, with nothing patched
except credentials arriving via CLI flags. Its `evaluate.py` is deliberately not
imported — the substring scorer is re-implemented in `eval/scorers.py` so the
comparability row is a choice rather than an inheritance.

## Inventory

| Module | Purpose |
|---|---|
| `record.py` | Frozen unified record; every stage consumes and returns it |
| `paths.py` | Data resolution (parquet release, or per-file JSON) |
| `manifest.py` | Run manifest: commit, spec hashes, model families, anomaly ids |
| `models.py` | Model-family resolution and the disjointness constraints |
| `cli.py` | Subcommand routing only |
| `loaders/` | `whowhen_ag.py`, `whowhen_hc.py`, `_common.py` — counts and flags |
| `typing/` | `normalize.py` (HC role parsing + AG rules), `validate.py` (the gate), `refine.py` (hierarchical plan/delegate splitter) |
| `judge/` | `client.py` (KV prefix sharing), `prompts.py` (one scaffold, three readouts), `score.py` (scoring loop, evidence policies, truncation) |
| `normalize/` | `fit.py` (per-type statistics + thresholds by leave-one-file-out CV), `apply.py` (z-scoring, field sanity, stability) |
| `attribute/` | `rules.py` — first-crossing, argmin, changepoint, agent-first, relative-crossing |
| `eval/` | `scorers.py` (exact + substring comparability), `ci.py` (bootstrap, reliability) |
| `baselines/` | `whowhen_repo.py`, `surrogate.py` |
| `specs/` | Frozen prompts, type-rule table, E0 criteria, judge identities, hashes |
| `runs/` | One argparse module per experiment: `load`, `typecheck`, `retype`, `judge`, `e0_field`, `e1`–`e7`, `e9`, `baselines` |

## LOC

Measured over `src/masattr/**.py` (AST-based; docstring lines counted separately).

| | lines |
|---|---|
| code | **3926** |
| docstrings | 781 |
| blank + comment | 1144 |
| physical total | 5851 |
| tests (not counted toward the target) | 1662 |

**Over the ≤2.5k target by ~1.4k code lines.** The re-export walls were already
cut; what remains is features. Largest modules: `judge/score.py` (334),
`runs/_shared.py` (329), `attribute/rules.py` (238), `normalize/fit.py` (216).

Honest options if the target is hard, in the order I would cut them:

- `baselines/whowhen_repo.py` `--impl local` (~110 lines): the three strategies
  re-prompted so the pipeline runs without their checkout. Not the reproduction;
  deletable if the checkout is always present.
- `runs/e2/e3/e4/e5/e6` (~50 lines total): five thin argparse wrappers over one
  shared body. They exist because the spec asks for one `run.py` per experiment;
  a single `masattr ablate --axis` would replace them.
- Mock client / mock proxy LM / mock splitter (~90 lines): required for the
  dependency-free test suite and dry runs; deleting them makes CI need a GPU.
- `paths.py` `ALTERNATES` (~20 lines): tolerance for differently-laid-out copies.

None of these is an abstraction; each is a capability. The ~1.4k overage is
mostly the experiment surface the manifest asks for (E0–E9, four rules, three
readouts, four evidence axes, two normalization arms, two scorers, four slices),
and cutting it would cut experiments.

## Reproducibility

`specs/` holds the serialised prompts and type-rule table plus two owner-set
files — `e0_criteria.json` and `judge.json` — each hashed. Every run writes
`manifest.json` with those hashes, the git commit, the resolved model families,
the anomalous file ids, and the full argument set; every experiment verifies the
live code still matches the frozen artifacts before it starts. A run is
reproducible from `(commit, manifest)`.

Two guards are deliberate refusals rather than defaults: `e0_criteria.json` must
be marked `registered` before E0 runs (otherwise the decision rule could be
chosen after the outcome), and `judge.json` must be `confirmed` before any
reported run.
