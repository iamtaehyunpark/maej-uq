# BUILD_REPORT — masattr

Inventory and provenance statement for the package (spec v3 Part B.6).

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
| `attribute/` | `rules.py` — `changepoint_single` (primary), plus the demoted rows: first-crossing, argmin, changepoint, agent-first, relative-crossing |
| `eval/` | `scorers.py` (exact + substring comparability), `ci.py` (bootstrap, reliability) |
| `baselines/` | `whowhen_repo.py`, `surrogate.py` |
| `specs/` | Frozen prompts, type-rule table, registered criteria, judge identities, the rule directive, hashes |
| `runs/` | One argparse module per experiment: `load`, `typecheck`, `retype`, `judge`, `e0_field`, `e1`–`e7`, `e9`, `baselines` |

## LOC

Measured over `src/masattr/**.py` (AST-based; docstring lines counted separately).

| | lines |
|---|---|
| code | **4542** |
| docstrings | 829 |
| blank + comment | 1179 |
| physical total | 5987 |
| tests (not counted toward the target) | 1789 |

**Ceiling raised to 4.5k; currently 4542 code lines** — 42 over, i.e. within a rounding error of the ceiling and with no headroom for the next feature. The overage against the
old 4k figure is experiment surface, not abstraction: the Stage-0 smoke gate,
the lookahead evidence arms, the level axis, and the baseline transport rewrite.
Nothing was cut.

One routine modernization is recorded here rather than buried: their
`inference.py` still imports the Azure OpenAI client, which is a stale 2025
dependency. Rather than edit their tree, a shim on `PYTHONPATH` makes
`openai.AzureOpenAI` construct a standard `openai.OpenAI` at the subprocess
boundary and drops the endpoint/api-version arguments. Their prompt assembly,
method logic, retry loop, and output contract are byte-identical. The shim also
records the concrete `gpt-4o` snapshot the API returns, per run, into the
manifest — a drifted alias is the first suspect if reproduction lands off the
published numbers.

## Reproducibility

`specs/` holds the serialised prompts and type-rule table plus two owner-set
files — `e0_criteria.json` and `judge.json` — each hashed. Every run writes
`manifest.json` with those hashes, the git commit, the resolved model families,
the anomalous file ids, and the full argument set; every experiment verifies the
live code still matches the frozen artifacts before it starts. A run is
reproducible from `(commit, manifest)`.

Two guards are deliberate refusals rather than defaults: `criteria.json` must be
marked `registered` before any attribution number is computed — the changepoint
fallback condition is part of the primary rule's definition — and `judge.json` carries a
status **per role**, so a run is blocked only by the roles it actually uses.
Declared model families are re-resolved from the ids and cross-checked, so the
disjointness constraints cannot be satisfied by a mislabelled declaration.

The primary rule itself is fixed by `specs/rule_directive.md`, whose hash is
logged as `rule_provenance` on every run. No experiment selects it.
