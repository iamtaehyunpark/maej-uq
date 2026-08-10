# E0 + E1 results — reference row

> **Generated file.** Rebuild with `python tools/e1_report.py runs <data-root>`. Do not hand-edit: the previous hand-written version went stale when the readout scaffold changed and had to be superseded.

The B0 reference row of the pilot suite, reported in full: every rule still reported (the four in the master table plus `changepoint` and the `relative_crossing` k-sweep), both GT settings, every pre-registered slice. `agent_first` is withdrawn — see `rules.WITHDRAWN`.

Run provenance: commit `fd8f3e930e9985935d9681d38563c5b95baaceaf`, rule directive hash `cdcb43b542297e70`, prompts `e8bc3b7bb8f22151`, type rules `434c9b068a083738`, criteria `0e172a4c4d77fc09`, judge `6fd7662046af1ae1`.

Evidence arm **W0** (prefix-conditional, no lookahead). Anomaly policy `flag`. Normalization: per-type leave-one-file-out CV, fit separately per subset and per GT setting. E0 is sanity-only and gates nothing — the primary rule is fixed by `specs/rule_directive.md`.
## 1. Scoring pass

| subset | GT | assessments | wall-clock | truncated assessments | truncated trajectories | max prefix | parse failures |
|---|---|---|---|---|---|---|---|
| hc | gt | 1,541 | 432 s | 10.5% | 17.1% | 25,166 tok | 0 |

> Cost figures cover only the trajectories judged in each run's **final** invocation. The scoring pass was resumable and was resumed, so trajectories served from cache report no timing. Row counts on disk are complete (1,099 alg / 2,993 hc per setting); the wall-clock column is not a total for the whole corpus.

## 2. E0 — score field (GT off)

| cell | n | mean | sd | p05 | median | p95 | saturated | distinct | AUROC vs derived labels |
|---|---|---|---|---|---|---|---|---|---|
| alg/delegate | 13 | 0.317 | 0.296 | 0.004 | 0.245 | 0.848 | 0.0% | 13 | 0.583 |
| alg/execute | 818 | 0.338 | 0.337 | 0.003 | 0.202 | 0.974 | 2.8% | 118 | 0.565 |
| alg/final | 253 | 0.288 | 0.273 | 0.002 | 0.223 | 0.842 | 2.8% | 79 | 0.328 |
| alg/plan | 13 | 0.556 | 0.328 | 0.094 | 0.706 | 0.952 | 0.0% | 12 | 0.556 |
| alg/unknown | 2 | 0.043 | 0.010 | 0.034 | 0.043 | 0.052 | 0.0% | 2 | — |
| hc/delegate | 689 | 0.544 | 0.329 | 0.042 | 0.562 | 0.988 | 0.1% | 94 | 0.474 |
| hc/execute | 689 | 0.361 | 0.332 | 0.002 | 0.269 | 0.937 | 3.2% | 104 | 0.608 |
| hc/final | 56 | 0.185 | 0.231 | 0.005 | 0.057 | 0.637 | 1.8% | 39 | 0.778 |
| hc/plan | 1,559 | 0.417 | 0.298 | 0.033 | 0.349 | 0.967 | 0.0% | 94 | 0.689 |

**Degenerate cells: none.**

Cells with n < 20 are flagged undersized by the fitter and are not read as evidence: `alg/delegate` (n=13), `alg/plan` (n=13), `alg/unknown` (n=2).

## 3. E0 — cross-fold threshold stability

| GT | subset | folds | global threshold mean | sd | CV | range |
|---|---|---|---|---|---|---|
| nogt | alg | 126 | +2.068 | 0.012 | **0.006** | [+2.047, +2.114] |
| nogt | hc | 58 | -1.252 | 0.016 | **0.013** | [-1.311, -1.217] |
| gt | alg | 126 | -0.259 | 0.893 | **1.135** | [-0.638, +2.122] |
| gt | hc | 58 | -1.178 | 0.011 | **0.009** | [-1.219, -1.155] |

Worst per-type threshold CV — nogt: `hc/delegate` 0.291; gt: `alg/global` 1.135.

## 4. E1 — primary rule fallback

`changepoint_single` falls back to argmin on boundary splits, contrast below the registered bound (z units), or trajectories too short to split.

| subset | GT off | GT on | reasons (GT off) |
|---|---|---|---|
| alg (n=126) | **74.6%** | 69.8% | boundary 67, low_contrast 27, regime_found 32 |
| hc (n=58) | **5.2%** | 6.9% | boundary 3, regime_found 55 |

Fallback against trajectory length, pooled across subsets (GT off):

| steps | n | fallback |
|---|---|---|
| < 10 | 61 | 82.0% |
| 10–20 | 78 | 60.3% |
| 20–50 | 22 | 0.0% |
| 50+ | 23 | 0.0% |

## 5. E1 — attribution accuracy, exact scorer, all files

Bootstrap CIs over files, 2,000 resamples. **Bold** = registered primary.

### GT off

| rule | alg agent | alg step | hc agent | hc step |
|---|---|---|---|---|
| **changepoint_single** | **0.333 [0.246, 0.413]** | **0.190 [0.127, 0.254]** | **0.483 [0.345, 0.621]** | **0.103 [0.034, 0.190]** |
| first_crossing | 0.500 [0.413, 0.587] | 0.159 [0.095, 0.222] | 0.534 [0.397, 0.655] | 0.086 [0.017, 0.155] |
| argmin | 0.333 [0.254, 0.413] | 0.175 [0.111, 0.246] | 0.397 [0.276, 0.517] | 0.086 [0.017, 0.172] |
| changepoint | 0.238 [0.167, 0.317] | 0.111 [0.056, 0.167] | 0.362 [0.241, 0.483] | 0.103 [0.034, 0.190] |
| relative_crossing@1.5 | 0.333 [0.254, 0.413] | 0.183 [0.119, 0.254] | 0.397 [0.276, 0.517] | 0.069 [0.017, 0.138] |
| relative_crossing@2.0 | 0.333 [0.254, 0.413] | 0.175 [0.111, 0.246] | 0.397 [0.276, 0.517] | 0.086 [0.017, 0.172] |
| relative_crossing@2.5 | 0.333 [0.254, 0.413] | 0.175 [0.111, 0.246] | 0.397 [0.276, 0.517] | 0.086 [0.017, 0.172] |

### GT on

| rule | alg agent | alg step | hc agent | hc step |
|---|---|---|---|---|
| **changepoint_single** | **0.341 [0.262, 0.429]** | **0.206 [0.143, 0.278]** | **0.517 [0.397, 0.638]** | **0.103 [0.034, 0.190]** |
| first_crossing | 0.421 [0.333, 0.508] | 0.254 [0.183, 0.333] | 0.448 [0.328, 0.586] | 0.052 [0.000, 0.121] |
| argmin | 0.357 [0.270, 0.444] | 0.214 [0.151, 0.286] | 0.431 [0.310, 0.552] | 0.103 [0.034, 0.190] |
| changepoint | 0.238 [0.167, 0.310] | 0.095 [0.048, 0.151] | 0.414 [0.276, 0.534] | 0.086 [0.017, 0.172] |
| relative_crossing@1.5 | 0.357 [0.270, 0.444] | 0.222 [0.151, 0.302] | 0.431 [0.310, 0.552] | 0.086 [0.017, 0.172] |
| relative_crossing@2.0 | 0.357 [0.270, 0.444] | 0.214 [0.151, 0.286] | 0.431 [0.310, 0.552] | 0.103 [0.034, 0.190] |
| relative_crossing@2.5 | 0.357 [0.270, 0.444] | 0.214 [0.151, 0.286] | 0.431 [0.310, 0.552] | 0.103 [0.034, 0.190] |

## 6. E1 — orchestrator vs worker fault (changepoint_single, exact scorer)

| GT | subset | fault | n | agent | step |
|---|---|---|---|---|---|
| nogt | alg | worker | 126 | 0.333 [0.254, 0.413] | 0.190 [0.127, 0.262] |
| nogt | hc | orchestrator | 18 | 0.611 [0.389, 0.833] | 0.056 [0.000, 0.167] |
| nogt | hc | worker | 40 | 0.425 [0.275, 0.575] | 0.125 [0.025, 0.250] |
| gt | alg | worker | 126 | 0.341 [0.254, 0.421] | 0.206 [0.135, 0.278] |
| gt | hc | orchestrator | 18 | 0.611 [0.389, 0.833] | 0.056 [0.000, 0.167] |
| gt | hc | worker | 40 | 0.475 [0.325, 0.625] | 0.125 [0.025, 0.250] |

## 7. E1 — dual reporting across slices (changepoint_single)

| GT | subset | column | all | excl_flagged | excl_anomalous | excl_all_excluded |
|---|---|---|---|---|---|---|
| nogt | alg | agent | 0.333 (126) | 0.341 (123) | 0.339 (124) | 0.347 (121) |
| nogt | alg | step | 0.190 (126) | 0.195 (123) | 0.194 (124) | 0.198 (121) |
| nogt | hc | agent | 0.483 (58) | 0.509 (55) | 0.491 (55) | 0.519 (52) |
| nogt | hc | step | 0.103 (58) | 0.109 (55) | 0.109 (55) | 0.115 (52) |
| gt | alg | agent | 0.341 (126) | 0.341 (123) | 0.347 (124) | 0.347 (121) |
| gt | alg | step | 0.206 (126) | 0.203 (123) | 0.210 (124) | 0.207 (121) |
| gt | hc | agent | 0.517 (58) | 0.509 (55) | 0.527 (55) | 0.519 (52) |
| gt | hc | step | 0.103 (58) | 0.109 (55) | 0.109 (55) | 0.115 (52) |

## 8. E1 — normalized position of the attributed step (changepoint_single)

Where the rule places its pick, against where gold sits, as a fraction of trajectory length.

| GT | subset | mean predicted | mean gold | pred in [0,0.2) | gold in [0,0.2) |
|---|---|---|---|---|---|
| nogt | alg | 0.408 | 0.369 | 27.8% | 34.1% |
| nogt | hc | 0.323 | 0.435 | 44.8% | 29.1% |
| gt | alg | 0.406 | 0.369 | 24.6% | 34.1% |
| gt | hc | 0.339 | 0.435 | 46.6% | 29.1% |

## 9. Related documents

| doc | scope |
|---|---|
| `docs/RESULTS.md` | the current consolidated report |
| `runs/base/MASTER.md` | all rows, four rules, one table |
| `runs/base/TOPK.md` | recall@k over the score ranking |

