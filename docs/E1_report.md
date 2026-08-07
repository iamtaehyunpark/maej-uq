# E0 + E1 results

**2026-08-07 · Who&When, 184 files · judge `Qwen/Qwen3.6-35B-A3B` (vLLM 0.23, served, prefix caching on)**

Run provenance: commit `dac39d53d3e7`, rule directive hash `cdcb43b542297e70`,
spec hashes — prompts `1842b6d031e65362`, type rules `434c9b068a083738`,
criteria `0e172a4c4d77fc09`, judge `6fd7662046af1ae1`.
Evidence arm **W0** (prefix-conditional, no lookahead), locked by the Step-3
rule after Stage-0. Anomaly policy `flag`; 5 anomalous files carried and
dual-reported. Normalization: per-type leave-one-file-out CV, fit separately
per subset and per GT setting.

---

## 1. Scoring pass

| subset | assessments | wall-clock | truncated assessments | truncated trajectories | max prefix |
|---|---|---|---|---|---|
| AG | 1,099 | 203 s | 0.0% | 0.0% | 9,931 tok |
| HC | 2,993 | 824 s | 13.8% | 24.1% | 26,950 tok |

8,184 assessments total across both GT settings. Readout fallback to `echo`
scoring: **0 occurrences** — `True`/`False` were in the top-20 at every
assessment.

## 2. E0 — score field (GT off)

| cell | n | mean | sd | p05 | median | p95 | saturated | distinct | AUROC vs derived labels |
|---|---|---|---|---|---|---|---|---|---|
| alg/plan | 13 | 0.718 | 0.265 | 0.228 | 0.867 | 0.964 | 0.0% | 13 | 0.556 |
| alg/delegate | 13 | 0.590 | 0.242 | 0.147 | 0.622 | 0.880 | 0.0% | 11 | 0.667 |
| alg/execute | 818 | 0.483 | 0.299 | 0.071 | 0.438 | 0.965 | 0.0% | 129 | 0.562 |
| alg/final | 253 | 0.436 | 0.253 | 0.090 | 0.407 | 0.882 | 0.0% | 73 | 0.259 |
| alg/unknown | 2 | 0.132 | 0.042 | — | 0.132 | — | 0.0% | 2 | — |
| hc/plan | 1,559 | 0.674 | 0.230 | 0.245 | 0.731 | 0.983 | 0.0% | 131 | 0.714 |
| hc/delegate | 689 | 0.736 | 0.248 | 0.262 | 0.827 | 0.991 | 0.0% | 123 | 0.523 |
| hc/execute | 689 | 0.545 | 0.309 | 0.073 | 0.562 | 0.979 | 0.0% | 125 | 0.572 |
| hc/final | 56 | 0.460 | 0.240 | 0.113 | 0.469 | 0.862 | 0.0% | 29 | 0.556 |

**Degenerate cells: none.** No cell is constant, saturated, or near-binary.

Level differs by type: `hc/delegate` mean 0.736 and `hc/plan` 0.674 against
`hc/execute` 0.545 — a 0.19 gap between coordination and execution types within
the same subset. `alg/final` AUROC 0.259 is the only cell below 0.5.

## 3. E0 — cross-fold threshold stability

| GT | subset | folds | global threshold mean | sd | CV | range |
|---|---|---|---|---|---|---|
| off | AG | 126 | +1.759 | 0.015 | **0.008** | [+1.704, +1.804] |
| off | HC | 58 | −0.848 | 0.436 | **0.514** | [−1.603, −0.271] |
| on | AG | 126 | +0.987 | 1.077 | **0.790** | [−0.601, +1.760] |
| on | HC | 58 | −1.191 | 0.166 | 0.140 | [−1.738, −1.068] |

Per-type threshold CV (GT off): HC delegate 0.320, execute 0.067, plan 0.045;
AG execute 0.028. Normalization statistics themselves are stable everywhere —
mean CV ≤ 0.006, sd CV ≤ 0.008 in every cell.

Worst cell: `hc/global` 0.514 (GT off), `alg/global` 0.790 (GT on).

## 4. E1 — primary rule fallback

`changepoint_single` falls back to argmin on boundary splits, contrast below the
registered 1.0 (z units), or trajectories shorter than 2×`min_seg`.

| subset | GT off | GT on | reasons (GT off) |
|---|---|---|---|
| AG (n=126) | **81.8%** | 77.0% | boundary 73, low_contrast 30, regime found 23 |
| HC (n=58) | **5.2%** | 5.2% | boundary 3, regime found 55 |

Fallback against trajectory length, pooled across subsets (GT off):

| steps | n | fallback |
|---|---|---|
| < 10 | 61 | 90.2% |
| 10–20 | 78 | 65.4% |
| 20–50 | 22 | 0.0% |
| 50+ | 23 | 0.0% |

AG trajectories run 7–10 steps; HC has a median of 33. With `min_seg = 2`, a
9-step trajectory admits splits at k ∈ [2,7], of which 2 are boundary positions.

## 5. E1 — attribution accuracy, exact scorer, all files

Bootstrap CIs over files, 2,000 resamples. **Bold** = primary rule.

### GT off

| rule | AG agent | AG step | HC agent | HC step |
|---|---|---|---|---|
| **changepoint_single** | **0.278** [0.198, 0.357] | **0.183** [0.119, 0.246] | **0.448** [0.328, 0.569] | **0.086** [0.017, 0.155] |
| first_crossing | 0.476 [0.389, 0.563] | 0.143 [0.079, 0.198] | 0.483 [0.345, 0.621] | 0.121 [0.034, 0.224] |
| argmin | 0.302 [0.222, 0.381] | 0.167 [0.103, 0.230] | 0.431 [0.310, 0.569] | 0.155 [0.069, 0.259] |
| changepoint (unnormalised gap) | — | — | 0.379 [0.259, 0.517] | 0.069 [0.017, 0.138] |
| agent_first | 0.278 [0.206, 0.357] | 0.183 [0.119, 0.246] | 0.362 [0.241, 0.483] | 0.103 [0.034, 0.190] |
| relative_crossing@1.5 | 0.302 [0.222, 0.381] | 0.159 [0.095, 0.222] | 0.466 [0.345, 0.603] | 0.069 [0.017, 0.138] |
| relative_crossing@2.0 | 0.302 [0.222, 0.381] | 0.167 [0.103, 0.230] | 0.431 [0.310, 0.569] | 0.155 [0.069, 0.259] |
| relative_crossing@2.5 | 0.302 [0.222, 0.381] | 0.167 [0.103, 0.230] | 0.431 [0.310, 0.569] | 0.155 [0.069, 0.259] |

### GT on

| rule | AG agent | AG step | HC agent | HC step |
|---|---|---|---|---|
| **changepoint_single** | **0.325** [0.246, 0.405] | **0.183** [0.119, 0.254] | **0.517** [0.397, 0.638] | **0.103** [0.034, 0.190] |
| first_crossing | 0.405 [0.317, 0.492] | 0.278 [0.206, 0.349] | 0.500 [0.362, 0.621] | 0.138 [0.052, 0.241] |
| argmin | 0.357 [0.278, 0.444] | 0.198 [0.127, 0.270] | 0.431 [0.310, 0.569] | 0.103 [0.034, 0.190] |
| changepoint (unnormalised gap) | 0.230 [0.159, 0.302] | 0.111 [0.056, 0.167] | 0.414 [0.276, 0.534] | 0.086 [0.017, 0.155] |
| agent_first | 0.302 [0.230, 0.389] | 0.238 [0.167, 0.317] | 0.397 [0.259, 0.534] | 0.155 [0.069, 0.259] |
| relative_crossing@1.5 | 0.365 [0.286, 0.452] | 0.206 [0.135, 0.278] | 0.431 [0.310, 0.569] | 0.069 [0.017, 0.138] |
| relative_crossing@2.0 | 0.357 [0.278, 0.444] | 0.198 [0.127, 0.270] | 0.431 [0.310, 0.569] | 0.121 [0.052, 0.207] |
| relative_crossing@2.5 | 0.357 [0.278, 0.444] | 0.198 [0.127, 0.270] | 0.431 [0.310, 0.569] | 0.121 [0.052, 0.207] |

`first_crossing` exceeds `changepoint_single` on agent accuracy in 3 of 4 cells;
the intervals are disjoint only for AG GT off (0.476 [0.389, 0.563] vs
0.278 [0.198, 0.357]). The `relative_crossing` k sweep moves ≤ 0.035 across
k ∈ {1.5, 2, 2.5} in every cell.

Published reference: 53.5% agent / 14.2% step.

## 6. E1 — orchestrator vs worker fault (primary rule, exact scorer)

| GT | subset | fault | n | agent accuracy | step accuracy |
|---|---|---|---|---|---|
| off | HC | orchestrator | 18 | 0.667 [0.444, 0.889] | 0.111 [0.000, 0.278] |
| off | HC | worker | 40 | 0.350 [0.200, 0.500] | 0.075 [0.000, 0.175] |
| on | HC | orchestrator | 18 | 0.667 [0.444, 0.889] | 0.111 [0.000, 0.278] |
| on | HC | worker | 40 | 0.450 [0.300, 0.600] | 0.100 [0.025, 0.200] |
| off | AG | worker | 126 | 0.278 [0.198, 0.357] | 0.183 [0.119, 0.254] |
| on | AG | worker | 126 | 0.325 [0.246, 0.405] | 0.183 [0.119, 0.254] |

AG contains no orchestrator-fault files under the collapse rule. In the Stage-0
smoke (n=2 orchestrator files) the normalized rank of the annotated step was
0.565 for orchestrator faults against 0.323 for worker faults; at n=18 the
ordering is reversed.

## 7. E1 — dual reporting across slices (primary rule)

| GT | subset | all | excl_flagged | excl_anomalous | excl_all_excluded |
|---|---|---|---|---|---|
| off | AG agent | 0.278 (126) | 0.276 (123) | 0.282 (124) | 0.281 (121) |
| off | HC agent | 0.448 (58) | 0.473 (55) | 0.455 (55) | 0.481 (52) |
| on | AG agent | 0.325 (126) | 0.325 (123) | 0.323 (124) | 0.322 (121) |
| on | HC agent | 0.517 (58) | 0.527 (55) | 0.509 (55) | 0.519 (52) |

Largest movement from any exclusion: 0.033 (HC agent, GT off, excl_all_excluded).

**Exact and substring scorers agree on every cell in this table.** The primary
rule emits an integer step index and an agent name copied from the trajectory,
so the substring artifact (gold contained in prediction) never fires. The
comparability row separates from exact match only for free-text predictions,
i.e. the baseline arms.

Anomalous files carried: AG `9318445f…`, `99da66d8…`; HC `56137764…`,
`72c06643…`, `a0c07678…`.

## 8. E1 — normalized position of the attributed step (primary rule)

| GT | subset | gold median | predicted median | mean delta | predicted after gold |
|---|---|---|---|---|---|
| off | AG | 0.333 | 0.333 | −0.004 | 38.1% |
| off | HC | 0.341 | 0.250 | −0.104 | 34.5% |
| on | AG | 0.333 | 0.333 | −0.039 | 34.9% |
| on | HC | 0.341 | 0.231 | −0.109 | 29.3% |

Gold medians reproduce the 0.29–0.33 early skew reported for both subsets.
Predictions sit at or earlier than gold in every cell.

## 9. Not run

- **gpt-4o baseline arm.** The supplied key authenticates but the account
  returns `insufficient_quota`. `gpt-4o` is present in their
  `KNOWN_GPT_MODELS`; the wrapper, the join through `question_ID`, and the
  snapshot receipt are wired and untested against live traffic.
- **Capability-control baseline arm** (their three methods under our judge).
- E2, E3 (as a standalone run), E4, E5, E6, E7, E9. E8 remains gated.

## 10. Run notes

The first attempt at the full pass was lost 40 minutes in: another user's
Llama-3.3-70B (TP=2) took host memory to 1 GB free of 251 GB and the kernel
killed the vLLM engine core. The run recorded a 500 followed by connection
refused, leaving HC at 846 of 2,993 rows. Retry-with-backoff and resume were
added, the server was restarted at `--gpu-memory-utilization 0.90
--max-num-seqs 2`, and the completed AG rows were reused. The numbers above are
from the completed pass.
