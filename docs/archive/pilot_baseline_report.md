# Pilot baseline report

Deliverable for `docs/pilot_baseline_spec.md`. Every number below is read from
artifacts written by the runs themselves (`tools/master_table.py`,
`tools/reanalyses.py`); nothing here is recomputed by hand.

Corpus: Who&When, 184 files / 4,092 steps — Algorithm-Generated (`alg`) 126
files / 1,099 steps, Hand-Crafted (`hc`) 58 files / 2,993 steps. Judge
`Qwen/Qwen3.6-35B-A3B` served over vLLM. Bootstrap CIs are file-level,
`n_boot=2000`, `seed=0`. Unless stated otherwise: exact scorer, slice `all`.

---

## 1. Master table

Rows are B0–B4; `rule` applies only to rows that produce a per-step score field
and then need an attribution rule on top of it. `fallback` is the primary
rule's rate of falling back to argmin.

| row | GT | subset | rule | agent | step | fallback |
|---|---|---|---|---|---|---|
| B0 P(True)/W0 | off | alg | changepoint_single | 0.333 [0.246,0.413] | 0.190 [0.127,0.254] | 74.6% |
| B0 P(True)/W0 | off | alg | first_crossing | 0.500 [0.413,0.587] | 0.159 [0.095,0.222] | 74.6% |
| B0 P(True)/W0 | off | alg | argmin | 0.333 [0.254,0.413] | 0.175 [0.111,0.246] | 74.6% |
| B0 P(True)/W0 | off | alg | relative_crossing@2.0 | 0.333 [0.254,0.413] | 0.175 [0.111,0.246] | 74.6% |
| B0 P(True)/W0 | off | hc | changepoint_single | 0.483 [0.345,0.621] | 0.103 [0.034,0.190] | 5.2% |
| B0 P(True)/W0 | off | hc | first_crossing | 0.534 [0.397,0.655] | 0.086 [0.017,0.155] | 5.2% |
| B0 P(True)/W0 | off | hc | argmin | 0.397 [0.276,0.517] | 0.086 [0.017,0.172] | 5.2% |
| B0 P(True)/W0 | off | hc | relative_crossing@2.0 | 0.397 [0.276,0.517] | 0.086 [0.017,0.172] | 5.2% |
| B0 P(True)/W0 | on | alg | changepoint_single | 0.341 [0.262,0.429] | 0.206 [0.143,0.278] | 69.8% |
| B0 P(True)/W0 | on | alg | first_crossing | 0.421 [0.333,0.508] | 0.254 [0.183,0.333] | 69.8% |
| B0 P(True)/W0 | on | alg | argmin | 0.357 [0.270,0.444] | 0.214 [0.151,0.286] | 69.8% |
| B0 P(True)/W0 | on | alg | relative_crossing@2.0 | 0.357 [0.270,0.444] | 0.214 [0.151,0.286] | 69.8% |
| B0 P(True)/W0 | on | hc | changepoint_single | 0.517 [0.397,0.638] | 0.103 [0.034,0.190] | 6.9% |
| B0 P(True)/W0 | on | hc | first_crossing | 0.448 [0.328,0.586] | 0.052 [0.000,0.121] | 6.9% |
| B0 P(True)/W0 | on | hc | argmin | 0.431 [0.310,0.552] | 0.103 [0.034,0.190] | 6.9% |
| B0 P(True)/W0 | on | hc | relative_crossing@2.0 | 0.431 [0.310,0.552] | 0.103 [0.034,0.190] | 6.9% |
| B1 first_step | — | alg | *direct* | 0.492 [0.405,0.579] | 0.159 [0.095,0.222] | — |
| B1 last_step | — | alg | *direct* | 0.357 [0.270,0.444] | 0.008 [0.000,0.024] | — |
| B1 majority_agent | — | alg | *direct* | 0.365 [0.278,0.444] | 0.103 [0.056,0.167] | — |
| B1 prior_position | — | alg | *direct* | 0.317 [0.238,0.405] | 0.167 [0.103,0.238] | — |
| B1 uniform_random_step | — | alg | *direct* | 0.326 [0.294,0.358] | 0.117 [0.110,0.124] | — |
| B1 first_step | — | hc | *direct* | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | — |
| B1 last_step | — | hc | *direct* | 0.603 [0.466,0.724] | 0.103 [0.034,0.190] | — |
| B1 majority_agent | — | hc | *direct* | 0.310 [0.190,0.431] | 0.052 [0.000,0.121] | — |
| B1 prior_position | — | hc | *direct* | 0.379 [0.241,0.500] | 0.034 [0.000,0.086] | — |
| B1 uniform_random_step | — | hc | *direct* | 0.363 [0.293,0.434] | 0.038 [0.027,0.051] | — |
| B2 all_at_once | — | alg | *direct* | 0.524 [0.437,0.611] | 0.119 [0.063,0.183] | — |
| B2 step_by_step | — | alg | *direct* | 0.452 [0.365,0.540] | 0.294 [0.214,0.373] | — |
| B2 binary_search | — | alg | *direct* | 0.421 [0.333,0.508] | 0.167 [0.103,0.238] | — |
| B2 all_at_once | — | hc | *direct* | 0.207 [0.103,0.310] | 0.017 [0.000,0.052] | — |
| B2 step_by_step | — | hc | *direct* | 0.362 [0.241,0.483] | 0.086 [0.017,0.172] | — |
| B2 binary_search | — | hc | *direct* | 0.328 [0.224,0.448] | 0.138 [0.052,0.241] | — |
| B3 verbalized | off | alg | changepoint_single | 0.460 [0.373,0.556] | 0.183 [0.119,0.246] | 77.8% |
| B3 verbalized | off | alg | first_crossing | 0.524 [0.437,0.611] | 0.190 [0.127,0.262] | 77.8% |
| B3 verbalized | off | alg | argmin | 0.516 [0.429,0.603] | 0.175 [0.111,0.238] | 77.8% |
| B3 verbalized | off | alg | relative_crossing@2.0 | 0.516 [0.429,0.603] | 0.175 [0.111,0.238] | 77.8% |
| B3 verbalized | off | hc | changepoint_single | 0.448 [0.328,0.569] | 0.086 [0.017,0.155] | 31.0% |
| B3 verbalized | off | hc | first_crossing | 0.052 [0.000,0.121] | 0.000 [0.000,0.000] | 31.0% |
| B3 verbalized | off | hc | argmin | 0.328 [0.207,0.448] | 0.034 [0.000,0.086] | 31.0% |
| B3 verbalized | off | hc | relative_crossing@2.0 | 0.293 [0.190,0.414] | 0.034 [0.000,0.086] | 31.0% |
| B3 verbalized | on | alg | changepoint_single | 0.389 [0.302,0.476] | 0.159 [0.103,0.222] | 75.4% |
| B3 verbalized | on | alg | first_crossing | 0.476 [0.389,0.563] | 0.175 [0.111,0.238] | 75.4% |
| B3 verbalized | on | alg | argmin | 0.460 [0.373,0.548] | 0.151 [0.087,0.214] | 75.4% |
| B3 verbalized | on | alg | relative_crossing@2.0 | 0.460 [0.373,0.548] | 0.151 [0.087,0.214] | 75.4% |
| B3 verbalized | on | hc | changepoint_single | 0.466 [0.345,0.603] | 0.103 [0.034,0.190] | 25.9% |
| B3 verbalized | on | hc | first_crossing | 0.034 [0.000,0.086] | 0.000 [0.000,0.000] | 25.9% |
| B3 verbalized | on | hc | argmin | 0.345 [0.224,0.466] | 0.052 [0.000,0.121] | 25.9% |
| B3 verbalized | on | hc | relative_crossing@2.0 | 0.328 [0.207,0.448] | 0.052 [0.000,0.121] | 25.9% |
| B3 binary | off | alg | changepoint_single | 0.421 [0.333,0.508] | 0.206 [0.143,0.278] | 73.0% |
| B3 binary | off | alg | first_crossing | 0.460 [0.373,0.548] | 0.167 [0.103,0.230] | 73.0% |
| B3 binary | off | alg | argmin | 0.444 [0.357,0.532] | 0.151 [0.087,0.214] | 73.0% |
| B3 binary | off | alg | relative_crossing@2.0 | 0.444 [0.357,0.532] | 0.151 [0.087,0.214] | 73.0% |
| B3 binary | off | hc | changepoint_single | 0.397 [0.276,0.534] | 0.121 [0.034,0.207] | 8.6% |
| B3 binary | off | hc | first_crossing | 0.414 [0.276,0.534] | 0.052 [0.000,0.121] | 8.6% |
| B3 binary | off | hc | argmin | 0.397 [0.276,0.517] | 0.086 [0.017,0.172] | 8.6% |
| B3 binary | off | hc | relative_crossing@2.0 | 0.397 [0.276,0.517] | 0.086 [0.017,0.172] | 8.6% |
| B3 binary | on | alg | changepoint_single | 0.429 [0.341,0.516] | 0.222 [0.159,0.294] | 70.6% |
| B3 binary | on | alg | first_crossing | 0.492 [0.405,0.579] | 0.183 [0.119,0.254] | 70.6% |
| B3 binary | on | alg | argmin | 0.452 [0.365,0.540] | 0.159 [0.095,0.222] | 70.6% |
| B3 binary | on | alg | relative_crossing@2.0 | 0.452 [0.365,0.540] | 0.159 [0.095,0.222] | 70.6% |
| B3 binary | on | hc | changepoint_single | 0.379 [0.259,0.500] | 0.121 [0.034,0.207] | 10.3% |
| B3 binary | on | hc | first_crossing | 0.276 [0.155,0.397] | 0.052 [0.000,0.121] | 10.3% |
| B3 binary | on | hc | argmin | 0.362 [0.241,0.483] | 0.052 [0.000,0.121] | 10.3% |
| B3 binary | on | hc | relative_crossing@2.0 | 0.362 [0.241,0.483] | 0.052 [0.000,0.121] | 10.3% |
| B4 embed_divergence | off | alg | changepoint_single | 0.452 [0.365,0.540] | 0.135 [0.079,0.198] | 92.1% |
| B4 embed_divergence | off | alg | first_crossing | 0.492 [0.405,0.579] | 0.159 [0.095,0.222] | 92.1% |
| B4 embed_divergence | off | alg | argmin | 0.452 [0.365,0.540] | 0.151 [0.087,0.214] | 92.1% |
| B4 embed_divergence | off | alg | relative_crossing@2.0 | 0.452 [0.365,0.540] | 0.151 [0.087,0.214] | 92.1% |
| B4 embed_divergence | off | hc | changepoint_single | 0.259 [0.155,0.379] | 0.000 [0.000,0.000] | 67.2% |
| B4 embed_divergence | off | hc | first_crossing | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 67.2% |
| B4 embed_divergence | off | hc | argmin | 0.241 [0.138,0.362] | 0.000 [0.000,0.000] | 67.2% |
| B4 embed_divergence | off | hc | relative_crossing@2.0 | 0.207 [0.103,0.310] | 0.000 [0.000,0.000] | 67.2% |
| B4 nli_contradiction | off | alg | changepoint_single | 0.413 [0.325,0.500] | 0.119 [0.063,0.175] | 97.6% |
| B4 nli_contradiction | off | alg | first_crossing | 0.492 [0.405,0.579] | 0.159 [0.095,0.222] | 97.6% |
| B4 nli_contradiction | off | alg | argmin | 0.413 [0.325,0.500] | 0.119 [0.063,0.175] | 97.6% |
| B4 nli_contradiction | off | alg | relative_crossing@2.0 | 0.413 [0.325,0.500] | 0.119 [0.063,0.175] | 97.6% |
| B4 nli_contradiction | off | hc | changepoint_single | 0.414 [0.293,0.552] | 0.034 [0.000,0.086] | 65.5% |
| B4 nli_contradiction | off | hc | first_crossing | 0.328 [0.207,0.448] | 0.034 [0.000,0.086] | 65.5% |
| B4 nli_contradiction | off | hc | argmin | 0.414 [0.293,0.552] | 0.052 [0.000,0.121] | 65.5% |
| B4 nli_contradiction | off | hc | relative_crossing@2.0 | 0.483 [0.345,0.621] | 0.086 [0.017,0.172] | 65.5% |

### 1.1 What separates the rows

**Agent column.** Every row on `alg` lands in 0.317–0.524. The widest span
inside that band is B2 `all_at_once` (0.524) against B1 `prior_position`
(0.317); the reference row B0 (0.333–0.341) sits at the bottom of the band,
below the constant-output B1 predictors `first_step` (0.492) and
`majority_agent` (0.365). On `hc` the band is 0.000–0.603, with B1 `last_step`
(0.603) the highest row in the table and B2 `all_at_once` (0.207) among the
lowest. No B0/B3/B4 row's CI excludes the best B1 row on either subset.

**Step column.** The only row that exceeds 0.25 anywhere is B2 `step_by_step`
on `alg` (0.294 [0.214,0.373]). B0 reaches 0.190 (GT off) / 0.206 (GT on) on
`alg` and 0.103 on `hc`. B1 `uniform_random_step` is 0.117 on `alg` and 0.038
on `hc`; the four B1 heuristics span 0.008–0.167 (`alg`) and 0.000–0.103 (`hc`).

**Score field (B0 vs B3 vs B4).** Swapping the readout moves the agent column
by up to +0.13 on `alg` (B0 0.333 → B3 verbalized 0.460, GT off) while leaving
the step column inside 0.119–0.222 across all six fields and both GT settings.
The two B4 fields — which never see a judge prompt — produce agent accuracies
(0.413–0.452 on `alg`) in the same band as the judged fields.

**GT setting.** GT on vs off changes B0 by ≤0.034 on both columns and both
subsets. It is not the dominant axis in any row.

---

## 2. Re-analyses

### (i) Base-rate audit — `hc`, B1 predictors split by gold fault agent

| row | fault | n | agent | step |
|---|---|---|---|---|
| prior_position | orchestrator | 18 | 0.833 [0.667, 1.000] | 0.056 [0.000, 0.167] |
| prior_position | worker | 40 | 0.175 [0.075, 0.300] | 0.025 [0.000, 0.075] |
| majority_agent | orchestrator | 18 | 1.000 [1.000, 1.000] | 0.056 [0.000, 0.167] |
| majority_agent | worker | 40 | 0.000 [0.000, 0.000] | 0.050 [0.000, 0.125] |
| first_step | orchestrator | 18 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| first_step | worker | 40 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| last_step | orchestrator | 18 | 0.833 [0.611, 1.000] | 0.056 [0.000, 0.167] |
| last_step | worker | 40 | 0.500 [0.350, 0.650] | 0.125 [0.025, 0.225] |

| fault | n | mean share of steps owned by the gold agent |
|---|---|---|
| orchestrator | 18 | 0.761 |
| worker | 40 | 0.180 |

`majority_agent` is 1.000 on orchestrator-fault files and 0.000 on
worker-fault files. The step-ownership share is 0.761 vs 0.180 in the same
split: on `hc` the orchestrator owns most steps of most trajectories, so
selecting the most frequent agent reproduces the gold agent exactly when the
fault is the orchestrator's and never otherwise. Any row's `hc` agent accuracy
is therefore readable as a position on that ownership prior.

### (ii) `alg` / `final`-typed step score vs normalized gold position

Reference row (P(True), W0, GT off), raw scores before normalization.

| gold position | n | mean final-step score |
|---|---|---|
| [0.0, 0.2) | 99 | 0.272 |
| [0.2, 0.4) | 66 | 0.327 |
| [0.4, 0.6) | 45 | 0.253 |
| [0.6, 0.8) | 19 | 0.365 |
| [0.8, 1.0) | 24 | 0.251 |

Spread across bins is 0.114 (0.251 → 0.365) and non-monotone. The `final`-typed
score does not track where in the trajectory the gold mistake sits.

### (iii) Type composition, predicted vs gold decisive steps (primary rule)

| arm | delegate | execute | final | plan | unknown | anomalous |
|---|---|---|---|---|---|---|
| nogt, alg | 1 / 3 | 104 / 105 | 19 / 15 | 1 / 3 | 1 / 0 | — |
| nogt, hc | 1 / 6 | 24 / 31 | 1 / 6 | 29 / 12 | — | 3 |
| gt, alg | 1 / 3 | 102 / 105 | 23 / 15 | 0 / 3 | — | — |
| gt, hc | 1 / 6 | 25 / 31 | 1 / 6 | 28 / 12 | — | 3 |

(`pred / gold`; *anomalous* counts files whose released `mistake_step` is out
of range for the trajectory and therefore has no gold type.)

On `alg` the predicted composition tracks gold closely (`execute` 104 vs 105).
On `hc` it does not: the rule selects `plan` steps 28–29 times against a gold
count of 12, and selects `final` once against a gold count of 6.

### (iv) Tolerance curves, reference row (primary rule)

| GT | subset | exact | \|Δ\|≤1 | \|Δ\|≤2 |
|---|---|---|---|---|
| nogt | alg | 0.190 | 0.444 | 0.611 |
| nogt | hc | 0.103 | 0.224 | 0.241 |
| gt | alg | 0.206 | 0.468 | 0.651 |
| gt | hc | 0.103 | 0.172 | 0.207 |

`alg` gains +0.25 going exact → ≤1 and a further +0.17 going ≤1 → ≤2. `hc`
gains +0.07 and then +0.02–0.04. The near-miss structure differs by subset:
`alg` predictions cluster within two steps of gold; `hc` predictions do not.

---

## 3. Appendix A — parse failures

`parse_ok` per emitted score row. B0/B3 fields come from the judge; B4 fields
come from an embedding model and an NLI cross-encoder and have no parser, so
their failures are the step-0 neutral rows (no premise available), flagged
rather than parsed.

| field | subset | n rows | parse_ok |
|---|---|---|---|
| B0 ptrue, GT off | alg | 1,099 | 1.0000 |
| B0 ptrue, GT off | hc | 2,993 | 1.0000 |
| B0 ptrue, GT on | alg | 1,099 | 1.0000 |
| B0 ptrue, GT on | hc | 2,993 | 1.0000 |
| B3 verbalized, GT off | alg | 1,099 | 1.0000 |
| B3 verbalized, GT off | hc | 2,993 | 0.9993 |
| B3 verbalized, GT on | alg | 1,099 | 1.0000 |
| B3 verbalized, GT on | hc | 2,993 | 0.9980 |
| B3 binary, GT off | alg | 1,099 | 1.0000 |
| B3 binary, GT off | hc | 2,993 | 0.9940 |
| B3 binary, GT on | alg | 1,099 | 1.0000 |
| B3 binary, GT on | hc | 2,993 | 0.9930 |
| B4 embed_divergence | alg | 1,099 | 0.8854 |
| B4 embed_divergence | hc | 2,993 | 0.9806 |
| B4 nli_contradiction | alg | 1,099 | 0.8854 |
| B4 nli_contradiction | hc | 2,993 | 0.9806 |

Judged-field failure is 0.00–0.70%, entirely on `hc`. The B4 rates are the
step-0 fraction of each subset (1 − 126/1099 = 0.8854, 1 − 58/2993 = 0.9806) —
i.e. exactly one unscoreable row per file, by construction, not a failure mode.

B2 is a separate parser (their output format, not ours) and its failures are
scored as misses rather than dropped:

| method | subset | n gold | unparsed | rate |
|---|---|---|---|---|
| all_at_once | alg | 126 | 30 | 23.8% |
| step_by_step | alg | 126 | 1 | 0.8% |
| binary_search | alg | 126 | 1 | 0.8% |
| all_at_once | hc | 58 | 34 | 58.6% |
| step_by_step | hc | 58 | 0 | 0.0% |
| binary_search | hc | 58 | 14 | 24.1% |

The `all_at_once` rows are the ones with a large unparsed fraction; their
accuracies in §1 are computed over the full gold denominator, so an unparsed
file counts against them.

## 4. Appendix B — fallback rates

Rate at which `changepoint_single` declined its two-regime split and fell back
to argmin, with reasons, for the reference row:

| GT | subset | n | fallback | boundary | low_contrast | regime found |
|---|---|---|---|---|---|---|
| off | alg | 126 | 74.6% | 67 | 27 | 32 |
| off | hc | 58 | 5.2% | 3 | 0 | 55 |
| on | alg | 126 | 69.8% | 59 | 29 | 38 |
| on | hc | 58 | 6.9% | 4 | 0 | 54 |

Across all fields (§1 `fallback` column) the rate is 69.8–77.8% on `alg` for
judged fields and 92.1–97.6% for the B4 fields; on `hc` it is 5.2–10.3% for
B0/B3 binary, 25.9–31.0% for B3 verbalized, and 65.5–67.2% for B4. The
`alg`/`hc` asymmetry is dominated by the `boundary` reason, i.e. the best split
point falls at a trajectory edge — `alg` trajectories average 8.7 steps
(1,099/126) against 51.6 for `hc` (2,993/58).

## 5. Appendix C — E0 field sanity (sanity-only; gates nothing)

Per (subset, step-type) cell, AUROC against derived labels, saturated fraction,
and distinct-value count. The primary rule is fixed by
`specs/rule_directive.md`; E0's outcome does not select it.

| field | cells flagged degenerate | AUROC range (sized cells) | worst LOO threshold CV |
|---|---|---|---|
| B0 ptrue, GT off | 0 / 9 | 0.328 – 0.689 | 0.291 @ hc/delegate |
| B0 ptrue, GT on | 0 / 9 | 0.505 – 0.677 | 1.135 @ alg/global |
| B3 verbalized, GT off | 1 / 9 (`alg/unknown`, n=2) | 0.428 – 0.667 | 0.142 @ hc/delegate |
| B3 verbalized, GT on | 1 / 9 (`alg/unknown`, n=2) | 0.351 – 0.750 | 0.413 @ hc/global |
| B3 binary, GT off | 9 / 9 (all saturated) | 0.367 – 0.599 | 1.110 @ hc/delegate |
| B3 binary, GT on | 9 / 9 (all saturated) | 0.440 – 0.597 | 1.214 @ hc/execute |
| B4 embed_divergence | 0 / 9 | 0.008 – 0.563 | 0.860 @ hc/delegate |
| B4 nli_contradiction | 0 / 9 | 0.160 – 0.556 | 0.597 @ hc/execute |

Cells with n < 20 (`alg/delegate` n=13, `alg/plan` n=13, `alg/unknown` n=2) are
marked undersized and excluded from the AUROC ranges above. LOO normalization
used 184 folds (leave-one-file-out) for every field, fit per subset and per GT
setting.

Two specifics worth carrying forward: B3 `binary` is saturated in all nine
cells (2–3 distinct values per cell) yet still produces §1 numbers inside the
same band as the continuous fields; and `alg/final` under B4
`embed_divergence` has AUROC 0.008 — near-perfectly inverted against derived
labels — while its §1 row is 0.452 agent / 0.135 step.

## 6. Run manifest

| item | value |
|---|---|
| judge (B0, B3) | `Qwen/Qwen3.6-35B-A3B`, family `qwen`, served via vLLM OpenAI-compatible endpoint |
| B2 generator | same judge; their `inference.py` invoked as a subprocess with its OpenAI client redirected. API-returned snapshot confirmed `Qwen/Qwen3.6-35B-A3B` on all 6 runs |
| B4 embedder | `sentence-transformers/all-MiniLM-L6-v2` |
| B4 NLI | `cross-encoder/nli-deberta-v3-large` |
| spec `prompts` | `e8bc3b7bb8f22151` (all judged rows: B0, B3, B2) |
| spec `type_rules` | `434c9b068a083738` |
| spec `criteria` | `0e172a4c4d77fc09` |
| spec `judge` | `6fd7662046af1ae1` |
| spec `rule_directive` | `cdcb43b542297e70` |
| anomaly policy | `flag` — 5 released files kept, flagged, dual-reported (`alg`: 2, `hc`: 3) |
| CIs | file-level bootstrap, `n_boot=2000`, `seed=0` |
| platform | Linux 5.15.0, Python 3.11.7 (harness) / 3.12.13 (B4 arm) |

Run commits by row:

| rows | commit |
|---|---|
| B0 judge + E0 + E1 (`runs/main/*`) | `fd8f3e930e9985935d9681d38563c5b95baaceaf` |
| B3 judging (`runs/base/b3/judge_*`) | `1c03b7f92bb9779e4553bbba3f30342bfe4f1935` |
| B1, B2, B3 E0/E1, B4 E0/E1 | `52bffe8ffef438313a3970b1c311c28588437fcf` |
| B4 field extraction (`runs/base/b4`) | `6ae745293060c30c119d74ad96572c986d4ce1de` (dirty) |

The B4 field-extraction manifest carries `prompts=1842b6d031e65362`, the
pre-scaffold hash. That arm emits no prompts — it runs an embedder and an NLI
model over recorded text — so the hash records the tree state, not an input to
its numbers. Every arm that does send a prompt carries `e8bc3b7bb8f22151`.

Artifacts live under `runs/` on the run host:
`runs/main/{e0_gt,e0_nogt,e1_gt,e1_nogt,judge,scores}`,
`runs/base/{b1,b2,b3,b4}`, assembled tables at `runs/base/MASTER.md` and
`runs/base/REANALYSES.md`.

## 7. Out of scope

Not run, per spec and standing constraints:

- **gpt-4o published-regime arm of B2.** No quota; the capability-control arm
  (their prompts and logic, our judge) is what §1 reports. Their published
  gpt-4o numbers are therefore not reproduced here, and the B2 rows are not
  comparable to published Who&When figures.
- **Proxy-LM intrinsic field (E7).** Struck.
- **E8.** Gated.
- **Secondary judge (`meta-llama/Llama-3.1-70B-Instruct`) and the LLM type
  classifier (`google/gemma-3-27b-it`).** Declared and family-verified in
  `specs/judge.json`; no row in this report uses them.
