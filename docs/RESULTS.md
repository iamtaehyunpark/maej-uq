# Results

> **Generated file** — rebuild with `tools/results_report.py` (command at the
> end). Do not hand-edit: the earlier per-experiment write-ups went stale when
> the prompt scaffold changed, and are kept only for history in `docs/archive/`.

Every number here is read back from the stored run outputs. The corpus is
Who&When: **184 logs, 4,092 steps** — 126 algorithm-generated logs (1,099
steps) and 58 hand-crafted ones (2,993 steps). Every judged row covers the
whole corpus; nothing here is a subsample.

## 1. What is being measured

Each failed multi-agent log has one recorded answer: which agent made the
decisive mistake, and at which step. A method must name both. The two are
scored separately because they behave very differently (see §4).

Everything is built the same way. A **score field** assigns every step a number
meaning roughly *how suspect is this step*. Those numbers are put on a common
scale using statistics fit without the log being scored, so no log is judged
against itself. Then a **rule** turns the sequence of numbers into a single
accusation.

The score fields compared here:

| score field | what it is |
|---|---|
| judge probability | the judge's probability that the step is correct, read from a single token |
| judge stated confidence | the judge writes a confidence number instead |
| judge yes/no verdict | the judge answers correct/incorrect |
| embedding divergence | how far the step drifts from what came before — no judge involved |
| contradiction detector | an off-the-shelf model asking whether the step contradicts earlier ones |
| shift when the response is added | how much the judge's probability *moves* once the reply to that step is appended |
| shift when response + next turn are added | the same, also appending that agent's own next turn |

Against them, two kinds of reference point: **simple guesses** that ignore
content entirely (always blame the first step, the last step, the busiest
agent, a random step), and the **published methods** from the Who&When paper
run through our judge.

The **answer** column says whether the judge was shown the reference answer
while scoring. Both settings are reported because the benchmark defines both.

## 2. Does the score field find the labelled step?

The direct test, with no rule and no threshold in the way: within each trajectory, rank every step by the score and ask whether the step Who&When labels as the decisive mistake ranks above the others. Reported as AUROC per trajectory, averaged over trajectories. **0.5 is chance.** Intervals are bootstrapped over files.

| score field | logs | files | AUROC | mean score at labelled step | elsewhere |
|---|---|---|---|---|---|
| P(True), answer hidden | algorithm-generated | 126 | **0.573 [0.515, 0.630]** | 0.295 | 0.332 |
| P(True), answer hidden | hand-crafted | 55 | **0.613 [0.527, 0.695]** | 0.339 | 0.429 |
| P(True), answer shown | algorithm-generated | 126 | **0.586 [0.529, 0.639]** | 0.261 | 0.296 |
| P(True), answer shown | hand-crafted | 55 | **0.622 [0.539, 0.702]** | 0.346 | 0.450 |
| verbalized confidence | algorithm-generated | 126 | 0.508 [0.468, 0.545] | 0.435 | 0.424 |
| verbalized confidence | hand-crafted | 55 | 0.483 [0.408, 0.556] | 0.672 | 0.648 |
| binary verdict | algorithm-generated | 126 | 0.512 [0.473, 0.546] | 0.254 | 0.279 |
| binary verdict | hand-crafted | 55 | 0.563 [0.496, 0.625] | 0.336 | 0.398 |
| embedding divergence | algorithm-generated | 126 | 0.384 [0.327, 0.442] | 0.790 | 0.738 |
| embedding divergence | hand-crafted | 55 | 0.536 [0.469, 0.603] | 0.775 | 0.784 |
| NLI contradiction | algorithm-generated | 126 | 0.425 [0.366, 0.485] | 0.912 | 0.911 |
| NLI contradiction | hand-crafted | 55 | **0.612 [0.524, 0.704]** | 0.655 | 0.876 |
| P(True) shift, +response | algorithm-generated | 126 | 0.512 [0.455, 0.566] | -0.101 | -0.092 |
| P(True) shift, +response | hand-crafted | 55 | 0.444 [0.362, 0.530] | 0.034 | -0.024 |
| P(True) shift, +response +next turn | algorithm-generated | 126 | 0.484 [0.424, 0.544] | -0.114 | -0.111 |
| P(True) shift, +response +next turn | hand-crafted | 55 | 0.444 [0.364, 0.524] | 0.036 | -0.006 |

**Bold** marks fields whose interval excludes chance.

P(True) is above chance on both corpora and in both answer settings, and its mean sits lower on the labelled step than elsewhere — the judge is measurably less confident about the step the benchmark blames. The effect is real and small: an AUROC near 0.58.

Everything downstream follows from that number. On a trajectory averaging under nine steps, an AUROC of 0.58 turns into roughly 19% exact-step accuracy once you force a single pick, which is what §3 reports. The accuracy tables are a lossy projection of this table, not an independent result.

### Broken out by step type (P(True), answer hidden)

Same question, pooled within each step type instead of within each trajectory, and scored against the labelled step versus the steps *before* it. Cells under 20 steps are too small to read.

| step type | steps | AUROC |
|---|---|---|
| alg/delegate | 13 | 0.583 *(too small)* |
| alg/execute | 818 | 0.565 |
| alg/final | 253 | 0.328 |
| alg/plan | 13 | 0.556 *(too small)* |
| hc/delegate | 689 | 0.474 |
| hc/execute | 689 | 0.608 |
| hc/final | 56 | 0.778 |
| hc/plan | 1,559 | 0.689 |
## 3. What that becomes after forcing one pick

How often each method names the faulty agent, and the faulty step, exactly. Confidence intervals are bootstrapped over files (2,000 resamples). *Rule gave up* is how often the registered rule found no usable split and fell back to simply picking the lowest-scoring step. Rows marked *none* make a prediction directly and never use a rule.

| score field | answer | logs | rule | names agent | names step | rule gave up |
|---|---|---|---|---|---|---|
| P(True) | hidden | algorithm-generated | two-regime split (registered) | 0.333 [0.246,0.413] | 0.190 [0.127,0.254] | 74.6% |
| P(True) | hidden | algorithm-generated | first step below threshold | 0.500 [0.413,0.587] | 0.159 [0.095,0.222] | 74.6% |
| P(True) | hidden | algorithm-generated | lowest-scoring step | 0.333 [0.254,0.413] | 0.175 [0.111,0.246] | 74.6% |
| P(True) | hidden | algorithm-generated | relative drop (2x) | 0.333 [0.254,0.413] | 0.175 [0.111,0.246] | 74.6% |
| P(True) | hidden | hand-crafted | two-regime split (registered) | 0.483 [0.345,0.621] | 0.103 [0.034,0.190] | 5.2% |
| P(True) | hidden | hand-crafted | first step below threshold | 0.534 [0.397,0.655] | 0.086 [0.017,0.155] | 5.2% |
| P(True) | hidden | hand-crafted | lowest-scoring step | 0.397 [0.276,0.517] | 0.086 [0.017,0.172] | 5.2% |
| P(True) | hidden | hand-crafted | relative drop (2x) | 0.397 [0.276,0.517] | 0.086 [0.017,0.172] | 5.2% |
| P(True) | shown | algorithm-generated | two-regime split (registered) | 0.341 [0.262,0.429] | 0.206 [0.143,0.278] | 69.8% |
| P(True) | shown | algorithm-generated | first step below threshold | 0.421 [0.333,0.508] | 0.254 [0.183,0.333] | 69.8% |
| P(True) | shown | algorithm-generated | lowest-scoring step | 0.357 [0.270,0.444] | 0.214 [0.151,0.286] | 69.8% |
| P(True) | shown | algorithm-generated | relative drop (2x) | 0.357 [0.270,0.444] | 0.214 [0.151,0.286] | 69.8% |
| P(True) | shown | hand-crafted | two-regime split (registered) | 0.517 [0.397,0.638] | 0.103 [0.034,0.190] | 6.9% |
| P(True) | shown | hand-crafted | first step below threshold | 0.448 [0.328,0.586] | 0.052 [0.000,0.121] | 6.9% |
| P(True) | shown | hand-crafted | lowest-scoring step | 0.431 [0.310,0.552] | 0.103 [0.034,0.190] | 6.9% |
| P(True) | shown | hand-crafted | relative drop (2x) | 0.431 [0.310,0.552] | 0.103 [0.034,0.190] | 6.9% |
| verbalized confidence | hidden | algorithm-generated | two-regime split (registered) | 0.460 [0.373,0.556] | 0.183 [0.119,0.246] | 77.8% |
| verbalized confidence | hidden | algorithm-generated | first step below threshold | 0.524 [0.437,0.611] | 0.190 [0.127,0.262] | 77.8% |
| verbalized confidence | hidden | algorithm-generated | lowest-scoring step | 0.516 [0.429,0.603] | 0.175 [0.111,0.238] | 77.8% |
| verbalized confidence | hidden | algorithm-generated | relative drop (2x) | 0.516 [0.429,0.603] | 0.175 [0.111,0.238] | 77.8% |
| verbalized confidence | hidden | hand-crafted | two-regime split (registered) | 0.448 [0.328,0.569] | 0.086 [0.017,0.155] | 31.0% |
| verbalized confidence | hidden | hand-crafted | first step below threshold | 0.052 [0.000,0.121] | 0.000 [0.000,0.000] | 31.0% |
| verbalized confidence | hidden | hand-crafted | lowest-scoring step | 0.328 [0.207,0.448] | 0.034 [0.000,0.086] | 31.0% |
| verbalized confidence | hidden | hand-crafted | relative drop (2x) | 0.293 [0.190,0.414] | 0.034 [0.000,0.086] | 31.0% |
| verbalized confidence | shown | algorithm-generated | two-regime split (registered) | 0.389 [0.302,0.476] | 0.159 [0.103,0.222] | 75.4% |
| verbalized confidence | shown | algorithm-generated | first step below threshold | 0.476 [0.389,0.563] | 0.175 [0.111,0.238] | 75.4% |
| verbalized confidence | shown | algorithm-generated | lowest-scoring step | 0.460 [0.373,0.548] | 0.151 [0.087,0.214] | 75.4% |
| verbalized confidence | shown | algorithm-generated | relative drop (2x) | 0.460 [0.373,0.548] | 0.151 [0.087,0.214] | 75.4% |
| verbalized confidence | shown | hand-crafted | two-regime split (registered) | 0.466 [0.345,0.603] | 0.103 [0.034,0.190] | 25.9% |
| verbalized confidence | shown | hand-crafted | first step below threshold | 0.034 [0.000,0.086] | 0.000 [0.000,0.000] | 25.9% |
| verbalized confidence | shown | hand-crafted | lowest-scoring step | 0.345 [0.224,0.466] | 0.052 [0.000,0.121] | 25.9% |
| verbalized confidence | shown | hand-crafted | relative drop (2x) | 0.328 [0.207,0.448] | 0.052 [0.000,0.121] | 25.9% |
| binary verdict | hidden | algorithm-generated | two-regime split (registered) | 0.421 [0.333,0.508] | 0.206 [0.143,0.278] | 73.0% |
| binary verdict | hidden | algorithm-generated | first step below threshold | 0.460 [0.373,0.548] | 0.167 [0.103,0.230] | 73.0% |
| binary verdict | hidden | algorithm-generated | lowest-scoring step | 0.444 [0.357,0.532] | 0.151 [0.087,0.214] | 73.0% |
| binary verdict | hidden | algorithm-generated | relative drop (2x) | 0.444 [0.357,0.532] | 0.151 [0.087,0.214] | 73.0% |
| binary verdict | hidden | hand-crafted | two-regime split (registered) | 0.397 [0.276,0.534] | 0.121 [0.034,0.207] | 8.6% |
| binary verdict | hidden | hand-crafted | first step below threshold | 0.414 [0.276,0.534] | 0.052 [0.000,0.121] | 8.6% |
| binary verdict | hidden | hand-crafted | lowest-scoring step | 0.397 [0.276,0.517] | 0.086 [0.017,0.172] | 8.6% |
| binary verdict | hidden | hand-crafted | relative drop (2x) | 0.397 [0.276,0.517] | 0.086 [0.017,0.172] | 8.6% |
| binary verdict | shown | algorithm-generated | two-regime split (registered) | 0.429 [0.341,0.516] | 0.222 [0.159,0.294] | 70.6% |
| binary verdict | shown | algorithm-generated | first step below threshold | 0.492 [0.405,0.579] | 0.183 [0.119,0.254] | 70.6% |
| binary verdict | shown | algorithm-generated | lowest-scoring step | 0.452 [0.365,0.540] | 0.159 [0.095,0.222] | 70.6% |
| binary verdict | shown | algorithm-generated | relative drop (2x) | 0.452 [0.365,0.540] | 0.159 [0.095,0.222] | 70.6% |
| binary verdict | shown | hand-crafted | two-regime split (registered) | 0.379 [0.259,0.500] | 0.121 [0.034,0.207] | 10.3% |
| binary verdict | shown | hand-crafted | first step below threshold | 0.276 [0.155,0.397] | 0.052 [0.000,0.121] | 10.3% |
| binary verdict | shown | hand-crafted | lowest-scoring step | 0.362 [0.241,0.483] | 0.052 [0.000,0.121] | 10.3% |
| binary verdict | shown | hand-crafted | relative drop (2x) | 0.362 [0.241,0.483] | 0.052 [0.000,0.121] | 10.3% |
| embedding divergence | hidden | algorithm-generated | two-regime split (registered) | 0.452 [0.365,0.540] | 0.135 [0.079,0.198] | 92.1% |
| embedding divergence | hidden | algorithm-generated | first step below threshold | 0.492 [0.405,0.579] | 0.159 [0.095,0.222] | 92.1% |
| embedding divergence | hidden | algorithm-generated | lowest-scoring step | 0.452 [0.365,0.540] | 0.151 [0.087,0.214] | 92.1% |
| embedding divergence | hidden | algorithm-generated | relative drop (2x) | 0.452 [0.365,0.540] | 0.151 [0.087,0.214] | 92.1% |
| embedding divergence | hidden | hand-crafted | two-regime split (registered) | 0.259 [0.155,0.379] | 0.000 [0.000,0.000] | 67.2% |
| embedding divergence | hidden | hand-crafted | first step below threshold | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 67.2% |
| embedding divergence | hidden | hand-crafted | lowest-scoring step | 0.241 [0.138,0.362] | 0.000 [0.000,0.000] | 67.2% |
| embedding divergence | hidden | hand-crafted | relative drop (2x) | 0.207 [0.103,0.310] | 0.000 [0.000,0.000] | 67.2% |
| NLI contradiction | hidden | algorithm-generated | two-regime split (registered) | 0.413 [0.325,0.500] | 0.119 [0.063,0.175] | 97.6% |
| NLI contradiction | hidden | algorithm-generated | first step below threshold | 0.492 [0.405,0.579] | 0.159 [0.095,0.222] | 97.6% |
| NLI contradiction | hidden | algorithm-generated | lowest-scoring step | 0.413 [0.325,0.500] | 0.119 [0.063,0.175] | 97.6% |
| NLI contradiction | hidden | algorithm-generated | relative drop (2x) | 0.413 [0.325,0.500] | 0.119 [0.063,0.175] | 97.6% |
| NLI contradiction | hidden | hand-crafted | two-regime split (registered) | 0.414 [0.293,0.552] | 0.034 [0.000,0.086] | 65.5% |
| NLI contradiction | hidden | hand-crafted | first step below threshold | 0.328 [0.207,0.448] | 0.034 [0.000,0.086] | 65.5% |
| NLI contradiction | hidden | hand-crafted | lowest-scoring step | 0.414 [0.293,0.552] | 0.052 [0.000,0.121] | 65.5% |
| NLI contradiction | hidden | hand-crafted | relative drop (2x) | 0.483 [0.345,0.621] | 0.086 [0.017,0.172] | 65.5% |
| P(True) shift, +response | hidden | algorithm-generated | two-regime split (registered) | 0.381 [0.294,0.468] | 0.159 [0.095,0.230] | 64.3% |
| P(True) shift, +response | hidden | algorithm-generated | first step below threshold | 0.516 [0.429,0.603] | 0.198 [0.127,0.270] | 64.3% |
| P(True) shift, +response | hidden | algorithm-generated | lowest-scoring step | 0.365 [0.278,0.444] | 0.135 [0.079,0.198] | 64.3% |
| P(True) shift, +response | hidden | algorithm-generated | relative drop (2x) | 0.365 [0.278,0.444] | 0.135 [0.079,0.198] | 64.3% |
| P(True) shift, +response | hidden | hand-crafted | two-regime split (registered) | 0.414 [0.293,0.534] | 0.069 [0.017,0.138] | 32.8% |
| P(True) shift, +response | hidden | hand-crafted | first step below threshold | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 32.8% |
| P(True) shift, +response | hidden | hand-crafted | lowest-scoring step | 0.362 [0.241,0.483] | 0.052 [0.000,0.121] | 32.8% |
| P(True) shift, +response | hidden | hand-crafted | relative drop (2x) | 0.379 [0.259,0.500] | 0.052 [0.000,0.121] | 32.8% |
| P(True) shift, +response +next turn | hidden | algorithm-generated | two-regime split (registered) | 0.413 [0.325,0.492] | 0.151 [0.087,0.214] | 71.4% |
| P(True) shift, +response +next turn | hidden | algorithm-generated | first step below threshold | 0.468 [0.381,0.556] | 0.151 [0.087,0.214] | 71.4% |
| P(True) shift, +response +next turn | hidden | algorithm-generated | lowest-scoring step | 0.421 [0.333,0.508] | 0.167 [0.103,0.238] | 71.4% |
| P(True) shift, +response +next turn | hidden | algorithm-generated | relative drop (2x) | 0.421 [0.333,0.508] | 0.167 [0.103,0.238] | 71.4% |
| P(True) shift, +response +next turn | hidden | hand-crafted | two-regime split (registered) | 0.397 [0.276,0.517] | 0.069 [0.017,0.138] | 20.7% |
| P(True) shift, +response +next turn | hidden | hand-crafted | first step below threshold | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 20.7% |
| P(True) shift, +response +next turn | hidden | hand-crafted | lowest-scoring step | 0.328 [0.207,0.448] | 0.017 [0.000,0.052] | 20.7% |
| P(True) shift, +response +next turn | hidden | hand-crafted | relative drop (2x) | 0.328 [0.207,0.448] | 0.017 [0.000,0.052] | 20.7% |
| simple guess: first step | — | algorithm-generated | *none* | 0.492 [0.405,0.579] | 0.159 [0.095,0.222] | — |
| simple guess: last step | — | algorithm-generated | *none* | 0.357 [0.270,0.444] | 0.008 [0.000,0.024] | — |
| simple guess: majority agent | — | algorithm-generated | *none* | 0.365 [0.278,0.444] | 0.103 [0.056,0.167] | — |
| simple guess: prior position | — | algorithm-generated | *none* | 0.317 [0.238,0.405] | 0.167 [0.103,0.238] | — |
| simple guess: uniform random step | — | algorithm-generated | *none* | 0.326 [0.294,0.358] | 0.117 [0.110,0.124] | — |
| simple guess: first step | — | hand-crafted | *none* | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | — |
| simple guess: last step | — | hand-crafted | *none* | 0.603 [0.466,0.724] | 0.103 [0.034,0.190] | — |
| simple guess: majority agent | — | hand-crafted | *none* | 0.310 [0.190,0.431] | 0.052 [0.000,0.121] | — |
| simple guess: prior position | — | hand-crafted | *none* | 0.379 [0.241,0.500] | 0.034 [0.000,0.086] | — |
| simple guess: uniform random step | — | hand-crafted | *none* | 0.363 [0.293,0.434] | 0.038 [0.027,0.051] | — |
| published method: all at once | — | algorithm-generated | *none* | 0.524 [0.437,0.611] | 0.119 [0.063,0.183] | — |
| published method: step by step | — | algorithm-generated | *none* | 0.452 [0.365,0.540] | 0.294 [0.214,0.373] | — |
| published method: binary search | — | algorithm-generated | *none* | 0.421 [0.333,0.508] | 0.167 [0.103,0.238] | — |
| published method: all at once | — | hand-crafted | *none* | 0.207 [0.103,0.310] | 0.017 [0.000,0.052] | — |
| published method: step by step | — | hand-crafted | *none* | 0.362 [0.241,0.483] | 0.086 [0.017,0.172] | — |
| published method: binary search | — | hand-crafted | *none* | 0.328 [0.224,0.448] | 0.138 [0.052,0.241] | — |

### Every rule tried, on the judge-probability field

The main table shows four rules for every field. All eight are listed here for the reference field only. The registered rule was fixed in advance, before any of these numbers existed — that it is not the winner is a result, not a reason to swap it.

**Reference answer hidden**

| rule | agent, alg-generated | step, alg-generated | agent, hand-crafted | step, hand-crafted |
|---|---|---|---|---|
| two-regime split (registered) | 0.333 [0.246,0.413] | 0.190 [0.127,0.254] | 0.483 [0.345,0.621] | 0.103 [0.034,0.190] |
| first step below threshold | 0.500 [0.413,0.587] | 0.159 [0.095,0.222] | 0.534 [0.397,0.655] | 0.086 [0.017,0.155] |
| lowest-scoring step | 0.333 [0.254,0.413] | 0.175 [0.111,0.246] | 0.397 [0.276,0.517] | 0.086 [0.017,0.172] |
| two-regime split, unscaled | 0.238 [0.167,0.317] | 0.111 [0.056,0.167] | 0.362 [0.241,0.483] | 0.103 [0.034,0.190] |
| first step of the worst-scoring agent | 0.262 [0.190,0.341] | 0.183 [0.119,0.246] | 0.362 [0.241,0.483] | 0.086 [0.017,0.172] |
| relative drop (1.5x) | 0.333 [0.254,0.413] | 0.183 [0.119,0.254] | 0.397 [0.276,0.517] | 0.069 [0.017,0.138] |
| relative drop (2x) | 0.333 [0.254,0.413] | 0.175 [0.111,0.246] | 0.397 [0.276,0.517] | 0.086 [0.017,0.172] |
| relative drop (2.5x) | 0.333 [0.254,0.413] | 0.175 [0.111,0.246] | 0.397 [0.276,0.517] | 0.086 [0.017,0.172] |

**Reference answer shown**

| rule | agent, alg-generated | step, alg-generated | agent, hand-crafted | step, hand-crafted |
|---|---|---|---|---|
| two-regime split (registered) | 0.341 [0.262,0.429] | 0.206 [0.143,0.278] | 0.517 [0.397,0.638] | 0.103 [0.034,0.190] |
| first step below threshold | 0.421 [0.333,0.508] | 0.254 [0.183,0.333] | 0.448 [0.328,0.586] | 0.052 [0.000,0.121] |
| lowest-scoring step | 0.357 [0.270,0.444] | 0.214 [0.151,0.286] | 0.431 [0.310,0.552] | 0.103 [0.034,0.190] |
| two-regime split, unscaled | 0.238 [0.167,0.310] | 0.095 [0.048,0.151] | 0.414 [0.276,0.534] | 0.086 [0.017,0.172] |
| first step of the worst-scoring agent | 0.286 [0.214,0.373] | 0.183 [0.119,0.254] | 0.310 [0.190,0.431] | 0.086 [0.017,0.155] |
| relative drop (1.5x) | 0.357 [0.270,0.444] | 0.222 [0.151,0.302] | 0.431 [0.310,0.552] | 0.086 [0.017,0.172] |
| relative drop (2x) | 0.357 [0.270,0.444] | 0.214 [0.151,0.286] | 0.431 [0.310,0.552] | 0.103 [0.034,0.190] |
| relative drop (2.5x) | 0.357 [0.270,0.444] | 0.214 [0.151,0.286] | 0.431 [0.310,0.552] | 0.103 [0.034,0.190] |


## 4. Naming the right agent mostly measures position

The agent is whoever owns the step a method picks. So this score inherits any positional pattern in the corpus — and there is a big one, which the simple guesses collect for free.

How often a fixed position belongs to the faulty agent, against that agent's average share of all steps (the rate a randomly placed pick would get):

| logs | always step 0 | step 1 | step 2 | last step | faulty agent's share of steps |
|---|---|---|---|---|---|
| algorithm-generated | **0.492** | 0.429 | 0.230 | 0.357 | **0.332** |
| hand-crafted | **0.000** | 0.310 | 0.310 | 0.603 | **0.361** |

Where the registered rule actually picks, and how often it names the right agent at each depth (reference answer hidden):

| logs | picks step 0 | right agent at step 0 | at step 1 | steps 2–4 | step 5+ | overall |
|---|---|---|---|---|---|---|
| algorithm-generated | 17/126 (13.5%) | 0.471 | 0.409 | 0.314 | 0.250 | 0.333 |
| hand-crafted | 0/58 (0.0%) | — | — | 0.619 | 0.405 | 0.483 |

On algorithm-generated logs the faulty agent owns about half of all first steps but only a third of steps overall, so *always blame whoever spoke first* banks a positional bonus the score field never tries to earn. The registered rule picks step 0 in roughly one log in seven, and its agent score falls steadily the deeper it picks — ending up at the plain ownership share. On this column the rule performs at the level of picking a step at random. Naming the right **step** is where the score field's signal actually shows up.

## 5. Allowing more than one guess

Two different ways of being lenient. **Near-miss** accepts a pick that lands within one or two steps of the true one. **Three guesses** accepts the true step appearing anywhere among the three most suspicious steps. The second is ranked on the score field itself — the registered rule returns a single step and has no three-guess form.

### Near-miss

| answer | logs | exact | within 1 step | within 2 steps |
|---|---|---|---|---|
| hidden | algorithm-generated | 0.190 | 0.444 | 0.611 |
| hidden | hand-crafted | 0.103 | 0.224 | 0.241 |
| shown | algorithm-generated | 0.206 | 0.468 | 0.651 |
| shown | hand-crafted | 0.103 | 0.172 | 0.207 |

### Three guesses — is the true step among the 3 most suspicious?

*Random 3* is the matched control: three steps drawn at random from the same log. Short logs make that control strong, so the gain over it is the number that matters.

| score field | logs | files | top 1 | top 3 | random 3 | gain | top 5 |
|---|---|---|---|---|---|---|---|
| P(True), answer hidden | algorithm-generated | 126 | 0.175 | 0.444 [0.357,0.532] | 0.360 | +0.085 | 0.683 |
| P(True), answer hidden | hand-crafted | 55 | 0.091 | 0.218 [0.109,0.327] | 0.116 | +0.102 | 0.273 |
| P(True), answer shown | algorithm-generated | 126 | 0.214 | 0.492 [0.405,0.571] | 0.360 | +0.132 | 0.698 |
| P(True), answer shown | hand-crafted | 55 | 0.109 | 0.218 [0.109,0.327] | 0.116 | +0.102 | 0.291 |
| verbalized confidence | algorithm-generated | 126 | 0.175 | 0.437 [0.349,0.516] | 0.360 | +0.077 | 0.706 |
| verbalized confidence | hand-crafted | 55 | 0.036 | 0.073 [0.018,0.145] | 0.116 | -0.043 | 0.164 |
| binary verdict | algorithm-generated | 126 | 0.151 | 0.508 [0.421,0.595] | 0.360 | +0.148 | 0.746 |
| binary verdict | hand-crafted | 55 | 0.091 | 0.145 [0.055,0.236] | 0.116 | +0.030 | 0.182 |
| embedding divergence | algorithm-generated | 126 | 0.151 | 0.206 [0.135,0.278] | 0.360 | -0.154 | 0.405 |
| embedding divergence | hand-crafted | 55 | 0.000 | 0.036 [0.000,0.091] | 0.116 | -0.079 | 0.127 |
| NLI contradiction | algorithm-generated | 126 | 0.119 | 0.270 [0.198,0.349] | 0.360 | -0.090 | 0.468 |
| NLI contradiction | hand-crafted | 55 | 0.055 | 0.109 [0.036,0.200] | 0.116 | -0.007 | 0.255 |
| P(True) shift, +response | algorithm-generated | 126 | 0.135 | 0.365 [0.278,0.452] | 0.360 | +0.005 | 0.587 |
| P(True) shift, +response | hand-crafted | 55 | 0.055 | 0.091 [0.018,0.182] | 0.116 | -0.025 | 0.127 |
| P(True) shift, +response +next turn | algorithm-generated | 126 | 0.167 | 0.333 [0.254,0.421] | 0.360 | -0.027 | 0.563 |
| P(True) shift, +response +next turn | hand-crafted | 55 | 0.018 | 0.055 [0.000,0.127] | 0.116 | -0.061 | 0.200 |

### Three guesses — naming the agent

*Agents covered* is how many distinct agents the three picks span. Once that approaches the number of agents present, three guesses stops distinguishing anything.

| score field | logs | files | top 1 | top 3 | random 3 | gain | agents covered |
|---|---|---|---|---|---|---|---|
| P(True), answer hidden | algorithm-generated | 126 | 0.333 | 0.651 | 0.703 | -0.052 | 2.16 |
| P(True), answer hidden | hand-crafted | 55 | 0.400 | 0.582 | 0.636 | -0.054 | 1.33 |
| P(True), answer shown | algorithm-generated | 126 | 0.357 | 0.683 | 0.703 | -0.021 | 2.14 |
| P(True), answer shown | hand-crafted | 55 | 0.436 | 0.600 | 0.636 | -0.036 | 1.33 |
| verbalized confidence | algorithm-generated | 126 | 0.516 | 0.738 | 0.703 | +0.035 | 2.40 |
| verbalized confidence | hand-crafted | 55 | 0.345 | 0.418 | 0.636 | -0.218 | 1.22 |
| binary verdict | algorithm-generated | 126 | 0.444 | 0.762 | 0.703 | +0.059 | 2.43 |
| binary verdict | hand-crafted | 55 | 0.400 | 0.491 | 0.636 | -0.145 | 1.18 |
| embedding divergence | algorithm-generated | 126 | 0.452 | 0.643 | 0.703 | -0.060 | 2.29 |
| embedding divergence | hand-crafted | 55 | 0.255 | 0.364 | 0.636 | -0.272 | 1.71 |
| NLI contradiction | algorithm-generated | 126 | 0.413 | 0.690 | 0.703 | -0.013 | 2.29 |
| NLI contradiction | hand-crafted | 55 | 0.418 | 0.655 | 0.636 | +0.019 | 1.60 |
| P(True) shift, +response | algorithm-generated | 126 | 0.365 | 0.675 | 0.703 | -0.029 | 2.25 |
| P(True) shift, +response | hand-crafted | 55 | 0.382 | 0.527 | 0.636 | -0.109 | 1.51 |
| P(True) shift, +response +next turn | algorithm-generated | 126 | 0.421 | 0.683 | 0.703 | -0.021 | 2.23 |
| P(True) shift, +response +next turn | hand-crafted | 55 | 0.327 | 0.600 | 0.636 | -0.036 | 1.55 |

## 6. Why hand-crafted logs flatter the simple guesses

| simple guess | who was at fault | logs | names agent | names step |
|---|---|---|---|---|
| prior position | orchestrator | 18 | 0.833 [0.667, 1.000] | 0.056 [0.000, 0.167] |
| prior position | worker | 40 | 0.175 [0.075, 0.300] | 0.025 [0.000, 0.075] |
| majority agent | orchestrator | 18 | 1.000 [1.000, 1.000] | 0.056 [0.000, 0.167] |
| majority agent | worker | 40 | 0.000 [0.000, 0.000] | 0.050 [0.000, 0.125] |
| first step | orchestrator | 18 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| first step | worker | 40 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| last step | orchestrator | 18 | 0.833 [0.611, 1.000] | 0.056 [0.000, 0.167] |
| last step | worker | 40 | 0.500 [0.350, 0.650] | 0.125 [0.025, 0.225] |

| who was at fault | logs | that agent's mean share of steps |
|---|---|---|
| orchestrator | 18 | 0.761 |
| worker | 40 | 0.180 |

*Blame the busiest agent* is right every time the orchestrator was at fault and wrong every time a worker was, and the ownership shares split the same way. On hand-crafted logs the orchestrator owns most steps of most logs, so that guess reproduces the answer exactly when the orchestrator erred and never otherwise. It is a property of the corpus, not a skill.

## 7. Appendix — unreadable outputs and rule fallback

| score field | logs | rows | share readable |
|---|---|---|---|
| P(True), answer hidden | algorithm-generated | 1,099 | 1.0000 |
| P(True), answer hidden | hand-crafted | 2,993 | 1.0000 |
| P(True), answer shown | algorithm-generated | 1,099 | 1.0000 |
| P(True), answer shown | hand-crafted | 2,993 | 1.0000 |
| verbalized confidence | algorithm-generated | 1,099 | 1.0000 |
| verbalized confidence | hand-crafted | 2,993 | 0.9993 |
| binary verdict | algorithm-generated | 1,099 | 1.0000 |
| binary verdict | hand-crafted | 2,993 | 0.9940 |
| embedding divergence | algorithm-generated | 1,099 | 0.8854 |
| embedding divergence | hand-crafted | 2,993 | 0.9806 |
| NLI contradiction | algorithm-generated | 1,099 | 0.8854 |
| NLI contradiction | hand-crafted | 2,993 | 0.9806 |
| P(True) shift, +response | algorithm-generated | 1,099 | 1.0000 |
| P(True) shift, +response | hand-crafted | 2,993 | 1.0000 |
| P(True) shift, +response +next turn | algorithm-generated | 1,099 | 1.0000 |
| P(True) shift, +response +next turn | hand-crafted | 2,993 | 1.0000 |

> The two non-judge fields cannot score a log's first step — there is nothing before it to compare against — so their figure is exactly one unscoreable row per log, by construction rather than a failure.

The published methods use their own output format and parser. Output we could not parse counts as a miss against the full set of logs, not as a dropped log:

| published method | logs | logs total | unparseable | rate |
|---|---|---|---|---|
| all at once | algorithm-generated | 126 | 30 | 23.8% |
| step by step | algorithm-generated | 126 | 1 | 0.8% |
| binary search | algorithm-generated | 126 | 1 | 0.8% |
| all at once | hand-crafted | 58 | 34 | 58.6% |
| step by step | hand-crafted | 58 | 0 | 0.0% |
| binary search | hand-crafted | 58 | 14 | 24.1% |

How often the registered rule gave up, and why:

| answer | logs | files | gave up | reasons |
|---|---|---|---|---|
| hidden | algorithm-generated | 126 | 74.6% | boundary 67, low contrast 27, regime found 32 |
| hidden | hand-crafted | 58 | 5.2% | boundary 3, regime found 55 |
| shown | algorithm-generated | 126 | 69.8% | boundary 59, low contrast 29, regime found 38 |
| shown | hand-crafted | 58 | 6.9% | boundary 4, regime found 54 |

## 8. How these numbers were produced

| item | value |
|---|---|
| judge | `Qwen/Qwen3.6-35B-A3B`, served locally |
| embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| contradiction model | `cross-encoder/nli-deberta-v3-large` |
| prompt set hash | `e8bc3b7bb8f22151` |
| rule registration hash | `cdcb43b542297e70` |
| step-type rules hash | `434c9b068a083738` |
| confidence intervals | bootstrap over files, 2,000 resamples, seed 0 |
| irregular logs | the 5 released logs that break the per-step checks are kept, flagged, and reported both ways |

The published-method rows use the original authors' code and prompts with our judge substituted for theirs. That makes them a control for judge capability — **not** a reproduction of their published figures, which used a different model we had no quota for.

Rebuild every table here with:

```
python tools/results_report.py runs <data-root> > docs/RESULTS.md
```

## 9. What this adds up to

1. **External P(True) does find the labelled step, weakly.** Within-trajectory
   AUROC ≈ 0.57–0.62, interval above chance on both corpora and in both answer
   settings, with the judge measurably less confident on the labelled step than
   on the rest of the trajectory. This is the result; everything else is a
   consequence of it.
2. **A 0.58 AUROC does not survive being forced into one pick.** On logs
   averaging under nine steps it becomes roughly 19% exact-step accuracy. The
   gap between §2 and §3 is arithmetic, not a second finding, and no rule in §3
   recovers it.
3. **Naming the agent tests something else.** That column largely rewards
   picking an early or busy position, which the content-free guesses do by
   construction — see §4. Rankings taken from it are not attribution skill.
4. **One published method still beats the rule layer on the step column** —
   step-by-step, 0.294 against 0.190, same judge, same logs. That gap is about
   method and has no positional explanation.
5. **The registered rule rarely fires on short logs** — it falls back to the
   lowest-scoring step in about three quarters of algorithm-generated logs,
   because the best split keeps landing on a trajectory edge.
6. **The lookahead shift is a dead end.** It cannot rank the labelled step
   anywhere (every interval spans chance) and points the wrong way on
   hand-crafted logs.

## 9. Limits

- The published-method rows are a judge-capability control, not a reproduction;
  they are not comparable to the figures in that paper.
- The lookahead-shift rows were run only with the reference answer hidden. A
  wider window (following a delegation to its resolution) is implemented and
  unrun — that, not the shift idea itself, is the open question, since the
  narrow window rarely reaches the consequence on hand-crafted logs.
- Three hand-crafted logs record a mistake step outside the log's own length.
  They are counted and reported, never silently dropped, which is why some
  tables show 55 hand-crafted logs rather than 58.
- Rankings taken from the agent column should not be read as attribution
  skill. See §3.

