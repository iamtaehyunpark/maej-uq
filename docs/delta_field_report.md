# Lookahead-shift (δ) as an attribution field — prior experiment

**Question.** C5 and C6 lose to C3 on level, which is why they were dropped. But
level and *shift* are different quantities. A step that reads as sound in
isolation and collapses once what actually followed it is appended has been
falsified by its own consequence; a step that reads the same either way has not.
So: does the shift localize the fault, even though the level does not?

**Field.**

```
δ_C5[t] = p_C5[t] − p_C3[t]        δ_C6[t] = p_C6[t] − p_C3[t]
```

Polarity needs no adjustment: falsified ⇒ the score falls ⇒ δ negative, and the
rule set is already low-is-suspicious, so `changepoint_single` / `argmin` /
`first_crossing` / `relative_crossing` apply unchanged.

**Condition mapping** (`src/masattr/judge/score.py:242`):

| condition | this harness | window |
|---|---|---|
| C3 (base) | `lookahead=none` (W0) | prefix `0..t`, nothing appended |
| C5 | `lookahead=resp` | + the contiguous run of *other*-agent steps after `t`, cap 2 |
| C6 | `lookahead=own` | + that run **plus** the acting agent's next appearance |

C6 is a **superset** of C5 in this implementation, not a disjoint arm. `δ_C6 − δ_C5`
therefore does **not** isolate the agent's own next-step reasoning; it is the
marginal effect of adding own-next on top of the response window. Run at your
direction; noted so no reader draws the disjoint reading.

**Scope.** GT-off only, both subsets, full corpus (4,092 steps per arm). Judge,
spec hashes, and anomaly policy identical to the B0 reference row, so δ is
differenced against exactly the W0 rows that produced B0.

---

## 1. δ as an attribution field (rules identical to E1)

Exact scorer, slice `all`, file-level bootstrap CIs. B0 reference row repeated
for comparison.

| field | subset | rule | agent | step | fallback |
|---|---|---|---|---|---|
| **B0 (C3 level)** | alg | changepoint_single | 0.333 [0.246,0.413] | 0.190 [0.127,0.254] | 74.6% |
| δ_C5 | alg | changepoint_single | 0.381 [0.294,0.468] | 0.159 [0.095,0.230] | 64.3% |
| δ_C5 | alg | first_crossing | 0.516 [0.429,0.603] | 0.198 [0.127,0.270] | 64.3% |
| δ_C5 | alg | argmin | 0.365 [0.278,0.444] | 0.135 [0.079,0.198] | 64.3% |
| δ_C5 | alg | relative_crossing@2.0 | 0.365 [0.278,0.444] | 0.135 [0.079,0.198] | 64.3% |
| δ_C6 | alg | changepoint_single | 0.413 [0.325,0.492] | 0.151 [0.087,0.214] | 71.4% |
| δ_C6 | alg | first_crossing | 0.468 [0.381,0.556] | 0.151 [0.087,0.214] | 71.4% |
| δ_C6 | alg | argmin | 0.421 [0.333,0.508] | 0.167 [0.103,0.238] | 71.4% |
| δ_C6 | alg | relative_crossing@2.0 | 0.421 [0.333,0.508] | 0.167 [0.103,0.238] | 71.4% |
| **B0 (C3 level)** | hc | changepoint_single | 0.483 [0.345,0.621] | 0.103 [0.034,0.190] | 5.2% |
| δ_C5 | hc | changepoint_single | 0.414 [0.293,0.534] | 0.069 [0.017,0.138] | 32.8% |
| δ_C5 | hc | first_crossing | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 32.8% |
| δ_C5 | hc | argmin | 0.362 [0.241,0.483] | 0.052 [0.000,0.121] | 32.8% |
| δ_C5 | hc | relative_crossing@2.0 | 0.379 [0.259,0.500] | 0.052 [0.000,0.121] | 32.8% |
| δ_C6 | hc | changepoint_single | 0.397 [0.276,0.517] | 0.069 [0.017,0.138] | 20.7% |
| δ_C6 | hc | first_crossing | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 20.7% |
| δ_C6 | hc | argmin | 0.328 [0.207,0.448] | 0.017 [0.000,0.052] | 20.7% |
| δ_C6 | hc | relative_crossing@2.0 | 0.328 [0.207,0.448] | 0.017 [0.000,0.052] | 20.7% |

Every δ row sits inside the band every other field occupies. On the step column
— the one attribution actually turns on — δ is **at or below** the C3 level
field it was built from: 0.135–0.198 vs B0's 0.190 on `alg`, 0.017–0.069 vs
B0's 0.103 on `hc`. No δ row's CI excludes its B0 counterpart in either
direction.

The one place δ moves something is the fallback rate on `alg`: 64.3% (δ_C5)
against B0's 74.6%, i.e. the shift field has a two-regime structure the level
field lacks slightly more often. It does not convert into accuracy. On `hc` the
fallback rate moves the wrong way, 5.2% → 32.8%.

## 2. Step level — can the shift rank the gold step inside its own trajectory?

AUROC of −δ against *is this the gold step*, computed within each file (so
trajectory length and per-file offset cancel) and pooled.

| field | n files | within-file AUROC | pooled AUROC | mean δ at gold | mean δ elsewhere |
|---|---|---|---|---|---|
| δ_C5 | alg | 126 | 0.512 [0.455,0.566] | 0.511 | −0.1008 | −0.0922 |
| δ_C6 | alg | 126 | 0.484 [0.424,0.544] | 0.489 | −0.1139 | −0.1109 |
| δ_C5 | hc | 55 | 0.444 [0.362,0.530] | 0.417 | +0.0336 | −0.0238 |
| δ_C6 | hc | 55 | 0.444 [0.364,0.524] | 0.440 | +0.0356 | −0.0060 |

(`hc` n=55, not 58: the three released files whose `mistake_step` is out of
range have no gold step to rank against.)

All four CIs contain 0.5. The shift does not rank the faulty step above its
neighbours in either subset.

**The sign inverts on `hc`.** On `alg`, appending the lookahead lowers P(True)
almost uniformly — mean δ is negative at the gold step (−0.101) and negative
everywhere else (−0.092), a gap of 0.009. On `hc` the gold step's mean δ is
**positive** (+0.034) while the rest of the trajectory is negative (−0.024):
showing the judge what happened after the faulty step makes it *more* confident
that step was fine. That is the opposite of the mechanism the field was built
on, and it is the larger of the two effects.

## 3. File level — does the shift at the selected step predict a correct selection?

If δ can't rank steps, it might still say *when to trust* the pick. Here the
rule (`changepoint_single`) chooses a step, and δ at that step is tested against
whether the choice was right.

| field | n | correct | mean δ when correct | mean δ when wrong | AUROC |
|---|---|---|---|---|---|
| δ_C5 | alg | 126 | 20 | −0.3592 | −0.3276 | 0.558 |
| δ_C6 | alg | 126 | 19 | −0.4185 | −0.3652 | 0.592 |
| δ_C5 | hc | 58 | 4 | −0.0945 | −0.1985 | 0.264 |
| δ_C6 | hc | 58 | 4 | −0.1417 | −0.2110 | 0.417 |

Risk-coverage — keep only the fraction of files with the most negative δ and
measure accuracy on what's kept. Full coverage is the row's own accuracy.

| field | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| δ_C5 alg | 0.120 | 0.180 | 0.197 | 0.178 | 0.159 |
| δ_C6 alg | 0.200 | 0.180 | 0.171 | 0.168 | 0.151 |
| δ_C5 hc | 0.000 | 0.043 | 0.029 | 0.043 | 0.069 |
| δ_C6 hc | 0.000 | 0.087 | 0.057 | 0.043 | 0.069 |

On `alg` the direction is the hypothesized one — correct picks sit on larger
drops, AUROC 0.558 / 0.592 — but the curves are flat-to-noisy: δ_C6's best cell
is 0.200 at 20% coverage against 0.151 at full coverage, i.e. 5 correct out of
25 files against 19 out of 126. On `hc` both AUROCs are below 0.5 and the most
confident quintile contains **zero** correct attributions.

## 4. Field sanity (E0 on the δ fields)

Neither δ field is degenerate — no saturation, no near-binary collapse — so the
null results above are not an artifact of a broken field.

| field | degenerate | worst LOO threshold CV | per-cell AUROC range (sized cells) |
|---|---|---|---|
| δ_C5 | no | 0.292 @ hc/delegate | 0.401 – 0.778 |
| δ_C6 | no | 0.533 @ hc/delegate | 0.389 – 0.608 |

Per-cell means show where the shift actually lands:

| cell | δ_C5 mean | δ_C6 mean |
|---|---|---|
| alg/execute (n=818) | −0.1025 | −0.1167 |
| alg/final (n=253) | −0.0692 | −0.0986 |
| hc/plan (n=1559) | −0.0075 | +0.0189 |
| hc/delegate (n=689) | −0.0004 | +0.0116 |
| hc/execute (n=689) | −0.0783 | −0.0749 |
| hc/final (n=56) | +0.0005 | −0.0041 |

The lookahead's effect is concentrated on `alg/execute` (−0.10) and is
essentially nil on the two cells that dominate `hc` — `plan` (n=1,559, δ ≈
−0.008 / +0.019) and `delegate` (n=689, δ ≈ 0.000 / +0.012). On hand-crafted
trajectories, appending the next couple of steps to a plan or a delegation
barely moves the judge at all, which is why there is no shift signal to attribute
with.

## 5. Reading

The C5/C6 drop decision holds, and holds for a stronger reason than it was made
on. The arms were dropped because their *level* underperformed C3. This says the
*shift* carries no localization either: it can't rank the faulty step
(AUROC ≈ 0.5, all four CIs spanning chance), it doesn't improve any rule's step
accuracy over the C3 field it's differenced from, and as a confidence signal it
is weak on `alg` and inverted on `hc`.

The `hc` sign inversion is the substantive finding rather than the null. Two
things are true of it together: the gold step's mean δ is positive while the
trajectory's is negative, and the lookahead has almost no effect on the `plan`
and `delegate` cells that make up 75% of `hc` steps. A cap-2 window after a
plan or a delegation captures the assignee's acknowledgment, not the downstream
struggle that makes the step wrong — the same mechanism recorded for W+resp in
`score.py`'s Stage-0 note. Where the window does reach the consequence
(`alg/execute`), δ is a clean −0.10; where it doesn't, δ is ~0 and the residual
correlation runs backwards.

That points at window *length*, not at the shift idea, as what this experiment
actually tested. `deleg` (cap 5, extended to a delegation's resolution) is
implemented and unrun; it is the arm that would separate "the shift carries
nothing" from "a cap-2 window never sees the consequence on hand-crafted logs."

## 6. Run manifest

| item | value |
|---|---|
| judge | `Qwen/Qwen3.6-35B-A3B`, two single-GPU vLLM servers (A100 80GB), TP=1 |
| arms | `lookahead=resp`, `lookahead=own`, `readout=ptrue`, `policy=typed`, GT off |
| base | `runs/main/scores/*_ptrue_typed_nogt.jsonl` — the B0 reference rows |
| rows | 4,092 per arm (alg 1,099 / hc 2,993); parse_ok 1.0000 on all four files |
| wall clock | 19 min for both arms concurrently, after ~7 min of server startup |
| normalization | LOO folds refit on each δ field's own distribution, 184 folds |
| spec `prompts` | `e8bc3b7bb8f22151` — same scaffold as B0/B3 |
| anomaly policy | `flag` |
| CIs | file-level bootstrap, `n_boot=2000`, `seed=0` |

Artifacts: `runs/delta/{scores,fields,e0_*,e1_*,corr_*}` on the run host;
δ rows also appear as `D1` in `runs/base/MASTER.md`.

Reproduce:

```
bash run_delta.sh          # judge both lookahead arms
bash run_delta_eval.sh     # derive δ, E0 folds + sanity, E1 rules, correlations
python tools/master_table.py runs > runs/base/MASTER.md
```

**Not run:** GT-on (staged behind a positive GT-off result, which did not
arrive); `deleg` (cap 5) as a third δ arm; a disjoint C6 that excludes the
response window.
