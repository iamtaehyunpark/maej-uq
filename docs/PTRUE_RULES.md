# Locating the faulty step from P(True): what works

Findings from a rule search over raw per-step P(True) scores (Qwen3.6-35B-A3B judge,
Who&When, 4,092 steps across 184 trajectories). Only results that changed a number
or ruled out a direction are recorded here; the many null variants are summarised
in one line at the end.

Corpora behave differently enough that they are always reported separately:
**algorithm-generated** (126 logs, mean 8.7 steps) and **hand-crafted** (58 logs,
mean 51.6 steps). Intervals are bootstrapped over trajectories, 2,000 resamples.

---

## 1. The rule that works

**Two-stage: `argmin` selects the agent, then first-crossing inside that agent's
own steps at that agent's median.**

| rule | alg step | alg agent | hc step | hc agent |
|---|---|---|---|---|
| random step | 0.120 | 0.333 | 0.037 | 0.492 |
| always step 1 (reads nothing) | 0.270 | — | 0.017 | — |
| first crossing, whole trajectory | 0.254 | 0.468 | 0.103 | 0.414 |
| argmin | 0.262 | 0.452 | 0.155 | 0.603 |
| **two-stage** | **0.302** [.222,.389] | 0.452 | **0.190** [.086,.293] | 0.603 |

**Where the gain comes from is not what it looks like.** Agent accuracy is
identical to `argmin` in every cell, because stage 1 *is* `argmin` — the rule
cannot improve agent attribution and does not try to. The entire step-level gain
comes from *restricting the search to one agent's trace*, which removes the other
agents' steps as distractors. Three independent analyses converged on this: no
transformation of the confidence values helps; the structure does.

That is the practical finding. Everything below either qualifies it or explains
why the obvious alternatives fail.

---

## 2. The method has an operating range, and you can detect it in advance

**Trajectory length gates whether the signal is usable at all.** On hand-crafted
logs, `argmin` accuracy declines monotonically across every length quartile:

| hand-crafted, by length | argmin step accuracy |
|---|---|
| 5–20 steps | **0.50** |
| 20–32 steps | 0.167 |
| 37–52 steps | 0.0 |
| 67–129 steps | 0.0 |

Point-biserial r = −0.41. Split at 30 steps: 0.300 versus 0.059 — a 5× gap.

*(Measured on a sealed-split exploration half, n=27 trajectories; the direction is
monotonic across every quartile but the exact cut is illustrative, not tuned. A
pre-specified confirmation on the held-out half is written and unrun.)*

This matters more than another point of accuracy: `n_steps` is known before any
judging happens, so the method can decline in advance. It also reframes the weak
headline number — 0.155 on hand-crafted is an average over a regime where the
signal works and one where it does not.

**Margin predicts whether the pick is right.** Ranking trajectories by the gap
between the lowest and second-lowest P(True), and keeping the most confident
quartile:

| | full set | top quartile by margin |
|---|---|---|
| hc step | 0.155 | **0.357** |
| hc agent | 0.603 | **0.714** |
| alg step | 0.262 | 0.344 |

Mean margin is higher when `argmin` is correct than when it is wrong (hc 0.160 vs
0.048). This is the one thing the continuous scale buys that a verdict cannot
express: not a better decision, but a calibrated sense of when to trust it.

Two independent routes to abstention — one free (length), one from the signal
(margin) — is the most deployable result in this document.

---

## 3. Confidence means different things for different roles

The optimal decision threshold, fit per agent role against the labels, varies far
more than expected:

| corpus | per-role optimal bar | global bar |
|---|---|---|
| algorithm-generated | **0.032 → 0.351** (11×) | 0.321 |
| hand-crafted | **0.016 → 0.679** (40×) | 0.407 |

`WebSurfer`'s bar (0.679) sits well *above* the global one — even moderately
confident WebSurfer steps are suspect. `Assistant`'s is 0.016. A single global
threshold is demonstrably the wrong instrument.

**But converting this into a rule does not clear the noise floor.** Two-stage with
a role-learned bar reaches **0.241** on hand-crafted against two-stage's 0.190 —
the best hand-crafted number produced — but the intervals overlap almost entirely
([.138,.362] vs [.086,.293]) at n=58. Directionally encouraging; not established.

Full likelihood-ratio scoring, using label-conditional densities per role,
*underperformed* simple thresholds — with 19–30 gold examples per role, a density
fit overfits where a single number survives. The ceiling on sophistication here is
label scarcity, not method.

**Do not normalise the role differences away.** Residualising P(True) against
role, type, position and length made selection worse (alg 0.202 vs 0.258 held
out). The role offsets are correlated with genuine difficulty — verification and
final steps really are riskier — so removing the "unit mismatch" removes signal
with it. Per-type LOO normalization cost accuracy the same way (0.246 vs 0.262).

---

## 4. Delegation steps are inverted

On hand-crafted logs, gold `delegate` steps have **higher** P(True) than non-gold
delegate steps — AUC 0.402, below chance, on n=689 steps. Every other role and
type runs the other way.

This is mechanistically coherent: a delegation that reads clean is exactly the
kind that misroutes work, and the fault only becomes visible later, in the
assignee's struggle. It also explains an old observation from this project that
delegation faults scored below chance.

Untested consequence: for delegate steps specifically, *high* confidence may be
the suspicious signal. Two cheap tests exist — invert the polarity for that one
type, or run the implemented-but-never-executed `deleg` evidence window, which
extends the lookahead to the delegation's resolution rather than the next two
steps.

---

## 5. What the comparison against LLM judges actually shows

The working hypothesis was that P(True) wins because a binary verdict is stuck at
a fixed ~0.5 cut while a continuous score can be cut optimally. **The data does not
support that mechanism.**

- **The judge is not thresholding P(True).** Reconstructing 779 real per-step
  verdicts from `step_by_step`'s forward walk, P(True) predicts the judge's own
  Yes/No at only **AUC 0.60–0.61**. The best-agreeing threshold collapses to ~0.02
  and barely beats "always No". The judge is using materially different
  information, not a mis-placed cut on the same signal.
- **An optimal cut loses to pure ranking** on the clean corpus. Leave-one-out
  cross-validated: hand-crafted optimal fixed cut 0.138 step / 0.535 agent, versus
  `argmin` 0.155 / 0.603. Per-trajectory percentile cuts are worse still (0.121).
- **On algorithm-generated, the judge's entire step deficit is refusals.**
  `step_by_step` returns "no error found" on **21 of 126** trajectories (16.7%).
  Backfill those with `argmin` and the judge's accuracy becomes 0.262 — exactly
  `argmin`'s. Conditional on committing, it scores 0.248 against `argmin`'s 0.262.

The honest reframing: the continuous readout buys **coverage** (it always ranks
something, where a verdict can decline) and **selective prediction** (§2). Not
optimal thresholding.

On hand-crafted, refusals are 1 trajectory, so there the judge's deficit
(0.103 vs 0.155) is genuine.

---

## 6. Validity constraints that bound every number above

**The algorithm-generated step column is largely a position lottery.** 42.9% of
its labels sit in the first two steps (34 at step 1, 20 at step 0), so a rule that
reads nothing and always answers "step 1" scores **0.270** — beating `argmin`'s
0.262. Any algorithm-generated result must be compared against that control or it
means nothing. Hand-crafted has 1.7% of labels in the first two steps and is the
clean test bed; conclusions here are weighted accordingly.

**The ceiling is low.** Even with oracle selection among the three
lowest-scoring steps, accuracy caps at ~55% (alg) and ~27–31% (hc).
Within-trajectory AUROC of P(True) against the label is 0.573 / 0.613. The
binding constraint is the signal, not the rule — which is why an exhaustive
search found so little.

**A measurement bug, now fixed.** Hand-crafted agent accuracy was reported as
0.672; it is **0.603**. Hand-crafted roles encode delegation targets
(`Orchestrator (-> WebSurfer)`), and a substring match credited a hit whenever the
Orchestrator delegated to the faulty agent — the speaker is the Orchestrator. Four
of 58 trajectories. Three independent reimplementations disagreed with our number
before the cause was isolated. The matcher now strips parentheticals; step
accuracy and all algorithm-generated figures are unaffected.

---

## 7. Directions closed

A sweep of 228 signal × policy combinations, eight hypotheses derived from the
human-metacognition literature, and a label-conditional threshold study all failed
to beat two-stage under holdout. The useful negatives:

- **Level beats change.** Raw P(True) ranks the gold step first more often than
  first difference, second difference, EWMA residual, or within-agent z. "Sharp
  drop" is not more informative than "just low" — which argues against a
  changepoint framing.
- **Showing the judge the reference answer does not localise.** It recalibrates
  globally: gold's rank shift between hidden and shown conditions has a standard
  deviation eight times its mean, and |p_true − p_true_gt| predicts the gold step
  at AUC 0.41–0.47, below chance. Measured twice by different methods.
- **Fusing the judge-free signals hurts.** Rank-fusing P(True) with embedding
  divergence and NLI contradiction underperforms P(True) alone (0.198–0.238 vs
  0.262).
- **`mass_on_answer` is unusable** as a second-order confidence signal — standard
  deviation ~0.001, no discriminating variance.
- **The faulty agent's *minimum* P(True) is not lower** than other agents' minima;
  it trends slightly higher. "Blame the agent owning the single worst step" is a
  natural heuristic and it does not work, though "lowest mean" has a small edge.

---

## Provenance

Rule accuracies in §1 are from `tools/ptrue_two_stage.py` against
`runs/clean/ptrue_{nogt,gt}.jsonl`. Cross-validated comparisons (§3, §5) and the
sealed-split exploration (§2) were run as separate analyses over the same data;
where a number is exploration-half only or has an unrun confirmation, it says so.
The scores themselves come from the corrected chat-template readout — probability
mass on {True, False} is 0.9992, so the judge is answering the question asked.
