# External P(True) on Who&When — top-1 and top-2 accuracy

Can a judge model's confidence that a step is correct locate the step, and the
agent, that Who&When labels as the decisive mistake?

Every step of every failed trajectory is scored once by an external judge. The
judge sees the task and the transcript **up to and including** that step, never
what follows, and answers one question: *is this step correct and appropriate?*
P(True) is the probability it assigns to `True`. Steps are then ranked
least-confident-first, and we ask whether the labelled step — and the labelled
agent — appear in the top k.

Corpus: 126 algorithm-generated logs (1,099 steps) and 58 hand-crafted logs
(2,993 steps). Three hand-crafted logs carry a `mistake_step` outside their own
length and are excluded from scoring, leaving 55; they are counted, not hidden.
Intervals are bootstrapped over logs, 2,000 resamples.

---

## Top-1

The single least-confident step.

| method | corpus | names the step | names the agent |
|---|---|---|---|
| **P(True), answer hidden** | algorithm-generated | **0.262 [0.190, 0.341]** | **0.452 [0.365, 0.540]** |
| **P(True), answer shown** | algorithm-generated | **0.206 [0.135, 0.278]** | **0.421 [0.325, 0.508]** |
| baseline: first step | algorithm-generated | 0.159 [0.095, 0.222] | 0.500 [0.413, 0.587] |
| baseline: last step | algorithm-generated | 0.008 [0.000, 0.024] | 0.357 [0.278, 0.444] |
| baseline: busiest agent | algorithm-generated | 0.151 [0.095, 0.214] | 0.429 [0.341, 0.516] |
| baseline: random step | algorithm-generated | 0.120 [0.115, 0.125] | 0.333 [0.304, 0.363] |
| **P(True), answer hidden** | hand-crafted | **0.164 [0.073, 0.273]** | **0.691 [0.564, 0.818]** |
| **P(True), answer shown** | hand-crafted | **0.164 [0.073, 0.273]** | **0.618 [0.491, 0.745]** |
| baseline: first step | hand-crafted | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| baseline: last step | hand-crafted | 0.109 [0.036, 0.200] | 0.618 [0.491, 0.745] |
| baseline: busiest agent | hand-crafted | 0.018 [0.000, 0.055] | 0.327 [0.200, 0.436] |
| baseline: random step | hand-crafted | 0.039 [0.028, 0.050] | 0.503 [0.449, 0.555] |

## Top-2

The two least-confident steps.

| method | corpus | names the step | names the agent |
|---|---|---|---|
| **P(True), answer hidden** | algorithm-generated | **0.405 [0.325, 0.484]** | **0.675 [0.595, 0.754]** |
| **P(True), answer shown** | algorithm-generated | **0.365 [0.286, 0.452]** | **0.635 [0.548, 0.722]** |
| baseline: first two steps | algorithm-generated | 0.429 [0.341, 0.516] | 0.770 [0.690, 0.841] |
| baseline: last two steps | algorithm-generated | 0.143 [0.087, 0.206] | 0.516 [0.429, 0.603] |
| baseline: two busiest agents | algorithm-generated | 0.262 [0.190, 0.333] | 0.659 [0.571, 0.738] |
| baseline: two random steps | algorithm-generated | 0.240 [0.230, 0.250] | 0.553 [0.517, 0.588] |
| **P(True), answer hidden** | hand-crafted | **0.200 [0.091, 0.309]** | **0.709 [0.582, 0.818]** |
| **P(True), answer shown** | hand-crafted | **0.218 [0.109, 0.327]** | **0.745 [0.618, 0.855]** |
| baseline: first two steps | hand-crafted | 0.018 [0.000, 0.055] | 0.327 [0.200, 0.436] |
| baseline: last two steps | hand-crafted | 0.127 [0.055, 0.218] | 0.782 [0.673, 0.891] |
| baseline: two busiest agents | hand-crafted | 0.055 [0.000, 0.127] | 0.891 [0.800, 0.964] |
| baseline: two random steps | hand-crafted | 0.077 [0.056, 0.100] | 0.719 [0.659, 0.772] |

---

## Attribution rules on the same field

The two tables above use no rule: top-1 is simply the lowest-scoring step. The
harness also carries a set of registered attribution rules, which read the score
field after it has been put on a common scale by leave-one-file-out
normalization. Those are reported here on the corrected field, so the rule layer
and the readout are no longer confounded.

**Answer hidden**

| rule | step, alg-generated | agent, alg-generated | step, hand-crafted | agent, hand-crafted |
|---|---|---|---|---|
| two-regime split (registered) | 0.238 | 0.421 | 0.121 | 0.466 |
| **first step below threshold** | **0.325** | 0.460 | 0.103 | 0.293 |
| lowest-scoring step | 0.246 | 0.437 | 0.121 | 0.414 |
| two-regime split, unscaled | 0.111 | 0.286 | 0.086 | 0.328 |
| relative drop 1.5x | 0.262 | 0.437 | 0.103 | 0.414 |
| relative drop 2x | 0.246 | 0.437 | 0.121 | 0.397 |
| relative drop 2.5x | 0.246 | 0.437 | 0.121 | 0.414 |

**Answer shown**

| rule | step, alg-generated | agent, alg-generated | step, hand-crafted | agent, hand-crafted |
|---|---|---|---|---|
| two-regime split (registered) | 0.190 | 0.405 | 0.138 | 0.466 |
| **first step below threshold** | **0.325** | 0.460 | **0.224** | **0.569** |
| lowest-scoring step | 0.230 | 0.437 | 0.121 | 0.431 |
| two-regime split, unscaled | 0.127 | 0.278 | 0.086 | 0.276 |
| relative drop 1.5x | 0.238 | 0.429 | 0.086 | 0.397 |
| relative drop 2x | 0.230 | 0.437 | 0.103 | 0.431 |
| relative drop 2.5x | 0.230 | 0.437 | 0.121 | 0.431 |

`agent_first` — first step of the worst-scoring agent — is absent from these
tables. It is withdrawn from the reported rule set: it lost to another rule in
every cell but one (hand-crafted step accuracy, answer hidden: 0.172 against
0.121, roughly three logs on the noisiest subset), and the selector sweep below
shows why. It remains implemented and callable.

Rate at which the registered rule found no usable split and fell back to the
lowest-scoring step: 65.9% / 10.3% (answer hidden, alg / hand-crafted) and
54.8% / 8.6% (answer shown). Both are lower than on the old field, where the
same rule fell back on 74.6% of algorithm-generated logs.

**First-step-below-threshold is now the best rule**, at 0.325 step accuracy on
algorithm-generated logs in both settings, against 0.238 / 0.190 for the
registered primary and 0.246 for the lowest-scoring step. On the earlier,
broken field this rule scored 0.159 / 0.254 and the ordering between rules was
unstable. A cleaner score field is what made a threshold-crossing rule viable —
the threshold now sits in a distribution the judge actually produced.

The registered rule is not the winner. It was fixed in advance, before any of
these numbers existed, and swapping it now on the strength of the table would
be exactly the selection effect that registering it was meant to prevent.

**It also removes the backwards answer-shown effect — for that rule only.**
Under first-step-below-threshold, showing the reference answer ties on
algorithm-generated (0.325 both ways) and clearly helps hand-crafted (step
0.103 to 0.224, agent 0.293 to 0.569). The degradation noted above is specific
to lowest-scoring-step selection, not a property of the field.

**One number does not carry across.** The lowest-scoring step reads 0.246 here
against 0.262 in the top-1 table, on identical data. The difference is
normalization: the rule path z-scores per step type using leave-one-file-out
statistics before ranking, while the top-1 table ranks raw P(True). Per-type
normalization costs about two logs on this field. The two are not
interchangeable and should not be quoted as though they were.

---

## Choosing the agent directly

The rule table's `agent_first` selects the agent whose *best* step is still the
worst — a minimax criterion — then picks a step inside that agent. That is one
aggregation among many and had never been compared against the alternatives.
Each selector below scores every agent from its own steps' P(True), takes the
worst-scoring agent, and reports that agent's lowest-scoring step. Raw P(True),
no normalization.

**Algorithm-generated**

| agent selector | agent, hidden | step, hidden | agent, shown | step, shown |
|---|---|---|---|---|
| worst single step (min) | 0.452 [0.365, 0.540] | **0.262** [0.190, 0.341] | 0.421 [0.325, 0.508] | 0.206 [0.135, 0.278] |
| best step still worst (max) — *withdrawn* | 0.294 [0.222, 0.373] | 0.151 [0.095, 0.214] | 0.325 [0.246, 0.413] | 0.198 [0.135, 0.270] |
| mean | 0.405 [0.325, 0.492] | 0.222 [0.151, 0.294] | 0.381 [0.302, 0.468] | 0.214 [0.143, 0.286] |
| median | 0.429 [0.341, 0.516] | 0.238 [0.167, 0.317] | 0.389 [0.310, 0.476] | 0.222 [0.151, 0.302] |
| **mean of its 2 worst steps** | **0.484** [0.397, 0.571] | 0.254 [0.183, 0.333] | **0.452** [0.365, 0.540] | **0.246** [0.175, 0.325] |
| fraction of steps below 0.5 | 0.397 [0.317, 0.484] | 0.214 [0.143, 0.286] | 0.373 [0.286, 0.452] | 0.206 [0.143, 0.278] |
| count of steps below 0.5 | 0.452 [0.365, 0.540] | 0.183 [0.119, 0.254] | 0.437 [0.357, 0.524] | 0.183 [0.119, 0.254] |
| **total suspicion, sum(1−p)** | **0.484** [0.397, 0.571] | 0.198 [0.135, 0.270] | 0.437 [0.349, 0.524] | 0.175 [0.111, 0.246] |

**Hand-crafted**

| agent selector | agent, hidden | step, hidden | agent, shown | step, shown |
|---|---|---|---|---|
| **worst single step (min)** | **0.691** [0.564, 0.818] | **0.164** [0.073, 0.273] | 0.618 [0.491, 0.745] | 0.164 [0.073, 0.273] |
| best step still worst (max) — *withdrawn* | 0.455 [0.327, 0.582] | 0.127 [0.055, 0.218] | 0.473 [0.345, 0.600] | 0.164 [0.073, 0.255] |
| mean | 0.509 [0.382, 0.636] | 0.145 [0.055, 0.236] | 0.527 [0.400, 0.655] | 0.145 [0.055, 0.236] |
| median | 0.545 [0.418, 0.673] | 0.127 [0.055, 0.218] | 0.509 [0.382, 0.636] | 0.145 [0.055, 0.236] |
| mean of its 2 worst steps | 0.618 [0.491, 0.745] | 0.145 [0.055, 0.236] | **0.636** [0.509, 0.764] | 0.164 [0.073, 0.273] |
| fraction of steps below 0.5 | 0.473 [0.345, 0.600] | 0.109 [0.036, 0.200] | 0.491 [0.364, 0.618] | 0.127 [0.055, 0.218] |
| count of steps below 0.5 | 0.382 [0.255, 0.509] | 0.109 [0.036, 0.200] | 0.345 [0.218, 0.473] | 0.109 [0.036, 0.200] |
| total suspicion, sum(1−p) | 0.382 [0.255, 0.509] | 0.109 [0.036, 0.200] | 0.364 [0.236, 0.491] | 0.109 [0.036, 0.200] |

**The current minimax selector is the worst of the eight on algorithm-generated
logs** — 0.294 agent accuracy against 0.484 for the two best, and last or
near-last on hand-crafted too. Requiring an agent's *best* step to be bad
selects whoever contributed fewest steps, since one confident step is enough to
clear them.

**No single selector wins both corpora.** On algorithm-generated it is
*mean of the 2 worst steps* and *total suspicion* (0.484 each); on hand-crafted
it is *worst single step* (0.691), where those two drop to 0.618 and 0.382.
The two corpora reward opposite things: short logs give each agent few steps, so
aggregating over two of them is informative, while long logs give the busiest
agent so many steps that any sum-like statistic just elects whoever talked most —
which is why *total suspicion* collapses from best to worst across the two.

**Mean of the 2 worst steps is the only selector that is near-best everywhere**
and is the one to prefer if a single rule has to be chosen: 0.484 / 0.452 on
algorithm-generated and 0.618 / 0.636 on hand-crafted, while also giving the
best step accuracy in the answer-shown setting (0.246).

**These are eight variants scored on the same data**, so the winners are picked
post hoc and their margins are inside the intervals — 0.484 against 0.452 for
*worst single step* is about four logs. Treat this as a diagnostic of what the
selector is sensitive to, not as a registered result.

---

## Reading the tables

**On algorithm-generated logs, P(True) locates the step better than anything
content-free.** Top-1 is 0.262 against 0.159 for always-blame-the-first-step and
0.120 for a random step — more than double chance, and the interval clears the
random control. This is the one place the signal is unambiguous.

**On hand-crafted logs it does not.** Top-1 step accuracy is 0.164 and top-2
barely moves it, to 0.200. Against a random-two-step control of 0.077 that is
still an edge, but these logs average 52 steps, so two guesses is a very narrow
window and the result rests on 9–11 logs out of 55.

**The agent column is not measuring the same thing.** On hand-crafted logs
P(True) reaches 0.691 at top-1 — the strongest cell anywhere in either table —
while its step accuracy on the same logs is 0.164. It is naming the right agent
without knowing where they went wrong, which is possible because a few agents
own most steps: *two busiest agents* alone reaches 0.891 at top-2, beating
P(True), without reading any content at all. Agent accuracy on this corpus is
largely a statement about who talks most.

**Top-2 helps the short logs and not the long ones.** Algorithm-generated step
accuracy goes 0.262 → 0.405 for a second guess; hand-crafted goes 0.164 → 0.200.
On the short logs a second guess is a meaningful widening; on 52-step logs it is
noise.

**Showing the judge the reference answer makes step localization worse**, not
better: 0.262 → 0.206 on algorithm-generated, with hand-crafted unchanged. The
answer-shown setting is supposed to be an upper bound on what the judge could
know, so this is backwards and I have no mechanism for it. It is the clearest
open question in these results.

**Where the baselines win.** *First two steps* beats P(True) on
algorithm-generated top-2 (0.429 vs 0.405), and *last step* beats it on
hand-crafted agent top-1 in the answer-shown setting. No content-free baseline
beats P(True) on algorithm-generated top-1 step accuracy, which is the cell the
method is actually for.

---

## Readout health

Earlier runs of this experiment were invalid because the judge was being asked
the question outside its chat template: the model wanted to open a reasoning
block, so the answer tokens sat in the tail of the distribution and P(True) was
a ratio between two near-zero numbers. The fix is to use the chat endpoint with
the template's own `enable_thinking: false` switch instead of prefilling a
literal `<think></think>` string.

Diagnostics from this run, all four settings:

| check | value |
|---|---|
| probability mass on {True, False} | mean **0.9992**, median 0.9997, worst log 0.948 |
| rows where neither token reached the top-20 | **0 of 8,184** |
| first-token split (answer hidden) | True 2,120 / False 1,972 |

The judge is answering the question asked, on essentially every step, and is not
collapsing onto one answer.

## Method notes

- **Judge**: `Qwen/Qwen3.6-35B-A3B`, served with vLLM (`yllm` env), scored
  through the OpenAI SDK (`Jagent` env). One generated token, `temperature=0`,
  `top_logprobs=20`.
- **P(True)** = P(true) / (P(true) + P(false)), with probability summed over
  spelling variants (`True`, ` True`, `true`, …) rather than taking the largest.
- **Prefix only.** Step *t* is judged from steps 0…*t*. Nothing after it is
  visible, so the score is not using hindsight.
- **Label indexing**: `mistake_step` is a 0-based index into the transcript.
  Verified rather than assumed — the labelled agent matches
  `history[mistake_step]` for 97.6% of algorithm-generated and 91.4% of
  hand-crafted logs, against 17.5% / 32.8% if the labels were 1-based.
- **Long transcripts**: per-step content is capped at 2,000 characters and the
  oldest steps lose their content first if a transcript exceeds 60,000
  characters. The step under judgment is never dropped.
- **No normalization, no attribution rule.** Raw P(True), ranked within each
  trajectory. Nothing is fit to the corpus.

Code: `tools/ptrue_clean.py` — standalone, sharing no code with `src/masattr`,
so agreement between the two is a real check rather than the same logic run
twice. Data: `runs/clean/ptrue_{nogt,gt}.jsonl`, 4,092 rows each.

```
bash serve.sh 0 8000
/opt/anaconda3/envs/Jagent/bin/python tools/ptrue_clean.py score <data-root> out.jsonl
/opt/anaconda3/envs/Jagent/bin/python tools/ptrue_clean.py report out.jsonl
```
