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
