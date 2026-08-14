# LLM-judge failure attribution — three judges, three methods

The three attribution methods from *Which Agent Causes Task Failures and When?*
(arXiv:2505.00212), run against three local judges. Prompts are transcribed
verbatim from the paper's Appendix G and the control flow from Appendix A;
each method has a without-answer and a with-answer variant, as the paper gives
two prompts per method.

**These are not the paper's numbers.** The paper uses GPT-4o. Everything here
is a local open-weights judge, so the grid is a judge-capability comparison
under the paper's own methods, not a reproduction of its figures.

Corpus: Who&When — 126 algorithm-generated logs and 58 hand-crafted ones.
*Spots the step* is exact match against the annotated mistake step, *spots the
agent* against the annotated agent. Intervals are bootstrapped over logs, 2,000
resamples. `mistake_step` is 0-based, matching the paper's own convention in
G.1.

| judge | size |
|---|---|
| `Qwen3.5-9B` | 9B dense |
| `Qwen3.6-35B-A3B` | 35B MoE, ~3B active |

## Answer hidden

**algorithm-generated**

| judge | method | logs | spots the step | spots the agent | no answer | calls/log |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | all_at_once | 126 | 0.159 [0.095,0.222] | 0.524 [0.429,0.611] | 0 | 1.0 |
| Qwen3.5-9B | step_by_step | 126 | 0.246 [0.175,0.325] | 0.373 [0.294,0.460] | 12 | 2.6 |
| Qwen3.5-9B | binary_search | 126 | 0.143 [0.087,0.206] | 0.484 [0.397,0.571] | 0 | 3.6 |
| Qwen3.6-35B-A3B | all_at_once | 126 | 0.167 [0.103,0.238] | 0.643 [0.556,0.722] | 0 | 1.0 |
| Qwen3.6-35B-A3B | step_by_step | 126 | 0.270 [0.190,0.349] | 0.413 [0.325,0.500] | 18 | 3.0 |
| Qwen3.6-35B-A3B | binary_search | 126 | 0.159 [0.095,0.222] | 0.500 [0.413,0.587] | 0 | 3.6 |

**hand-crafted**

| judge | method | logs | spots the step | spots the agent | no answer | calls/log |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | all_at_once | 58 | 0.000 [0.000,0.000] | 0.517 [0.397,0.655] | 0 | 1.0 |
| Qwen3.5-9B | step_by_step | 58 | 0.138 [0.052,0.224] | 0.569 [0.448,0.690] | 1 | 7.0 |
| Qwen3.5-9B | binary_search | 58 | 0.000 [0.000,0.000] | 0.017 [0.000,0.052] | 0 | 5.6 |
| Qwen3.6-35B-A3B | all_at_once | 58 | 0.069 [0.017,0.138] | 0.345 [0.224,0.466] | 0 | 1.0 |
| Qwen3.6-35B-A3B | step_by_step | 58 | 0.086 [0.017,0.155] | 0.431 [0.310,0.552] | 3 | 9.0 |
| Qwen3.6-35B-A3B | binary_search | 58 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0 | 5.7 |


## Answer shown

**algorithm-generated**

| judge | method | logs | spots the step | spots the agent | no answer | calls/log |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | all_at_once | 126 | 0.127 [0.071,0.190] | 0.492 [0.405,0.579] | 0 | 1.0 |
| Qwen3.5-9B | step_by_step | 126 | 0.214 [0.143,0.294] | 0.341 [0.254,0.429] | 4 | 2.8 |
| Qwen3.5-9B | binary_search | 126 | 0.127 [0.071,0.183] | 0.468 [0.381,0.556] | 0 | 3.5 |
| Qwen3.6-35B-A3B | all_at_once | 126 | 0.135 [0.079,0.198] | 0.659 [0.571,0.738] | 0 | 1.0 |
| Qwen3.6-35B-A3B | step_by_step | 126 | 0.302 [0.222,0.381] | 0.397 [0.310,0.484] | 5 | 2.9 |
| Qwen3.6-35B-A3B | binary_search | 126 | 0.159 [0.095,0.222] | 0.500 [0.413,0.587] | 0 | 3.6 |

**hand-crafted**

| judge | method | logs | spots the step | spots the agent | no answer | calls/log |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | all_at_once | 58 | 0.000 [0.000,0.000] | 0.534 [0.414,0.672] | 0 | 1.0 |
| Qwen3.5-9B | step_by_step | 58 | 0.121 [0.034,0.207] | 0.603 [0.483,0.724] | 0 | 7.8 |
| Qwen3.5-9B | binary_search | 58 | 0.000 [0.000,0.000] | 0.069 [0.017,0.138] | 0 | 5.6 |
| Qwen3.6-35B-A3B | all_at_once | 58 | 0.069 [0.017,0.138] | 0.414 [0.293,0.534] | 0 | 1.0 |
| Qwen3.6-35B-A3B | step_by_step | 58 | 0.138 [0.052,0.224] | 0.431 [0.293,0.552] | 8 | 14.7 |
| Qwen3.6-35B-A3B | binary_search | 58 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0 | 5.7 |

## Appendix — what binary search actually decided

Each call asks the judge to pick the upper or lower half. A judge that answers the same direction every time converges on one end of the log whatever it contains, which is indistinguishable from localisation unless the individual choices are recorded.

| judge | answer | 'lower' | 'upper' | unreadable | converged to step 0 |
|---|---|---|---|---|---|
| Qwen3.5-9B | hidden | 97% | 3% | 0 | 163/184 |
| Qwen3.5-9B | shown | 96% | 4% | 0 | 156/184 |
| Qwen3.6-35B-A3B | hidden | 0% | 0% | 789 | 184/184 |
| Qwen3.6-35B-A3B | shown | 0% | 0% | 789 | 184/184 |


## Llama-3.3-70B — ran, but the output is not usable

The 70B completed all 1,104 assessments, and the numbers are being withheld
rather than reported, because the model was emitting degenerate text rather
than answers:

| arm | corpus | answers that parsed |
|---|---|---|
| answer hidden | algorithm-generated | 125 / 126 |
| answer hidden | hand-crafted | **0 / 58** |
| answer shown | algorithm-generated | **0 / 126** |
| answer shown | hand-crafted | **0 / 58** |

The failure is not prompt-specific. Served fresh and asked for a raw completion
of `"The capital of France is"`, it returns `" other other other other ..."`.
Same for a one-line chat request, with and without a system message, and with
the reasoning switch on or off. `step_by_step` answered "No" on 367 of 367
logs, which is the same collapse seen through a Yes/No parser.

Ruled out by testing:

- **Prompt length** — it fails on a 1,373-token prompt as readily as a 22k one.
- **The with-answer prompt** — plain `"Say OK."` fails identically.
- **NCCL peer-to-peer transport** — `NCCL_P2P_DISABLE=1` changes nothing.
- **Checkpoint integrity** — 723 tensors across 30 shards, none missing, no
  NaNs in the embedding, sane magnitudes. The apparent 84KB size mismatch
  against the index is safetensors headers, which `total_size` excludes.

That leaves the serving path: vLLM 0.23 executing this checkpoint under
tensor parallelism on this host. The natural next test is `--enforce-eager`,
which disables CUDA graphs and inductor compilation; that run was interrupted
before it reported. Loading the model through transformers with a device map
would separate a vLLM bug from a hardware fault, but takes a 132GB load to
find out.

Two smaller results survive from the same run and are worth keeping in mind if
it is retried: the earliest binary-search probe against a freshly started 70B
returned clean `'upper half'` answers, and the first 125 algorithm-generated
`all_at_once` answers parsed correctly. So the collapse is not present from the
first token of a server's life, which is more consistent with an execution
fault than a bad checkpoint.

## How this was produced

| item | value |
|---|---|
| serving | vLLM (`yllm` env); 70B tensor-parallel over 2 GPUs |
| client | OpenAI SDK (`Jagent` env), `temperature=0` |
| reasoning | disabled via `chat_template_kwargs={'enable_thinking': false}` |
| code | `tools/llm_judge.py`; rebuild this file with `tools/judge_report.py` |

Parsing accepts both the labelled form the prompt requests (`Agent Name:` / `Step Number:`) and the numbered form judges often return instead (`1. <agent>` / `2. <n>`); scoring only the former would charge a formatting mismatch against the method. Where step-by-step walks a whole log without ever flagging an error, the paper leaves the outcome unspecified — it is counted as *no answer* and scored as a miss rather than defaulted to a step.
