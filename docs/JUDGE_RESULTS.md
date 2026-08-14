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
| `Llama-3.3-70B` | 70B dense |

## Answer hidden

**algorithm-generated**

| judge | method | logs | spots the step | spots the agent | no answer | calls/log |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | all_at_once | 126 | 0.270 [0.198,0.341] | 0.579 [0.492,0.667] | 1 | 1.0 |
| Qwen3.5-9B | step_by_step | 126 | 0.246 [0.175,0.325] | 0.349 [0.270,0.429] | 17 | 7.6 |
| Qwen3.5-9B | binary_search | 126 | 0.183 [0.111,0.254] | 0.397 [0.310,0.484] | 0 | 3.2 |
| Qwen3.6-35B-A3B | all_at_once | 126 | 0.198 [0.135,0.270] | 0.444 [0.357,0.532] | 0 | 1.0 |
| Qwen3.6-35B-A3B | step_by_step | 126 | 0.206 [0.135,0.278] | 0.270 [0.198,0.349] | 21 | 7.7 |
| Qwen3.6-35B-A3B | binary_search | 126 | 0.095 [0.048,0.151] | 0.357 [0.278,0.444] | 0 | 2.9 |
| Llama-3.3-70B | all_at_once | 126 | 0.111 [0.063,0.167] | 0.548 [0.460,0.635] | 0 | 1.0 |
| Llama-3.3-70B | step_by_step | 126 | 0.087 [0.040,0.143] | 0.135 [0.079,0.198] | 18 | 7.6 |
| Llama-3.3-70B | binary_search | 126 | 0.190 [0.127,0.262] | 0.413 [0.325,0.500] | 0 | 3.5 |

**hand-crafted**

| judge | method | logs | spots the step | spots the agent | no answer | calls/log |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | all_at_once | 58 | 0.034 [0.000,0.086] | 0.448 [0.328,0.586] | 0 | 1.0 |
| Qwen3.5-9B | step_by_step | 58 | 0.121 [0.052,0.207] | 0.466 [0.345,0.603] | 5 | 12.4 |
| Qwen3.5-9B | binary_search | 58 | 0.069 [0.017,0.138] | 0.552 [0.414,0.672] | 0 | 5.0 |
| Qwen3.6-35B-A3B | all_at_once | 58 | 0.017 [0.000,0.052] | 0.190 [0.086,0.293] | 0 | 1.0 |
| Qwen3.6-35B-A3B | step_by_step | 58 | 0.103 [0.034,0.190] | 0.552 [0.431,0.690] | 1 | 8.8 |
| Qwen3.6-35B-A3B | binary_search | 58 | 0.155 [0.069,0.259] | 0.621 [0.500,0.741] | 0 | 4.9 |
| Llama-3.3-70B | all_at_once | 58 | 0.052 [0.000,0.121] | 0.638 [0.517,0.776] | 0 | 1.0 |
| Llama-3.3-70B | step_by_step | 58 | 0.121 [0.052,0.207] | 0.448 [0.328,0.569] | 8 | 14.9 |
| Llama-3.3-70B | binary_search | 58 | 0.017 [0.000,0.052] | 0.483 [0.362,0.621] | 0 | 5.4 |


## Answer shown

**algorithm-generated**

| judge | method | logs | spots the step | spots the agent | no answer | calls/log |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | all_at_once | 126 | 0.214 [0.143,0.286] | 0.532 [0.444,0.619] | 2 | 1.0 |
| Qwen3.5-9B | step_by_step | 126 | 0.294 [0.214,0.373] | 0.421 [0.333,0.508] | 9 | 7.6 |
| Qwen3.5-9B | binary_search | 126 | 0.222 [0.151,0.302] | 0.365 [0.278,0.444] | 0 | 3.1 |
| Qwen3.6-35B-A3B | all_at_once | 126 | 0.183 [0.119,0.254] | 0.421 [0.333,0.508] | 0 | 1.0 |
| Qwen3.6-35B-A3B | step_by_step | 126 | 0.206 [0.135,0.278] | 0.294 [0.214,0.373] | 14 | 7.6 |
| Qwen3.6-35B-A3B | binary_search | 126 | 0.103 [0.056,0.159] | 0.397 [0.310,0.476] | 0 | 2.9 |
| Llama-3.3-70B | all_at_once | 126 | 0.111 [0.063,0.167] | 0.587 [0.500,0.667] | 0 | 1.0 |
| Llama-3.3-70B | step_by_step | 126 | 0.159 [0.103,0.230] | 0.214 [0.151,0.294] | 7 | 7.6 |
| Llama-3.3-70B | binary_search | 126 | 0.159 [0.095,0.230] | 0.381 [0.302,0.476] | 0 | 3.4 |

**hand-crafted**

| judge | method | logs | spots the step | spots the agent | no answer | calls/log |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | all_at_once | 58 | 0.034 [0.000,0.086] | 0.534 [0.414,0.655] | 0 | 1.0 |
| Qwen3.5-9B | step_by_step | 58 | 0.103 [0.034,0.190] | 0.552 [0.431,0.672] | 2 | 11.1 |
| Qwen3.5-9B | binary_search | 58 | 0.069 [0.017,0.138] | 0.500 [0.379,0.621] | 0 | 5.0 |
| Qwen3.6-35B-A3B | all_at_once | 58 | 0.069 [0.017,0.138] | 0.121 [0.052,0.207] | 0 | 1.0 |
| Qwen3.6-35B-A3B | step_by_step | 58 | 0.069 [0.017,0.138] | 0.552 [0.431,0.672] | 3 | 10.8 |
| Qwen3.6-35B-A3B | binary_search | 58 | 0.121 [0.052,0.207] | 0.638 [0.517,0.759] | 0 | 4.9 |
| Llama-3.3-70B | all_at_once | 58 | 0.052 [0.000,0.121] | 0.672 [0.552,0.793] | 0 | 1.0 |
| Llama-3.3-70B | step_by_step | 58 | 0.138 [0.052,0.241] | 0.466 [0.345,0.603] | 10 | 16.9 |
| Llama-3.3-70B | binary_search | 58 | 0.034 [0.000,0.086] | 0.500 [0.379,0.638] | 0 | 5.4 |

## Appendix — what binary search actually decided

Each call asks the judge to pick the upper or lower half. A judge that answers the same direction every time converges on one end of the log whatever it contains, which is indistinguishable from localisation unless the individual choices are recorded.

| judge | answer | 'lower' | 'upper' | unreadable | converged to step 0 |
|---|---|---|---|---|---|
| Qwen3.5-9B | hidden | 57% | 43% | 0 | 12/184 |
| Qwen3.5-9B | shown | 62% | 38% | 0 | 6/184 |
| Qwen3.6-35B-A3B | hidden | 81% | 19% | 0 | 1/184 |
| Qwen3.6-35B-A3B | shown | 80% | 20% | 0 | 1/184 |
| Llama-3.3-70B | hidden | 23% | 77% | 0 | 78/184 |
| Llama-3.3-70B | shown | 29% | 71% | 0 | 57/184 |

## How this was produced

| item | value |
|---|---|
| serving | vLLM (`yllm` env); 70B tensor-parallel over 2 GPUs |
| client | OpenAI SDK (`Jagent` env), `temperature=0` |
| reasoning | disabled via `chat_template_kwargs={'enable_thinking': false}` |
| code | `tools/llm_judge.py`; rebuild this file with `tools/judge_report.py` |

Parsing accepts both the labelled form the prompt requests (`Agent Name:` / `Step Number:`) and the numbered form judges often return instead (`1. <agent>` / `2. <n>`); scoring only the former would charge a formatting mismatch against the method. Where step-by-step walks a whole log without ever flagging an error, the paper leaves the outcome unspecified — it is counted as *no answer* and scored as a miss rather than defaulted to a step.
