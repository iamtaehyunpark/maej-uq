# Data layout

No data is committed to this repository. Download the corpora yourself and place
them under a single root, then set `MASUQ_DATA_ROOT` (or pass `--data-root`).

```
<root>/
  matu/quick_start/results/
      conversation_logs_Math_qwen2.5.json          # MATU-CAMEL trajectories
      conversation_logs_MMLU_Autogen_qwen2.5.json  # MATU-AutoGen trajectories
      accuracy_dict_Math_qwen2.5.pkl               # per-run correctness labels
      accuracy_dict_MMLU_Autogen_qwen2.5.pkl
  whowhen/
      Algorithm-Generated/*.json                   # W&W-AG  (subset id: alg)
      Hand-Crafted/*.json                          # W&W-HC  (subset id: hc)
```

Verify with:

```bash
masuq paths
```

It prints the resolved path for every expected file and exits non-zero listing
what is missing. A few alternate filenames from earlier dumps are tried
automatically (see `masuq/config.py`); anything else can be pointed at
explicitly with `--path` / `--labels`, or with a JSON config:

```json
{
  "root": "/data",
  "camel_math": "/somewhere/else/conversation_logs_Math_qwen2.5.json",
  "camel_math_labels": "/somewhere/else/accuracy_dict_Math_qwen2.5.pkl"
}
```

```bash
masuq --config paths.json load --assert
```

## Expected shapes

The loaders assert these; if a dump differs, they fail with the offending keys
rather than guessing.

| Subset | Step fields | Types | Labels |
|---|---|---|---|
| `camel_math` | `{role, output}` | classified | per-run correctness (pkl join) |
| `autogen_mmlu` | `{role, agent, turn, type, output}` | native | per-run correctness (pkl join) |
| `alg` | `{content, role, name}` | classified | `mistake_agent/step/reason` |
| `hc` | `{content, role}` (compound role) | parsed | `mistake_agent/step/reason` |

## Pre-registered counts

`masuq load --assert` checks these (`masuq/loaders/expectations.py`):

- Who&When: **126** AG files, **58** HC files, **4092** steps combined,
  **3 + 3** files flagged `agent_step_mismatch`
- MATU: **400 tasks × 10 runs** per cell, two cells

A mismatch is reported, not silently accepted — the counts are what the pilot's
`n` is graded against.

## Supplementary cells

The MATU HuggingFace release may carry additional cells (MoreHopQA, HumanEval).
If those appear, add adapters later — spec §7.5 is explicit that this must not
block the build order.
