# MAS-Attribution Harness Spec v3
**2026-08-07 · consolidated. Supersedes the v2 harness spec and the v2.1
severance amendment in full; both are withdrawn and this document is the single
source of truth.**

This project is standalone research. Nothing here depends on, ports from, or
refers to any other project. Motivating hypotheses are stated as hypotheses this
project's own ablations test, or attributed to published literature.

---

## Part A — Scope

Failure attribution over frozen multi-agent trajectories: given one realized
Who&When failure log, name the fault agent and the decisive step. No
trajectory-level aggregation anywhere — the per-step field is read pointwise and
localized, never summed.

Out of scope: trajectory-level uncertainty, cross-step aggregation, multi-run or
resampling methods, trained tracers, online monitoring.

## Part B — Codebase

1. Greenfield package, flat layout. No config frameworks, no plugin registries,
   no abstract base classes with a single subclass; argparse-only `run.py` per
   experiment; modules split by function at ~300 lines; pandas only in `eval/`.
2. Generic implementations, written plainly: judge client (model load, logit
   extraction for the single-token readout, batching, KV prefix sharing), prompt
   assembly, bootstrap-CI and reliability utilities.
3. The Who&When baseline repository is a dependency, not a fork. Its
   `inference.py` is imported and its three functions called; nothing is patched
   except credentials passed via CLI flags. Its `evaluate.py` is not imported —
   the substring scorer is re-implemented so the comparability row is a choice.
4. Data flows through one frozen record type. Every stage consumes and returns
   it, or plain arrays keyed by `(file_id, step_idx)`.
5. Artifacts are frozen with content hashes under `specs/`, logged per run.
6. `BUILD_REPORT.md`: inventory, LOC, and an explicit statement that no external
   project's code was vendored. **Target ≤4k LOC** excluding specs and tests.

## Part C — Harness

### 1. Unified record

```json
{
  "dataset": "whowhen", "subset": "alg|hc", "file_id": str,
  "query": str, "ground_truth": str,
  "steps": [{"idx": int, "agent": str, "role_raw": str,
             "type_norm": "plan|delegate|execute|final|unknown",
             "type_source": "parsed|classified", "content": str}],
  "label_mistake_agent": str, "label_mistake_step": int,
  "label_mistake_reason": str, "flags": [str]
}
```

Loader asserts: **hard-fail on count mismatch** — 126 AG files, 58 HC files,
4092 total steps, 3 + 3 files flagged `agent_step_mismatch`. **Flag on
record-level anomalies** — a step with empty content, or a `mistake_step`
outside trajectory bounds.

The release contains five such anomalies (3 HC out-of-range, 2 AG empty-content)
while requiring the 126/58 counts, so both asserts cannot be hard. Anomalous
records are loaded, flagged with a class distinct from `agent_step_mismatch`,
dual-reported with and without, and their file ids logged in the run manifest.
`--anomaly-policy {flag,fail,drop}`; **`flag` is the registered default**.

### 2. Typing — hierarchical

- **HC (parsed)**: `Orchestrator (thought)`→plan; `Orchestrator (-> X)`→delegate
  (agent := Orchestrator); `WebSurfer|Assistant|FileSurfer`→execute; a final
  answer-emission step→final.
- **AG (classified)**: rules — JSON-plan detection, delegation-verb patterns,
  answer-emission detection, tool-output detection; `unknown` permitted.
- **Validation gate, before any use on AG**: run the rules on HC, where types
  are parsed and therefore known. Confusion matrix is a reportable table; ≥90%
  agreement required.
- **Measured**: 0.547 overall on 2935 HC parsed steps. The confusion is not
  diffuse — coordination / execute / final splits at 0.9935, plan vs delegate
  inside coordination at 0.4162, *below* the 0.6934 majority-class baseline. In
  the Magentic-One idiom that distinction lives in the role, not the text.
- **Resolution**: rules keep the coarse split; an LLM classifier takes only the
  plan/delegate sub-split (~76% of HC parsed steps). The splitter is itself
  gated on HC and must beat both 0.90 **and** the majority-class baseline before
  it may touch AG. Collapsing plan+delegate was rejected — it buys 0.994 by
  deleting the delegation-error prediction and the orchestrator/worker analysis.
  Tuning the rules on HC's ledger markers was rejected as circular.

### 3. Judge

- Prefix-conditional: evidence(t) = query + ordered typed steps 0..t (agent +
  type + content). The with-GT arm appends the ground truth to the preamble;
  both arms run for all primary tables.
- Readout: single-token P(True) logit score (primary); verbalized numeric
  confidence and per-step binary verdict under the **identical** prompt
  scaffold, as the E2 ablation rows.
- Type-conditioned evidence: execute steps with content <16 chars receive a
  pointer to the assigned subtask and to same-turn peer steps. The two pointers
  are independently switchable, because E5 ablates them separately.
- Judge id: a Qwen3.6-35B-class open-weight checkpoint, exact id in
  `specs/judge.json`. Rationale on standalone merits: per-token logit access for
  the readout, mid scale so a 130-step trajectory fits the prefix budget on
  available hardware, cheap enough to run every ablation arm over both subsets.
- The type-classifier must be family-disjoint from **both** judge families.
  Families are resolved and recorded per run.
- KV prefix sharing is mandatory and asserted, not assumed. Tokens and
  wall-clock are logged per trajectory.
- **Truncation, registered**: whenever the assembled prefix exceeds budget,
  retain the query, the ground truth (with-GT arm), and all plan/delegate/final
  steps verbatim, plus the most recent execute steps; reduce the oldest execute
  contents to headers (agent + type + first 120 chars), never dropping a step
  row. Log truncation events per assessment; report the truncated fraction per
  subset. Measured on the release: 24.1% of HC trajectories, 13.8% of
  assessments.

### 4. Normalization — leave-one-out CV

Per-type normalization statistics (mean/sd of the raw score per `type_norm`) and
a crossing threshold, fit by **leave-one-file-out CV within each subset**: one
fold per file, statistics fit on the rest, the held-out file scored under them.
Fold order is the sorted file key, so the procedure is deterministic without a
seed. Subsets are normalized independently.

`apply` z-scores each step under its fold and writes `p_norm`; the raw score
stays on the row. Typed vs pooled statistics are two arms of one fit — that is
E4's typing on/off arm.

Per-step labels are derived, because Who&When annotates one decisive mistake per
trajectory rather than every step. Two registered policies: `prefix` (default —
steps `0..mistake_step`, `correct = idx < mistake_step`, excluding the
post-mistake tail rather than guessing it) and `point` (every step,
`correct = idx != mistake_step`).

### 5. Attribution

**Primary rule — `changepoint_single`.** A two-regime mean-shift split of the
per-step score sequence. The decisive step is the **first step of regime 2**;
the fault agent is its owner. The split is chosen by a contrast statistic

    (mean(p[:k]) − mean(p[k:])) / (s_pooled · sqrt(1/k + 1/(n−k)))

so a split isolating a couple of endpoint steps cannot outrank a genuine regime
change on raw gap alone. **Fallback to argmin** when the trajectory is too short
to hold two regimes, when the best split sits at a boundary, or when the best
contrast is below the registered minimum. The fallback condition lives in
`specs/criteria.json` and must be marked `registered` before any attribution
number is computed — it is part of the rule's definition.

The primary rule is fixed by `specs/rule_directive.md`, whose hash is logged as
`rule_provenance` on every run. **No experiment's outcome selects it.**

**E3 ablation rows**: `argmin`; `first_crossing` on the leave-one-out threshold;
`changepoint` (the same split chosen by an unnormalised mean gap, ablating the
contrast statistic); `agent_first`; `relative_crossing` at k ∈ {1.5, 2, 2.5}.
`RELATIVE_K` is deliberately unregistered — a demoted rule shows its sensitivity
rather than resting on one value.

Disagreement analysis: step-first vs agent-first, stratified by `type_norm` and
by orchestrator/worker (HC only).

### 6. Baselines and scoring

- Reproduce all_at_once / step_by_step / binary_search under (a) gpt-4o, their
  regime, and (b) our judge, as the capability control. Credentials via CLI
  flags.
- Surrogate-intrinsic: proxy-LM per-step sequence logprob and mean token entropy
  over step content, fed through the same rules.
- Scorers: exact-match (int) primary; the substring re-implementation as a
  comparability row, its `"1" in "12"` artifact footnoted. Dual reporting with
  and without the 6 flagged files and the 5 anomalous ones. Bootstrap CIs over
  files on everything.

### 7. Experiment manifest

E0 **sanity-only** (score-field degeneracy, per-type distributions, stability
reporting for the leave-one-out ablation arm; **decides nothing**) → E1 primary
vs 3 baselines (both judges, both subsets, both GT arms) → E2 readout ablation →
E3 rule ablation (incl. the k sweep) → E4 typing on/off (typed vs pooled
normalization) → E5 evidence ablation (+ hindsight ceiling figure) → E6
judge-family sensitivity → E7 surrogate-intrinsic → [E8 success-control, gated,
not built] → E9 uniformity stratification from E1 outputs.

## Part D — Gated

One loader for a few hundred successful third-party MAS runs, producing a single
figure: score-field distribution on successes vs failures. Build only on
explicit go. Not built.

## Part E — Pre-registered

Primary metrics: exact-match agent-accuracy and step-accuracy, per subset, per
GT arm. Flagged-file and anomalous-file dual reporting. Substring-scorer row
labeled comparability, never primary. Prompt, type-rule, criteria, judge, and
rule-directive hashes logged per run. `specs/criteria.json` registered before
any attribution number; `specs/judge.json` confirmed before any reported run.

## Part F — Language rule

No emitted file — code, comments, specs, reports, figures — refers to any other
project, its corpus, its harness, or its results.
