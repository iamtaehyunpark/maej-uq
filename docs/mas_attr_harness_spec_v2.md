# MAS-Attribution Harness Spec v2 + Codebase Diet
**2026-08-06 · attribution-only rescope (supersedes pilot spec v1 §§0–8 where they conflict)**

## Part A — What changed from v1 (delete-list first)

The trajectory-level track is gone. The implementing agent MUST NOT port or
build:
- ✗ MATU loaders (both cells: CAMEL×MATH, AutoGen×MMLU) — EXCEPT the single
  optional success-control adapter (Part D), which is gated behind an
  explicit owner decision and is not part of the default build.
- ✗ accuracy_dict_*.pkl label joins, per-run correctness plumbing.
- ✗ noisy-OR aggregation, trajectory-level U, AUROC/AUARC-vs-completion
  eval code.
- ✗ MATU-published-numbers comparability tables.
- ✗ Any cross-step aggregation utility beyond what the attribution rules in
  Part C consume pointwise.

Everything retained from v1: Who&When adapters, typing layer, judge harness
core, dual scorer, flags, pre-registration items — restated below as the
single source of truth so v1 does not need to be consulted.

## Part B — Codebase diet (instructions to the implementing agent)

The port source is paper 1's harness. Diet rules, in order:

1. **Port by allowlist, not by copy.** Do not vendor the paper-1 repo and
   delete; start an empty package and pull in ONLY:
   - the judge client (model load / API wrapper, logit extraction for the
     P(True) readout, batching, KV prefix-sharing path);
   - the prompt-assembly layer (prefix construction), to be re-templated for
     MAS steps;
   - the calibration module (per-type score→probability fit/apply,
     serialization of frozen maps);
   - bootstrap-CI and reliability-diagram utilities.
2. **Do NOT port** (paper-1 machinery with no consumer here): environment
   wrappers (ALFWorld/HotpotQA), agent-generation code, trajectory
   regeneration, the τ action-typing rules (environment-consequence based —
   MAS typing is new code, not an adaptation), evidence-grid decoy
   apparatus, loop-stratification analysis, noisy-OR combiner, judge-size
   curve scaffolding, hindsight-harness beyond the single context-swap flag
   needed for the ceiling figure.
3. **One package, flat layout, no framework.** Target shape:
   ```
   masattr/
     loaders/      # whowhen_ag.py, whowhen_hc.py, (success_control.py — gated)
     typing/       # normalize.py (rules), validate.py (confusion vs native/parsed)
     judge/        # client.py, prompts.py, score.py   (ported, re-templated)
     calib/        # fit.py (paper-1 corpus), apply.py, frozen/*.json
     attribute/    # rules.py (first_crossing, argmin, changepoint, agent_first)
     eval/         # scorers.py (exact + substring-repro), ci.py
     baselines/    # thin wrappers around the Who&When repo's three methods
     specs/        # this file + frozen prompt/calibration hashes
   ```
   No plugin registries, no config-framework (hydra etc.), no abstract base
   classes with one subclass. A single `run.py` per experiment, argparse
   only. If a module exceeds ~300 lines, split by function, not by
   abstraction.
4. **Data flows through one record type.** The unified record (Part C §1) is
   a frozen dataclass; every stage consumes and returns it or plain arrays
   keyed by (file_id, step_idx). No pandas until eval/.
5. **The Who&When baseline repo is a dependency, not a fork.** Wrap their
   `inference.py` via subprocess or import their three functions directly;
   patch nothing except passing credentials via CLI flags (their env-var
   fallback is documented but not implemented). Their substring scorer is
   re-implemented in eval/scorers.py from the four lines in evaluate.py —
   do not import their eval path.
6. **Freeze artifacts, not code paths.** Prompts, type-rule tables, and
   calibration maps are serialized under specs/ with content hashes logged
   in the run manifest. A run is reproducible from (commit, manifest) alone.
7. **Deletion receipts.** At the end of the port, emit a one-page
   PORT_REPORT.md: what was pulled in (file → origin), what was explicitly
   not ported (the ✗ list above + paper-1 exclusions), and LOC of the
   resulting package. Target: the whole package under ~2.5k LOC excluding
   specs and tests. If over, cut abstractions, not features.

## Part C — Harness spec (attribution-only)

### 1. Unified record
```json
{
  "dataset": "whowhen",
  "subset": "alg|hc",
  "file_id": str,
  "query": str,
  "ground_truth": str,          // for the with-GT attribution setting
  "steps": [{
    "idx": int,
    "agent": str,               // AG: name field; HC: parsed from compound role
    "role_raw": str,
    "type_norm": "plan|delegate|execute|final|unknown",
    "type_source": "parsed|classified",
    "content": str
  }],
  "label_mistake_agent": str,
  "label_mistake_step": int,    // string→int cast at load; hard-fail on cast error
  "label_mistake_reason": str,
  "flags": ["agent_step_mismatch", ...]   // 3 AG + 3 HC known files
}
```
Loader asserts (**amended 2026-08-07, after the release was loaded**):
hard-fail on **count** mismatch — 126 AG files / 58 HC files / 4092 total
steps; **flag** on record-level anomalies — a step with empty content, or a
`mistake_step` outside trajectory bounds.

Reason for the amendment: the release violates the original per-step asserts
for five files (3 HC with `mistake_step` past the trajectory end —
51 of 28, 8 of 5, 24 of 19 — and 2 AG with an empty-content step) while
simultaneously requiring the 126/58 counts. Both could not hold. Anomalous
records are therefore loaded, flagged with a class distinct from the six
`agent_step_mismatch` files, dual-reported with and without, and their file
ids logged in the run manifest. `--anomaly-policy {flag,fail,drop}` selects
the behaviour; `flag` is the pre-registered default.

### 2. Typing layer
- HC (parsed): `Orchestrator (thought)`→plan; `Orchestrator (-> X)`→delegate
  (agent:=Orchestrator); `WebSurfer|Assistant|FileSurfer`→execute; final
  answer-emission step→final.
- AG (classified): rule-based v1 — JSON-plan detection, delegation-verb
  patterns, answer-emission detection, tool-output detection; unknown
  allowed. Escalate to LLM classifier only if rules cover <90% of a
  100-step manual audit.
- Validation gate (before any use on AG): run rules on HC where parsed
  types are known; confusion matrix; ≥90% agreement required. The confusion
  matrix is a reportable table.
- **Measured 2026-08-07 (2935 HC parsed steps): rules score 0.547 overall.**
  The confusion is not diffuse — coordination / execute / final splits at
  0.9935, while plan vs delegate inside coordination splits at 0.4162,
  *below* the 0.6934 majority-class baseline. In the Magentic-One idiom that
  distinction lives in the role, not the text.
- **Resolution — hierarchical typing.** Rules keep the coarse split; an LLM
  classifier takes only the plan/delegate sub-split within coordination
  (~76% of HC parsed steps). Collapsing plan+delegate was rejected: it buys
  0.994 by deleting the delegation-error prediction (§5) and the
  orchestrator/worker analysis. Tuning the rules on HC ledger markers was
  rejected as circular — tuning on HC's idiom to license AG. The splitter is
  itself gated on HC, and must beat both 0.90 and the majority-class
  baseline before it may touch AG.

### 3. Judge
- Prefix-conditional: evidence(t) = query + ordered typed steps 0..t
  (agent + type + content). With-GT setting appends ground_truth to the
  prompt preamble; both settings run for all primary tables.
- Readout: single-token P(True)-style logit score (primary); verbalized
  numeric confidence and per-step binary verdict under the IDENTICAL prompt
  scaffold (baseline rows for the readout ablation).
- Type-conditioned evidence: execute-steps with content <16 chars get an
  explicit pointer to the assigned subtask + same-turn peer steps.
- Judges: primary mid-size open model with logprobs (paper-1 choice);
  second family for sensitivity. gpt-4o used ONLY inside the baseline
  reproduction, not for our field.
- Cost logging: tokens + wall-clock per trajectory (HC reaches 130 steps —
  KV prefix sharing mandatory; assert the shared-prefix path is active).
- **Truncation policy (pre-registered 2026-08-07).** HC's longest trajectory
  reaches ~38k estimated tokens of prefix, so capacity is not relied on.
  Type-aware retention: always keep the query, the ground truth (with-GT
  setting), and every plan/delegate step verbatim — they are structural and
  short; keep the most recent execute steps verbatim; when over budget,
  demote the *oldest* execute steps to a one-line header (agent + type +
  first 120 chars), never dropping the step row. Prefix-only asymmetry is
  preserved; what degrades is old execution detail. Truncation events are
  logged per assessment and the truncated fraction reported; a material
  fraction on HC becomes a limitations sentence.

### 4. Calibration
- Fit per-type score→probability maps ONCE on paper 1's ~30k step-labeled
  single-agent corpus (mapping paper-1 action/thought steps into the
  function-type space: rule table in specs/, frozen before Exp-0).
- **Handoff schema (fixed 2026-08-07)**: one JSONL row per judged step —
  `{step_id, arm, model, benchmark, step_kind, tau{info, world_mod,
  reversible, cost}, p_raw, judge_model, label_correct}`. `p_raw` is the raw
  single-probe P(True) under the arm matching ours, never a combined score.
  Rows are filtered to `judge_model` == the transferring judge. The
  step_kind×tau → function-type table lives in
  `specs/paper1_type_map.json` and must be marked `"status": "frozen"`
  before E0 runs; the loader refuses a draft.
- **Per-type thresholds**: the crossing threshold is fit per type alongside
  the maps (§5's "crosses *its* calibrated threshold"), with the pooled
  threshold as fallback and as E4's global-threshold arm.
- Freeze to calib/frozen/. Apply unchanged to Who&When.
- **Exp-0 (run first, pre-registered)**: single-agent→MAS transfer check —
  reliability diagrams on a 20-file held-aside slice (10 AG + 10 HC, chosen
  by seeded RNG, then returned to the test pool ONLY for non-calibration
  analyses; primary numbers reported with and without these 20).
  Fail ⇒ disclosed fallback: leave-one-out CV on Who&When; uniformity claim
  weakened accordingly in the paper text.

### 5. Attribution rules (attribute/rules.py)
- Primary: first-crossing on type-normalized p_t; fault agent = owner(t*).
- Ablations: argmin; changepoint (PELT or binary-seg on p_t, one
  hyperparam, frozen); agent-first two-stage (per-agent max p as selector →
  first-crossing within selected agent's steps).
- Disagreement analysis: step-first vs agent-first, stratified by type_norm
  and orchestrator/worker (HC only).

### 6. Baselines & scoring
- Reproduce all_at_once / step_by_step / binary_search: (a) gpt-4o (their
  regime), (b) our judge (capability control). Credentials via CLI flags.
- Surrogate-intrinsic: proxy-LM per-step sequence logprob + mean token
  entropy on content; argmin/first-crossing applied identically.
- Scorers: exact-match (int) primary; substring re-implementation as
  comparability row. Dual reporting with/without the 6 flagged files.
  Bootstrap CIs (over files) on everything.

### 7. Experiment manifest (order)
E0 calibration transfer → E1 primary vs 3 baselines (both judges, both
subsets, both GT settings) → E2 readout ablation → E3 rule ablation →
E4 typing on/off → E5 evidence ablation (+ hindsight ceiling figure) →
E6 judge-family sensitivity → E7 surrogate-intrinsic → [E8 success-control,
gated] → E9 uniformity stratification (computed from E1 outputs, no new
runs).

## Part D — Gated: success-control adapter (§7.ix of proposal, pending owner decision)
One loader (`loaders/success_control.py`) for a few hundred successful
third-party MAS runs (candidate source: MATU released AutoGen cell,
correct-labeled runs only — used as raw successful trajectories, no labels
plumbing, no comparison, no track). Output: one figure — p_t distribution /
crossing-rate on successes vs failures. Build ONLY on explicit go.

## Part E — Pre-registered before first full run
Primary metrics: exact-match agent-acc + step-acc, per subset, per GT
setting. Exp-0 outcome decides calibration fallback BEFORE any attribution
numbers are seen. Flagged-file dual reporting. Held-aside-20 dual
reporting. Substring-scorer row labeled as comparability, never primary.
Prompt/calibration/type-rule hashes logged per run.
