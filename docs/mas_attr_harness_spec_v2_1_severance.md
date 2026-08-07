# Harness Spec v2.1 — Severance Amendment (A23)
**2026-08-07 · applies on top of mas_attr_harness_spec_v2.md; where they
conflict, this document wins. Scope: this project is standalone research —
no dependency on, port from, or reference to the prior single-agent UQ
project ("paper 1") anywhere in code, specs, comments, or reports.**

## 1. Part B (codebase diet) — REPLACED

Old Part B instructed a port-by-allowlist from the paper-1 harness. Replace
with: **build fresh**. The diet rules that survive, restated without the
port framing:

1. Greenfield package, same target layout as v2 Part B.3, with `calib/`
   renamed `normalize/` (see §3 below).
2. No config frameworks, no plugin registries, no abstract base classes
   with one subclass; argparse-only `run.py` per experiment; modules split
   by function at ~300 lines; pandas only in eval/.
3. Generic implementations of: judge client (HF/vLLM load, logit
   extraction for single-token readout, batching, KV prefix sharing),
   prompt assembly, bootstrap-CI + reliability-diagram utilities. Write
   them plainly; do not copy from any prior project tree, and do not
   reference prior-project paths in comments or docs.
4. Who&When baseline repo remains a dependency, not a fork (v2 Part B.5
   unchanged).
5. Freeze artifacts with content hashes (v2 Part B.6 unchanged).
6. PORT_REPORT.md is replaced by BUILD_REPORT.md: package inventory, LOC,
   and an explicit statement that no prior-project code was vendored.
   Target ≤2.5k LOC unchanged.

## 2. Part C.4 (calibration) — REPLACED by normalization-via-CV

Delete: paper-1 corpus fit, frozen transfer maps, the step_kind/tau
mapping table, the E0 transfer test, the paper-1 JSONL adapter (never
build it).

Replace with `normalize/`:
- **fit.py**: per-type normalization statistics (mean/sd of s_t per
  type_norm) and crossing threshold, fit by leave-one-out CV over files
  within each Who&When subset (fold = one file held out; statistics fit on
  the rest; the held-out file scored under those statistics). Seeded,
  deterministic fold order.
- **apply.py**: z-score s_t per type under the fold's statistics; expose
  both normalized and raw fields downstream.
- **E0 (redefined, run first, pre-registered)**:
  (a) score-field sanity — s_t distributions per type/subset, degenerate-
  field checks (constant scores, saturation);
  (b) threshold stability — cross-fold variance of per-type statistics and
  of the crossing threshold; report coefficient of variation per type.
  **Pre-fixed decision criterion (execute before any attribution numbers
  are computed)**: if threshold CV exceeds the registered bound (set in
  specs/e0_criteria.json before E0 runs), primary rule switches from
  first-crossing to the threshold-free set (argmin / changepoint /
  within-trajectory relative crossing), and first-crossing demotes to
  ablation. Log the decision + criterion hash in the run manifest.

## 3. Part C.3 (judge) — AMENDED

- Judge id: Qwen3.6-35B-class open-weight checkpoint (owner confirms exact
  id in specs/judge.json). Rationale recorded on standalone merits
  (logit access, scale, cost); no lineage language.
- **Truncation policy (pre-registered, applies whenever assembled prefix
  exceeds budget)**: always retain query, ground_truth (with-GT arm), all
  plan/delegate/final steps verbatim, and the most recent execute steps;
  over budget → reduce oldest execute-step contents to headers
  (agent + type + first 120 chars), never drop a step row. Log truncation
  events per assessment; report truncated fraction per subset.
- Type-classifier model (if the typing audit escalates to LLM
  classification) must be family-disjoint from both judge families.
  Record families in specs/judge.json.

## 4. Decisions from the last review cycle — CONFIRMED, unchanged by severance

- Anomaly policy: **flag** — new flag class distinct from
  `agent_step_mismatch`; counts 126/58 preserved; hard-fail only on count
  mismatch; five anomalous file ids logged in the manifest; dual
  reporting.
- Typing: audit-first — run `typecheck --audit-out`; if confusion is
  confined to plan↔delegate, hierarchical (rules for coarse
  coordination/execute/final split, LLM only for plan-vs-delegate);
  if diffuse, full LLM classifier. ≥90% gate unchanged.
- Six self-contained gaps (per-type thresholds, pooled-normalization arm,
  peer-corroboration toggle, prefix slices, normalized-position table,
  judge-disjointness field): **go — close all now.** "Per-type thresholds"
  and "pooled arm" are implemented inside normalize/ (typed vs global
  normalization = E4's typing on/off arm).

## 5. Experiment manifest — AMENDED

E0 (redefined above) → E1 primary vs 3 baselines (both judges, both
subsets, both GT settings) → E2 readout ablation → E3 rule ablation (incl.
threshold-free variants) → E4 typing on/off (typed vs pooled
normalization) → E5 evidence ablation (+ hindsight ceiling figure) →
E6 judge-family sensitivity → E7 surrogate-intrinsic →
[E8 success-control, gated, unchanged] → E9 uniformity stratification
(from E1 outputs).

## 6. Language rule (applies to all emitted artifacts)

No emitted file — code, comments, specs, reports, figures — may reference
the prior single-agent project, its corpus, its harness, or its results.
Motivating hypotheses (readout operating point, intrinsic
incommensurability) are stated as hypotheses tested in this project's own
ablations, or attributed to published literature only.
