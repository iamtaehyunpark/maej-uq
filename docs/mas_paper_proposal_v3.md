# Conference Paper Proposal
## Whose Step Was It: Failure Attribution in Multi-Agent Systems via a Typed External Uncertainty Field

**Taehyun Park · UW–Madison CS · Draft v3 (standalone rescope), 2026-08-07** (v3 changes in §11)

---

## 0. Shape of the paper

One estimator, one query: a typed, externally-estimated per-step trust field
over a single realized MAS trajectory, whose temporal arg-localization is
failure attribution — fault agent, fault step. A structural argument
motivates (**exclusion**), a reframing delivers (**attribution as
uncertainty localization, not verdict extraction**), a policy layer enables
(**MAS typing**), a benchmark validation remains (**Who&When under a
corrected scorer**). No cross-step aggregation anywhere: the field is read
pointwise and localized, never summed. Everything else is a property, a
receipt, or discussion.

## 1. Problem

Multi-agent LLM systems fail through interaction — a wrong plan, a bad
delegation, a worker error a verifier rubber-stamps — and debugging them
requires knowing, from the single realized failure log, which agent's which
step was decisive. The field has named this task (automated failure
attribution: fault agent + decisive error step) and shown it is hard: best
published accuracy is 53.5% agent-level and 14.2% step-level, some methods
below random, even under frontier reasoning models. The published strategies
share one design: ask a judge for *verdicts* — per-log, per-half, or
per-step binary judgments. We locate the failure of that design not in the
judge but in the readout, and replace verdicts with a continuous typed
uncertainty field read from the judge's token probabilities.

## 2. Motivating argument: the estimator class is forced

On a realized MAS trajectory — the object actually available in debugging —
three structural facts hold:

- **Intrinsic signals are unavailable.** Realized logs from deployed or
  third-party systems carry text, not logits. Acting-model probes (entropy,
  sequence probability, self-elicited P(True)) require access the post-hoc
  setting does not grant; no released MAS trajectory dataset ships logprobs.
- **Intrinsic signals would be incommensurable anyway.** MAS steps span
  heterogeneous roles, backbones, and domains. Intrinsic uncertainty scales
  and winning recipes are model- and task-dependent (a recurring finding in
  the LLM-UQ literature); per-agent signals therefore do not share units,
  and any weighting over them aggregates unlike quantities. Attribution —
  an *ordering* question across steps owned by different agents — is
  exactly where incommensurable units are fatal.
- **Multi-run methods answer a different question.** Consistency/resampling
  UQ estimates the input's difficulty for the system by regenerating
  trajectories; it cannot localize anything in the single realized run —
  resampling replaces the object under study. On frozen logs it is
  additionally impossible (systems not reproducible), and where possible it
  costs N full MAS re-executions to diagnose one failure. The nearest
  antecedent concedes the premise: tensor-decomposition MAS UQ localizes
  recurring error patterns via per-step, per-agent factor structure — but
  needs N=10 runs to do statistically what the realized-log setting demands
  in one pass. We take that concession seriously in the single-trajectory
  regime.

What survives the exclusion: a single-pass external estimator reading the
realized trajectory on one scale. We characterize it rather than assume it.

## 3. Thesis

> **Failure attribution should be computed as the temporal arg-localization
> of a typed per-step trust field, estimated by one independent LLM judge
> under one protocol — prefix-conditional evidence, prompted P(True)-style
> logit readout, type-normalized scoring — rather than extracted as
> verdicts from the judge's text. One judge, one scale, every agent.**

The uniformity claim is the point: a single judge under a single protocol
yields commensurable per-step scores across roles, domains, step functions,
and MAS frameworks — no matter how the step type differs, the characterized
external estimator ranges uniformly — where intrinsic approaches would need
per-cell calibration of signals that do not share units, and verdict
approaches carry no ranking information at all.

## 4. Why the MAS setting is the novelty (not LLM-as-judge with more agents)

1. **The judged object has no standalone truth value, and the evidence is
   social.** A worker's step is evidenced by other agents' messages, not an
   environment observation; a bare answer token ("B") is judgeable only
   from its prefix (assigned subtask) and peer corroboration at the same
   turn — within-trajectory structure, categorically distinct from
   multi-run consistency. Which evidence classes carry the signal is an
   open empirical question the evidence ablation answers.
2. **Typing is systemic.** Steps differ by **function** (plan / delegate /
   execute / final) × **role** × **domain**, and these types differ in what
   evidence exists for them and how a judge's raw scores distribute over
   them. Typing conditions evidence selection and score normalization —
   never the estimator — which is what makes one scale possible.
   Domain-uniformity falls out as a corollary, not a separate claim.
3. **MAS frameworks emit incompatible schemas — the typing layer is the
   normalizer.** Step types are native (AutoGen plan/solve/final),
   parseable (Magnetic-One compound roles), or absent (expert-pool logs).
   One protocol running over all three regimes via a validated normalizer
   is the uniformity claim made operational.
4. **The failure modes are collaborative**, and the labels show it:
   decisive errors skew early (normalized median ≈ 0.29–0.33 across both
   Who&When subsets), and orchestrator-responsible failures (31% of the
   hand-crafted subset) sit at delegation steps whose content is not
   itself wrong-looking — the evidence is the assignee's downstream
   struggle.
5. **The label semantics dictate the estimator's shape.** "Decisive error"
   = earliest step whose correction flips the outcome — an inherently
   prefix-conditional, temporal-prior definition. Scoring step t against
   its prefix and localizing by first threshold crossing *matches the
   label's structure*; ranking by worst-looking step (argmin) is biased
   toward downstream damage, and the label distribution's early skew
   punishes that bias measurably. The attribution rule is derived from the
   setting, not chosen ad hoc.

## 5. The attribution model

Let s_t = the judge's typed per-step trust score (logit-derived,
prefix-conditional), on one scale across all steps regardless of owner. No
calibrated probabilities are required: attribution consumes ordering and a
crossing criterion, nothing more.

- **Primary rule**: first-crossing on the type-normalized field — per-type
  normalization (z-scoring) removes systematic score-level differences
  between step functions (exploratory plan-talk vs terse execution), so
  the crossing criterion is comparable across types; fault agent =
  owner(t*).
- **Threshold provenance (standalone)**: normalization statistics and the
  crossing threshold are fit by leave-one-out cross-validation within the
  benchmark, disclosed as such; cross-fold threshold stability is itself a
  pre-registered check (E0), and if unstable, the threshold-free rules
  below are promoted to primary — the decision criterion is fixed before
  any attribution numbers are seen.
- **Threshold-free ablations**: argmin; changepoint detection on the s_t
  sequence; within-trajectory relative crossing (first t with s_t beyond
  k·sd of the trajectory's own distribution); agent-first two-stage
  (per-agent aggregate → localize within the selected agent). Step-first
  vs agent-first disagreements stratified by step type and
  orchestrator/worker — prediction stated in advance: disagreements
  concentrate on delegation errors, where the wrong assignment (agent A)
  causes the visible struggle (agent B).

## 6. Protocol

**Judge**: one independent mid-size open-weight LLM with logprob access
(primary: Qwen3.6-35B-class; chosen for logit availability, cost, and
judge-capable scale — not tied to any prior artifact); second judge family
carried for sensitivity. **Scoring**: prefill (query + ordered typed steps
0..t) + 1-token P(True)-style readout per step; with-ground-truth and
without-ground-truth settings both run, matching the benchmark's two
regimes. **Type-conditioned evidence policy**: plan/delegate/final read the
prefix as-is; near-empty execution steps additionally receive an explicit
pointer to the assigned subtask and same-turn peer steps
(within-trajectory corroboration). **Type-normalized scoring**: per-type
score normalization fit via LOO-CV (§5). **Long-log handling**: type-aware
prefix truncation, pre-registered (structural steps kept verbatim, oldest
execution contents reduced to headers, step rows never dropped; truncated
fraction reported). **Readout baselines**: verbalized numeric confidence
and per-step binary verdict under the IDENTICAL prompt scaffold — the
hypothesis, tested here on its own receipts, is that published attribution
fails at the readout, not the judge; every published baseline is verdict-
or verbalization-based, and the same-judge readout ablation isolates
exactly this. **Cost (property, not contribution)**: prefill + 1 token per
step, KV prefix-shared over the trajectory; O(T) cheap assessments per log
vs N full MAS re-executions for any resampling approach; measured
wall-clock and token counts reported (hand-crafted logs reach 130 steps —
the long-log regime is where per-step cost discipline is earned).

## 7. Evaluation plan

**Benchmark**: Who&When, 126 algorithm-generated + 58 hand-crafted failure
logs, subsets reported separately, bootstrap CIs on all numbers; anomalous
records flagged and dual-reported (with/without), pre-registered counts
(126/58) preserved.

(i) vs the three published strategies (all-at-once / step-by-step / binary
search), reproduced under both their judge (gpt-4o) and ours — controlling
judge capability against method; (ii) trained-tracer line (AgenTracer,
StepFinder) as published numbers, cited not reproduced — the training-free
frontier is the claim; (iii) **readout ablation** (logit P(True) vs
verbalized vs verdict, same judge, same evidence) — the paper's central
receipt; (iv) evidence ablation (prefix slices; peer corroboration on/off;
hindsight-context ceiling as one figure); (v) attribution-rule ablation
(§5) including threshold-free variants; (vi) typing on/off
(type-normalized vs global scoring); (vii) judge-family sensitivity
(≥2 families; type-classifier family disjoint from the judge family);
(viii) surrogate-intrinsic baseline (proxy-LM sequence logprob / token
entropy on step text) — the only intrinsic-flavored signal computable on
frozen logs; its predicted weakness is itself evidence for §2;
(ix) **success-control figure (optional, pending decision)**: the score
field on a few hundred successful third-party MAS runs, showing the field
is quiet on successes — inoculation against "the judge distrusts
everything and wins on within-trajectory ranking alone."

**E0 (run first, pre-registered)**: score-field sanity + LOO threshold
stability — cross-fold variance of per-type normalization statistics and
crossing threshold; the primary-rule decision criterion (§5) executes on
this outcome before any attribution numbers are seen.

**Scoring**: exact-match agent/step accuracy primary; the benchmark's
released substring scorer as a comparability row (it inflates step
accuracy via substring collisions — both reported, artifact footnoted).
Pre-registered handling of the six files with agent/step-label
inconsistency: dual reporting with/without.

**Uniformity operationalization**: attribution accuracy stratified by
domain, framework/subset, role class, and step type under the one
protocol — the claim is a flat profile; a non-flat profile is reported as
the claim's measured limit, not hidden.

**Validity**: judge families disjoint from any labeling or type-classifier
model; type normalizer validated against native/parsed subsets (≥90%
agreement gate) before use on untyped ones; prefix-only information
asymmetry maintained for all primary numbers; fixed-scope primary
reporting.

## 8. Positioning (each clause vs its nearest neighbor)

Uncertainty-field attribution (vs the benchmark's verdict-based
strategies — same judge capability, different readout; the readout
ablation isolates exactly this) · training-free (vs RL-trained tracers —
MAS-data-hungry, backbone-committed; ours has zero MAS-specific training,
with only disclosed within-benchmark CV for normalization statistics) ·
single-pass single-trajectory (vs the multi-run consistency family, incl.
tensor-decomposition MAS UQ — complementary, not competing: it scores
prospective system reliability by resampling, cannot localize within the
realized run, and its own per-step/per-agent factor analysis is the
nearest antecedent for uncertainty-structure-localizes-errors — needing
N=10 runs for what the realized-log setting demands in one) · one-scale
typed judging (vs situational weighting of white-box stepwise uncertainty
— single-agent, and weighting intrinsic signals that do not share units
across agents) · step-granular on MAS logs (the agentic-UQ literature's
named MAS gap is the opening this fills).

Problem-statement lineage: the attribution task (fault agent, decisive
step) is adopted verbatim from the Who&When formulation, evaluated under a
corrected scorer with the released scorer as comparability. We depart on
the estimator: field localization, not verdict extraction.

## 9. Contributions (exactly four)

1. **Exclusion**: on realized MAS trajectories, intrinsic UQ is
   unavailable and incommensurable, and multi-run UQ cannot localize
   within the single run — the single-pass external one-scale estimator
   is structurally forced, not preferred (§2).
2. **Reframing**: failure attribution as temporal arg-localization of a
   typed per-step trust field, with the attribution rule derived from the
   decisive-error label semantics — and the diagnosis that published
   attribution fails at the readout, not the judge, receipted by the
   same-judge readout ablation (§1, §5, §7.iii).
3. **MAS typing as estimator policy**: function × role × domain typing
   conditioning evidence and score normalization over incompatible
   framework schemas, with a validated type normalizer — uniformity across
   domains and roles as a measured corollary (§4, §6, §7 uniformity).
4. **Validation**: training-free attribution on the standard benchmark
   against verdict-based and (cited) trained baselines under a corrected
   scorer, with all normalization provenance disclosed and pre-registered
   (§6, §7).

Out of scope (follow-up work): trajectory-level uncertainty and any
cross-step aggregation; self-generated MAS corpus with controlled
frameworks/backbones/domains (planned as the eventual primary corpus; this
paper is benchmark-scoped); online/streaming MAS monitoring; MAS
self-correction or system-improvement loops; trained judge or tracer
variants; delegation-step decomposition of embedded subtask assignments.

## 10. Status, timeline, and decision points

Banked: feasibility audit of Who&When complete (schemas, labels,
distributions, baseline reproducibility, scorer artifact — verified);
label-distribution tabulation done (early-skew, orchestrator fraction,
length regimes); harness spec frozen (v2 + v2.1 severance amendment);
loader/typing/baseline work in progress with anomaly policy (flag) and
typing escalation (audit-first, hierarchical-if-confined) decided.

Remaining: typing classifier per audit outcome; judge harness; E0;
baseline reproduction; ablation battery; writing.

**Direction decisions pending data**: (a) first-crossing vs threshold-free
primary — resolved by E0's stability check under the pre-fixed criterion;
(b) strength of the uniformity claim — resolved by the stratified
accuracy profiles; (c) whether peer-corroboration evidence earns its
place — resolved by the evidence ablation; (d) §7.ix success-control — in
or out, owner's call, pending.

**Open coordination item (blocking before external circulation)**: overlap
check with the in-department agentic-UQ line (the literature's named MAS
gap is the one this paper fills) — flagged previously, unheld, strictly
more urgent now.

## 11. v3 changelog (2026-08-07) — standalone rescope (A23)

- All cross-references to the prior single-agent UQ project removed:
  calibration no longer transfers from any external corpus; harness
  described generically; motivating claims restated as standalone
  hypotheses tested within this paper or attributed to published
  literature.
- Field demoted from calibrated probabilities p_t to raw typed trust
  scores s_t; calibration replaced by per-type normalization fit via
  disclosed LOO-CV within the benchmark; E0 redefined from transfer test
  to threshold-stability + score-field sanity, with a pre-fixed
  primary-rule decision criterion.
- Former contribution C4's cross-paper dependency claim removed; C4
  restated as pre-registered, provenance-disclosed validation.
- Judge selection restated on standalone merits.
- Anomaly policy (flag + dual-report) and typing escalation decisions
  folded into §7/§10.
