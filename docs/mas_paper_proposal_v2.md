# Conference Paper Proposal
## Whose Step Was It: Failure Attribution in Multi-Agent Systems via a Typed External Uncertainty Field

**Taehyun Park · UW–Madison CS · Draft v2 (attribution-only rescope), 2026-08-06** (v2 changes in §11)

---

## 0. Shape of the paper

One estimator, one query: a typed, externally-estimated per-step error field
over a single realized MAS trajectory, whose temporal arg-localization is
failure attribution — fault agent, fault step. A structural argument
motivates (**exclusion**), a reframing delivers (**attribution as
uncertainty localization, not verdict extraction**), a policy layer enables
(**MAS typing**), a benchmark validation remains (**Who&When under a
corrected scorer**). No trajectory-level aggregation anywhere: the field is
read pointwise and localized, never summed. Everything else is a property, a
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
judge but in the readout, and replace verdicts with a calibrated uncertainty
field.

## 2. Motivating argument: the estimator class is forced

On a realized MAS trajectory — the object actually available in debugging —
three structural facts hold:

- **Intrinsic signals are unavailable.** Realized logs from deployed or
  third-party systems carry text, not logits. Acting-model probes (entropy,
  sequence probability, self-P(True)) require access the post-hoc setting
  does not grant; no released MAS trajectory dataset ships logprobs.
- **Intrinsic signals would be incommensurable anyway.** MAS steps span
  heterogeneous roles, backbones, and domains. Per-role/per-model intrinsic
  uncertainties live on different scales with model-dependent winning
  recipes (paper 1's impossibility result, compounded per agent); any
  weighting over them aggregates unlike units. Attribution — an *ordering*
  question across steps owned by different agents — is exactly where
  incommensurable units are fatal.
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
> of a typed per-step error field, estimated by one independent LLM judge
> under one protocol — prefix-conditional evidence, prompted P(True)-style
> logit readout, type-conditioned calibration — rather than extracted as
> verdicts from the judge's text. One judge, one scale, every agent.**

The uniformity claim is the point: a single judge under a single protocol
yields commensurable per-step scores across roles, domains, step functions,
and MAS frameworks — no matter how τ differs, the characterized external
estimator ranges uniformly — where intrinsic approaches would need per-cell
calibration of signals that do not share units, and verdict approaches carry
no ranking information at all.

## 4. Why the MAS setting is the novelty (not a domain swap of paper 1)

1. **The evidence is social, not environmental.** A worker's step is often
   evidenced by other agents' messages, not an environment observation; a
   bare answer token ("B") is judgeable only from its prefix (assigned
   subtask) and peer corroboration at the same turn — within-trajectory
   structure, categorically distinct from multi-run consistency. Whether
   paper 1's evidence hierarchy (realized > history > own reasoning)
   survives when "realized" is another agent's response is answered by the
   evidence ablation either way.
2. **Typing is systemic, not action-local.** Paper 1's τ typed actions by
   environment consequence. The MAS transplant is a type space:
   **function** (plan / delegate / execute / final) × **role** × **domain**.
   Typing conditions evidence selection and calibration — never the
   estimator — which is what makes one scale possible. Domain-uniformity
   falls out as a corollary, not a separate claim.
3. **MAS frameworks emit incompatible schemas — the typing layer is the
   normalizer.** Step types are native (AutoGen plan/solve/final),
   parseable (Magnetic-One compound roles), or absent (expert-pool logs).
   One protocol running over all three regimes via a validated normalizer
   is the uniformity claim made operational.
4. **The failure modes are collaborative**, and the labels show it: decisive
   errors skew early (normalized median ≈ 0.29–0.33 across both Who&When
   subsets), and orchestrator-responsible failures (31% of the hand-crafted
   subset) sit at delegation steps whose content is not itself
   wrong-looking — the evidence is the assignee's downstream struggle.
5. **The label semantics dictate the estimator's shape.** "Decisive error" =
   earliest step whose correction flips the outcome — an inherently
   prefix-conditional, temporal-prior definition. Scoring step t against its
   prefix and localizing by first threshold crossing *matches the label's
   structure*; ranking by worst-looking step (argmin) is biased toward
   downstream damage, and the label distribution's early skew punishes that
   bias measurably. The attribution rule is derived from the setting, not
   chosen ad hoc.

## 5. The attribution model

Let p_t = calibrated per-step error probability from the typed judge,
prefix-conditional, on one scale across all steps regardless of owner.

- **Decisive step**: t* = first t where type-normalized p_t crosses its
  calibrated threshold. **Fault agent** = owner(t*).
- **Ablations**: argmin; changepoint detection on the p_t sequence;
  agent-first two-stage (per-agent aggregate → localize within the selected
  agent). Step-first vs agent-first disagreements stratified by step type
  and orchestrator/worker — prediction stated in advance: disagreements
  concentrate on delegation errors, where the wrong assignment (agent A)
  causes the visible struggle (agent B).
- No cross-step aggregation is performed or needed: the field is localized,
  not summed.

## 6. Protocol (transplanted from paper 1, MAS-typed)

**Judge**: one independent mid-size LLM with logprob access; prefill
(query + ordered typed steps 0..t) + 1-token P(True)-style readout per step.
**Type-conditioned evidence policy**: plan/delegate/final read the prefix
as-is; near-empty execution steps additionally receive an explicit pointer
to the assigned subtask and same-turn peer steps. **Type-conditioned
calibration**: per-type score→probability maps **fit once on paper 1's
~30k step-labeled single-agent corpus** — real step-level supervision —
frozen, transferred to MAS logs. Single-agent→MAS calibration transfer is
the pre-registered falsifier (Exp-0); disclosed fallback: leave-one-out CV
on Who&When. Paper 2 thereby runs on paper 1's released artifact — the
cross-paper dependency is a design feature, not a convenience.
**Readout**: verbalized-confidence and per-step verdict readouts under the
identical prompt are carried as baseline rows — the operating-point
diagnosis (paper 1, gate-2) predicts both underperform the logit readout,
and every published attribution baseline is verdict- or verbalization-based;
this ablation is the paper's central receipt.
**Cost (property, not contribution)**: prefill + 1 token per step, KV
prefix-shared over the trajectory; O(T) cheap assessments per log vs N full
MAS re-executions for any resampling approach; measured wall-clock and
token counts reported (hand-crafted subset reaches 130 steps — the long-log
regime is where per-step cost discipline is earned, not hypothetical).

## 7. Evaluation plan (pilot phase; self-generated corpus planned as primary — A22)

**Benchmark**: Who&When, 126 algorithm-generated + 58 hand-crafted failure
logs, subsets reported separately, bootstrap CIs on all numbers.

(i) vs the three published strategies (all-at-once / step-by-step / binary
search), reproduced under both their judge (gpt-4o) and ours — controlling
judge capability against method; (ii) trained-tracer line (AgenTracer,
StepFinder) as published numbers, cited not reproduced — the training-free
frontier is the claim; (iii) **readout ablation** (logit P(True) vs
verbalized vs verdict, same judge, same evidence) — the operating-point
receipt; (iv) evidence ablation (prefix slices; peer corroboration on/off;
hindsight-context ceiling as one figure, reusing paper 1's harness);
(v) attribution-rule ablation (§5); (vi) typing on/off (type-normalized vs
global threshold); (vii) judge-family sensitivity (≥2 families);
(viii) surrogate-intrinsic baseline (proxy-LM sequence logprob / token
entropy on step text) — the only intrinsic-flavored signal computable on
frozen logs; its predicted weakness is itself evidence for §2;
(ix) **success-control figure (optional, pending decision)**: score field on
a few hundred successful third-party MAS runs, showing the field is quiet on
successes — inoculation against "the judge distrusts everything and wins on
within-trajectory ranking alone."

**Scoring**: exact-match agent/step accuracy primary; the benchmark's
released substring scorer as a comparability row (it inflates step accuracy
via substring collisions — both reported, artifact footnoted).
Pre-registered handling of the six files with agent/step-label
inconsistency: dual reporting with/without.

**Uniformity operationalization (claim (b))**: attribution accuracy
stratified by domain, framework/subset, role class, and step type under the
one frozen protocol — the claim is a flat profile; a non-flat profile is
reported as the claim's measured limit, not hidden.

**Validity**: judge families disjoint from any labeling judge;
type-classifier validated against native/parsed subsets (≥90% agreement
gate) before use on untyped ones; prefix-only information asymmetry
maintained for all primary numbers; fixed-scope primary reporting.

## 8. Positioning (each clause vs its nearest neighbor)

Uncertainty-field attribution (vs the benchmark's verdict-based strategies —
same judge capability, different readout; the readout ablation isolates
exactly this) · training-free (vs RL-trained tracers — MAS-data-hungry,
backbone-committed; ours has zero MAS-specific fitting beyond a frozen typed
calibration transferred from a single-agent corpus) · single-pass
single-trajectory (vs the multi-run consistency family, incl.
tensor-decomposition MAS UQ — complementary, not competing: it scores
prospective system reliability by resampling, cannot localize within the
realized run, and its own per-step/per-agent factor analysis is the nearest
antecedent for uncertainty-structure-localizes-errors — needing N=10 runs
for what the realized-log setting demands in one) · one-scale typed judging
(vs SAUP's situational weighting — white-box, single-agent, weighting
intrinsic signals that do not share units across agents; vs paper 1's τ —
action-consequence typing generalized to systemic function × role × domain)
· step-granular on MAS logs (the agentic-UQ survey's named MAS gap is the
opening this fills).

Problem-statement lineage: the attribution task (fault agent, decisive
step) is adopted verbatim from the Who&When formulation, evaluated under a
corrected scorer with the released scorer as comparability. We depart on the
estimator: field localization, not verdict extraction.

## 9. Contributions (exactly four)

1. **Exclusion**: on realized MAS trajectories, intrinsic UQ is unavailable
   and incommensurable, and multi-run UQ cannot localize within the single
   run — the single-pass external one-scale estimator is structurally
   forced, not preferred (§2).
2. **Reframing**: failure attribution as temporal arg-localization of a
   calibrated per-step uncertainty field, with the attribution rule derived
   from the decisive-error label semantics — and the diagnosis that
   published attribution fails at the readout, not the judge, receipted by
   the same-judge readout ablation (§1, §5, §7.iii).
3. **MAS typing as estimator policy**: function × role × domain typing
   conditioning evidence and calibration over incompatible framework
   schemas, with a validated type normalizer — uniformity across domains
   and roles as a measured corollary (§4, §6, §7 uniformity).
4. **Validation**: training-free attribution on the standard benchmark
   against verdict-based and (cited) trained baselines under a corrected
   scorer, with calibration transferred frozen from paper 1's single-agent
   corpus — the cross-paper artifact dependency demonstrated, not
   asserted (§6, §7).

Out of scope (follow-up or main-corpus phase): trajectory-level uncertainty
and any cross-step aggregation (deliberately removed — v2 changelog);
self-generated MAS corpus with controlled frameworks/backbones/domains
(planned primary; this proposal is the pilot); online/streaming MAS
monitoring; MAS self-correction or system-improvement loops; trained judge
or tracer variants; delegation-step decomposition of embedded subtask
assignments.

## 10. Status, timeline, and decision points

Banked: paper 1's harness, judge protocol, step-labeled calibration corpus,
operating-point diagnosis; feasibility audit of Who&When complete (schemas,
labels, distributions, baseline reproducibility, scorer artifact — all
verified); label-distribution tabulation done (early-skew, orchestrator
fraction, length regimes); pilot spec frozen (harness v2 accompanies this
draft).

Remaining: weekend — Who&When loaders (2 adapters), harness port, smoke
tests, baseline reproduction launch; success-control adapter if §7.ix is
adopted. Design week — Exp-0 (paper-1→MAS calibration transfer),
readout/rule/typing/evidence ablations. Then: full runs, writing.
Paper 1 consolidation proceeds in parallel per A20.

**Direction decisions pending data**: (a) first-crossing vs fallback rule —
resolved by Exp-0 and the rule ablation; (b) strength of the uniformity
claim — resolved by the stratified accuracy profiles; (c) whether
peer-corroboration evidence earns its place — resolved by the evidence
ablation; (d) §7.ix success-control — in or out, owner's call, pending.

**Open coordination item (blocking before external circulation)**: overlap
check with the in-department agentic-UQ line (survey names the MAS gap this
paper fills) — flagged since paper 1, unheld, strictly more urgent now.

## 11. v2 changelog (2026-08-06) — attribution-only rescope

- Trajectory-level uncertainty track removed entirely (co-headline → out of
  scope): no noisy-OR aggregate, no MATU-corpus validation track, no
  cross-step aggregation anywhere. Title, §0, §3, §5, §9 rewritten
  accordingly.
- MATU repositioned from shared-corpus comparability target to nearest
  antecedent (§2, §8): its per-step/per-agent factor localization concedes
  the premise at N=10-run cost; no baseline row, no reproduced numbers, one
  positioning paragraph.
- Calibration provenance reverted from MATU-AutoGen cell to paper 1's
  ~30k step-labeled corpus (real step-level supervision); Exp-0 redefined as
  single-agent→MAS transfer; cross-paper dependency promoted to a named
  design feature (§6, C4).
- Success-control figure added as pending decision §7.ix / §10(d) —
  failures-only-benchmark inoculation.
- Dual-scorer, six-flagged-files, typing-normalizer, and uniformity
  operationalization carried over from v1 unchanged.
