# Pilot Baseline Suite — Spec (2026-08-07)
**Scope: Who&When only, everything runnable now. The completed E1 P(True)/W0
scoring pass (commit dac39d53, both GT settings) is the REFERENCE ROW — it is
not re-run; all baselines are evaluated against its numbers under identical
machinery. Phase-P (exploration): all numbers internal until Phase-F freeze.**

## B0. Reference row (exists)
P(True) logit field, W0, both GT arms · full rule set · exact + tolerance
scorers · CIs · AG/HC separate. Nothing to do.

## B1. Sanity rows — run first, minutes
prior_position (predict step at normalized 0.33, agent = owner) ·
majority_agent (most-step-owning agent; step = their median step) ·
uniform_random_step (seeded, 100 draws, report mean) · first_step · last_step.
Direct (agent, step) output, no rules. Every later row must clear these;
read the HC-orchestrator cell of majority_agent against E1 §6's 0.667 first.

## B2. Capability control — their 3 strategies, our judge
all_at_once / step_by_step / binary_search via the existing subprocess
wrapper, pointed at the served Qwen3.6-35B-A3B (OpenAI-compatible vLLM
endpoint). Both GT arms where their scripts support it. Their output contract,
question_ID join, both scorers (this is where substring vs exact diverges —
free-text predictions). Log served-model id as the snapshot receipt.

## B3. Readout variants — same judge, same evidence, same files
Two additional fields via the E2 runner, identical prompt scaffold and W0
evidence assembly: (a) verbalized numeric confidence (0–100 parsed → [0,1]);
(b) binary verdict per step (True/False token → {0,1} field). Both GT arms.
~16k assessments total. Parse-failure rate logged per arm; failures scored
0.5 and flagged, never dropped.

## B4. Semantic coherence — judge-free fields (one new module)
(a) embed_divergence: cosine distance of step t's embedding from the mean
embedding of steps 0..t−1 (prefix centroid). Embedder: any small on-cluster
sentence model; id logged. (b) nli_contradiction: off-the-shelf NLI,
premise = concatenated last 5 prefix steps (truncate to model limit),
hypothesis = step t; contradiction probability = error score. GT-off only
(GT-on optional, skip if awkward). Module ≤ ~250 LOC inside the 4.5k ceiling.

## Evaluation grid (identical for every field row: B0, B3, B4)
field × {changepoint_single, first_crossing, argmin, relative@1.5/2/2.5}
× GT arms × {exact (primary), tolerance |Δ|≤1, |Δ|≤2, substring (comparability)}
× {all, excl_flagged, excl_anomalous} · file-level bootstrap CIs ·
AG/HC separate. Direct rows (B1, B2) skip rules, keep scorers/CIs/slices.

## Free re-analyses bundled into the same report
(i) base-rate audit: B1 rows vs E1 §6 orchestrator/worker cells;
(ii) final-step scatter: alg/final score vs normalized gold position;
(iii) type composition of predicted vs gold steps (HC early-bias probe);
(iv) tolerance curves for the reference row.

## Deliverable
One report: a single master table (rows = B0–B4 × rule where applicable;
columns = AG/HC × agent/step × GT), the four re-analyses, parse-failure and
fallback-rate appendix, run manifest with spec/model hashes.

## Out of scope here
gpt-4o published-regime arm (quota — slot into B2's table when unblocked) ·
proxy-LM intrinsic (struck; follow-up corpus) · any probe/prompt iteration
or new evidence arms (improvement ladder, post-baseline) · E8.

## Order
B1 → B2 → B3 → B4 → report. B2 and B3 may interleave on the served judge;
B4 builds in parallel.
