# Registered attribution-rule directive

The primary attribution rule is fixed by this directive. It is not chosen by
any experiment's outcome. This file is hashed into every run manifest as
`rule_provenance`, so any reported number can be traced to the directive that
fixed the rule it used.

---

Primary rule → changepoint_single (two-regime mean-shift split on per-step
scores; decisive = first step of regime 2; agent = owner). Implement contrast
statistic + boundary/noise fallback to argmin, fallback condition in the
registered criteria file.

Demote: relative_crossing, LOO first-crossing, argmin → E3 ablation rows.
RELATIVE_K: no registration; k ∈ {1.5, 2, 2.5} sensitivity inside E3 only.

E0 → sanity-only (score-field degeneracy, per-type distributions, stability
reporting for the LOO ablation arm); e0_decision.json gating of E1 removed —
primary is fixed by this directive, not by E0's outcome. Log this directive's
hash as the rule provenance.

Typed normalization: retain as the E4 arm (typed vs pooled) — changepoint
benefits from type-normalized scores for the same level-shift reasons, but no
longer depends on any threshold from it.

Everything else from the last cycle stands: docs v3 consolidation, ≤4k LOC,
judge.json awaiting checkpoint id.
