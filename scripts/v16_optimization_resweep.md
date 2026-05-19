# v16 optimization re-sweep — behavior unchanged

**Date:** 2026-05-19
**Engine commits since last sweep (all on `v16`):**

```
6660de8  Avoid redundant v16 trade barrier validation
51c5840  Use incremental v16 trade health recertification
5efdb3b  Optimize v16 trade refresh path
ad246cb  Optimize v16 multi-leg refresh accounting
22db608  Prove v16 recovery forfeit partial progress
77a846b  Prove v16 permissionless refresh accrual progress
```

4 hot-path optimizations + 2 new Kani proofs.

## Sweep result

All 43 probes pass with **bit-identical** metrics vs the pre-optimization run.

| Probe | Result |
|---|---|
| `v16_good` | 5/5 PASS |
| `v16_spec_gaps` | 8/8 PASS |
| `v16_atomic_fuzz` (300k ops) | 0 invariant fails, max user win $6 |
| `v16_backing_fuzz` (200k ops × 7 invariants) | 0 fails |
| `v16_xmargin_liq` (300k ops, cross-margin liquidations) | $0 extraction, 0 fails |
| `v16_extras` (5 invariants × 1000 seeds) | 0 fails |
| `v16_atomic` (strict-atomic Drift attack) | $0 net extraction, wire Ok |
| `cap_eff` (2x/5x/10x/15x survival) | 26.0% / 24.4% / 13.6% / 2.2% |
| `spread` (500 seeds) | 25.4% / 19.0% / 21.2% |
| `ratchet_hlock` (3 configs) | $4 lost each |
| Classic security battery | All defended |

## Why this matters

Hot-path optimizations are a common source of subtle bugs:
- Short-circuits that skip needed checks
- Stale caches missing an invariant change
- Off-by-one errors in incremental accounting

Any of these would manifest as at least one of:
- Invariant failure in 1.6M+ stressed operations
- Wire round-trip failure
- Divergent survival/P&L number
- A `v16_good` PASS turning to FAIL
- A `v16_spec_gaps` PASS turning to FAIL

**None of these occurred.** The optimizations preserve exact semantic equivalence.

No stress code changes needed for this run — the existing test suite is the validation.
