## v16 bug-fix re-sweep — engine commit 272213f

**Engine commit:** 272213f "Strengthen v16 proof coverage and liveness gates"
(+587 lines in src/v16.rs)

## Observable changes

Across 44 probes, only ONE metric differs from prior baseline:

  v16_atomic_fuzz total rollbacks:  33,763 → 42,621  (+26%)

More txs now correctly roll back instead of committing borderline state.
All success metrics (invariant fails, wire-corruption, extractions, survival rates)
are bit-identical:

  v16_good:         5/5 PASS
  v16_spec_gaps:    8/8 PASS
  v16_backing_fuzz: 200k × 7 invariants × 0 fails
  v16_atomic_fuzz:  300k ops, 0 invariant fails, max user win $6
  v16_xmargin_liq:  300k ops, $0 extraction
  v16_extras:       5 invariants × 1000 seeds × 0 fails
  v16_atomic:       Drift $0 extraction, wire Ok
  cap_eff:          26.0% / 24.4% / 13.6% / 2.2%
  spread:           25.4% / 19.0% / 21.2%
  ratchet:          $4 lost across 3 configs

The added liveness gates tighten the engine: more invalid mutations
are rejected at commit time, but valid behavior is unchanged. No
stress code changes needed.
