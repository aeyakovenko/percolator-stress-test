# v16 stress sweep — full results

**Date:** 2026-05-18
**Engine:** `v16.8.0 Realizable Full Shared Cross-Margin` (engine commit `a37eb5f`)
**Stress branch:** `v16`
**Probes run:** 38 / 38 pass execution; all invariants hold across the suite.

## What the engine delivers (verified)

**HL-like equity-level cross-margin** is wired up end-to-end:

1. `apply_haircut_bounded_close_loss_to_pnl` and `apply_signed_kf_delta_to_pnl` consult source-credit with residual fallback (`MAX(global, source)`).
2. `settle_leg_kf_effects` passes `source_domain = opposite_side(leg.side)` for positive deltas and calls `reserve_new_capital_backed_loss_for_source_domain_not_atomic` on negative deltas. This is the auto-orchestrated `BackingReservationPlan` from spec §2.2.
3. Per-domain isolation: 32 buckets (16 assets × 2 sides) in a flat `source_backing_buckets` array, indexed by `insurance_domain_index(asset, side)` — cross-domain bleed structurally impossible.
4. `convert_released_pnl_to_capital_not_atomic` + `withdraw_not_atomic` correctly realizes spread profit as withdrawable USDC.

## End-to-end proof: spread profit realized as real cash

`spread_realize` probe (long SOL + short ETH, favorable move, close + realize + withdraw):

```
State after favorable spread move:
  cap=$1000, pnl=$1000, certified_equity=$2000
  residual=$1000, junior_bound=$1000 → haircut=100%
After closing both legs: cap=$1000, pnl=$1000
convert_released_pnl_to_capital: Ok($1000)
After convert: cap=$2000, pnl=$0
Withdraw $2000 → OK, final cap=$0
User's realized USDC: $2000 (started with $1000)
```

**$1000 deposit → $2000 withdrawal** on a profitable spread.

## Capital efficiency surveys (500 seeds × 30-day random walks, LP refreshed per tick)

### Single-asset survival

| Lev | Survival | Avg P&L (survivors) |
|---|---|---|
| 2x | 26.0% | +27.80% |
| 5x | 24.4% | +77.75% |
| 10x | 13.6% | +178.91% |
| 15x | 2.2% | +123.90% |

### Diversification ($10k notional, $5k cap)

| Config | Survival | Avg P&L |
|---|---|---|
| 1 asset | 26.8% | +22.06% |
| 4 assets | 26.2% | +8.90% |

### Spread trades ($5k each leg, $1k cap)

| Config | Survival | Avg P&L |
|---|---|---|
| Long SOL only (5x) | 25.4% | +59.94% |
| Long SOL + long ETH | 19.0% | +98.81% |
| Long SOL + short ETH (spread) | 21.2% | +109.63% |

### Round-trip ratchet (cap_eff)

| Move | Result |
|---|---|
| ±2% | $0 lost |
| ±5% | $0 lost |
| ±10% | $0 lost |

### h-lock variants

| Config | Total lost |
|---|---|
| instant (h_min=0, h_max=1) | $4 (0.5%) |
| default (h_min=0, h_max=30) | $4 (0.5%) |
| warmup (h_min=5, h_max=30) | $4 (0.5%) |

## Margin model (capital efficiency)

`v16_margin_snap` shows IM/MM is strictly additive per leg. Same as HL:

| Notional/leg | Config | IM | MM |
|---|---|---|---|
| $5k | single SOL | $250 | $250 |
| $5k | long SOL + long ETH (naked 2 legs) | $500 | $500 |
| $5k | long SOL + short ETH (hedge) | $500 | $500 |

No portfolio-risk netting — the hedge and naked positions have identical margin. What v16 delivers is **equity-level cross-margin** (gains on one leg back losses on another, gains are realizable as cash), not portfolio margin.

## Security & invariant battery

### Source-credit backing invariants (`v16_backing_fuzz`, 200,000 transitions)

All 7 invariants hold across 2000 seeds × 100 ops × 5 users × 3 assets:

| Invariant | Fails |
|---|---|
| Per-domain `fresh_reserved >= valid_liened` | 0 |
| `sum(account.source_claim_bound) == source_credit.positive_claim_bound` | 0 |
| `sum(account.source_lien_effective_reserved × BOUND_SCALE) == valid_liened_backing` | 0 |
| `vault >= c_tot + insurance` | 0 |
| `insurance_spent + reserved <= insurance_budget` per domain | 0 |
| No user withdrew more than they deposited | 0 |
| Engine's `assert_public_invariants` | 0 |

### Extraction attack (`v16_extract`, 500 seeds × random adversarial sequences)

```
seeds that withdrew > deposit:        0
max single-seed net extraction:       $0
engine invariant failures:            0
```

### Drift-style oracle attack (`v16_drift`)

Attacker opens 5x long on asset 0, pumps oracle $200 → $400 at engine's max 45 bps/slot, uses inflated PnL as cross-margin to open $50k BTC position, then tries to extract:

```
Deposited:  $1000
Withdrawn:  $0       ← engine blocked convert + withdraw (LockActive)
Left in:    $994
NET:        $-5
LP final cap: $49,994,994 (intact)
Engine invariants: Ok
```

The attacker successfully expanded notional via soft cross-margin credit, but **could not convert the fake gain to withdrawable USDC**. The defenses (bounded `max_price_move_bps_per_slot`, soft credit not directly withdrawable, hard lien required for realization) all hold.

### Extra attacks fuzz (`v16_extras`, 1000 seeds × ~200 ops × 7 accounts × 3 assets)

Five v16-specific corner cases:

| Attack | Fails |
|---|---|
| A) Bound-understate (`positive_claim_bound_num >= exact_positive_claim_num`, spec §0.17) | 0 |
| B) Multi-user same-domain race (per-account sums == global aggregate, spec §0.10) | 0 |
| C) Withdraw-while-encumbered (post-withdraw must be ≥ IM, spec §0.9) | 0 |
| D) Self-trade extracts no value (spec §0.3 says engine MUST NOT check identity, but no extraction allowed) | 0 |
| E) Stale-cert favorable action rejected after epoch bump | 0 |

### Per-domain bucket isolation (`v16_buckets`)

Three users each lose on a different (asset, side) pair → backing appears only in the corresponding 3 bucket slots out of 32. Other 29 buckets stay `Empty`. Each domain's `credit_rate` is computed independently from its own bucket. Storage layout (`source_backing_buckets: [BackingBucketV16; V16_DOMAIN_COUNT]`) makes cross-domain bleed impossible.

### Classic security probes (all defended)

| Probe | Status |
|---|---|
| exec_price_attack | LockActive at 9999 bps deviation |
| sybil_close | $0 attacker extraction across all deviations |
| hard / hard_ext (2000-seed extraction fuzz) | total withdrawn = sum of deposits, no excess; 0 invariant fails |
| drift | bad-asset isolation holds; healthy asset preserves cap=$499 |
| domain_attr | bankruptcy charged only to source domain; budget respected |
| corner_cases (adversarial keeper liquidates richest first) | no extraction across 20 liquidations |
| multileg (4-leg user crash) | insurance_used=$0, residual=$0, explicit_loss=$0 |
| f6 | conservative-pause wrapper-controlled correctly |
| boundary, config, resolve, pnl_trace, advanced | all paths pass |

## Bottom line

The v16 source-credit + auto-`BackingReservationPlan` is operating correctly under stress:

- **Healthy-market HL-like cross-margin** is delivered (spread profit realizes as cash).
- **Per-domain isolation** is structural (32 separate buckets, fixed-array indexing).
- **400k+ invariant checks** across 200k+ random state transitions: 0 fails.
- **Targeted extraction attempts** (random, Drift-style, multi-user race, encumbered withdraw, self-trade, stale cert): all $0 net extraction.
- **Backing reserve accounting** stays consistent: per-account sums match global aggregates, no double-spend, vault always ≥ c_tot + insurance, no insurance budget breach.

The lien lifecycle (create → consume / release / impair) and the auto-`BackingReservationPlan` together close the gap from "paper PnL stuck IOUs" (v14/v15 healthy-market behavior) to "realizable cash flow" (v16) while keeping the spec's safety boundaries (per-domain isolation, hard liens for withdrawal, bounded oracle moves) intact.
