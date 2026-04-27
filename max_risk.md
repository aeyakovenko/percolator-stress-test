# bounty_sol_20x_max — envelope-ceiling bug bounty market

Inverted SOL/USD perpetual, **20x max leverage**, atomic same-tx
withdrawals, $1,000 insurance bounty. Pushed to the absolute §1.4
envelope ceiling — the most aggressive deployable parameters where
the engine's safety proof still validates at init.

> **This is NOT the recommended deploy config.** See `config.md`
> (`bounty_sol_20x`) for the safer setting. This file documents what
> happens when you trade off engine slack for slightly cheaper fees
> and 0.2% more per-accrual oracle tolerance — empirically that
> tradeoff costs:
> - $21 insurance dip per 2000 simulated stress runs (was $0)
> - 1.4% worst-case haircut on user PnL (was 0.0001%)
> - 400× more rounding-overshoot events
>
> Use this config only if you specifically WANT to expose more
> attack surface for the bug bounty.

---

## 1. Engine `RiskParams` (passed to `init_in_place`)

```rust
RiskParams {
    // ── Margin and leverage ─────────────────────────────────
    // Same as bounty_sol_20x: mm = im = 500 (20x leverage, no opening buffer).
    // The §1.4 envelope budget binds against mm; we use the full 500-bps
    // budget here (with slope_gap=4 reserved for engine's exact-N proof).
    maintenance_margin_bps:         500,    // 5.00% — = im
    initial_margin_bps:             500,    // 5.00% initial = 20x max leverage

    // ── Trading and liquidation fees ────────────────────────
    // Pushed to the lowest practical values. Trading at 1 bp undercuts
    // every CEX/DEX globally (HL top tier is 2.4 bps). Liquidation fee
    // at 5 bps frees envelope for max_price_move, at the cost of keeper
    // economics — keepers earn 5 bps × notional which may be marginal
    // for small liquidations.
    trading_fee_bps:                  1,    // 0.01% — most aggressive globally
    liquidation_fee_bps:              5,    // 0.05% — frees envelope for max_move

    // ── Account capacity ────────────────────────────────────
    max_accounts:                 4_096,
    max_active_positions_per_side: 4_096,

    // ── Margin floors (envelope slack) ──────────────────────
    min_nonzero_mm_req:             500,
    min_nonzero_im_req:             600,
    min_liquidation_abs:    U128::ZERO,
    liquidation_fee_cap: U128::new(50_000_000_000),

    // ── Warmup horizon bounds ───────────────────────────────
    h_min:                            0,    // allow admit_h_min=0
    h_max:                       86_400,    // ~9.6h ceiling

    // ── Resolved mode ───────────────────────────────────────
    resolve_price_deviation_bps:  1_000,

    // ── Solvency envelope (v12.19) — ABSOLUTE CEILING ───────
    // §1.4 envelope: max_move*max_dt + funding_budget + liq_fee < mm
    //
    // With mm=500, liq=5, funding_budget_bps=1, max_dt=10, max_price_move=49:
    //   accrual_window_move = 49 * 10 = 490 bps  (= 4.9% per accrual)
    //   linear_budget       = 490 + 1 + 5 = 496 < 500  ✓
    //   slope_gap           = 4 bps  (the engine's exact-N rounding reservation)
    //
    // 49 bps/slot is the highest max_price_move the engine accepts at any
    // (mm, liq, max_dt) combination — pushing further would either fail
    // the strict-less-than check (linear ≥ mm) or the exact-N proof at
    // floor_region_max=10019.
    //
    // CRITICAL: the engine RESERVES slope_gap for its rounding proof.
    // Operating at 49 bps/slot uses the full reservation in practice,
    // producing measurable haircut and occasional insurance dip — see
    // §10 below for the empirical cost.
    max_accrual_dt_slots:            10,
    max_abs_funding_e9_per_slot: 10_000,
    min_funding_lifetime_slots:      10,
    max_price_move_bps_per_slot:     49,    // 0.49%/slot ≈ 1.23%/sec
                                            // = 99.2% of envelope ceiling
}
```

---

## 2. Wrapper policy (per-call args)

Identical to `bounty_sol_20x`:

```rust
admit_h_min  =                                  0   // INSTANT FAST PATH
admit_h_max  =                             86_400   // 9.6h slow path on stress
admit_h_max_consumption_threshold_bps_opt =
    Some(initial_margin_bps as u128) =       Some(500)   // 1/leverage = 5%

max_revalidations =                            64
rr_window_size    =                           192
```

---

## 3. Wrapper-side oracle clamp

Updated for the higher max_price_move:

```rust
fn wrapper_oracle(real: u64, last_engine_price: u64, dt: u64) -> u64 {
    let max_dp = (last_engine_price as u128
        * 49u128                        // max_price_move_bps_per_slot
        * dt as u128) / 10_000;
    let lower = last_engine_price.saturating_sub(max_dp as u64).max(1);
    let upper = (last_engine_price.saturating_add(max_dp as u64))
        .min(1_000_000_000_000);
    real.clamp(lower, upper)
}
```

---

## 4. Anti-spam economics (wrapper-enforced)

Identical to `bounty_sol_20x`:

| Knob | Value |
|---|---|
| Minimum initial deposit | $10 USDC |
| Minimum order notional | $10 USDC |
| Account creation fee | $0.50 |
| Recurring maintenance fee | rate = 1 atomic/slot ≈ $0.216/day |
| Empty-account reclaim incentive | $0.10 |

(See `config.md` §4 for justification.)

---

## 5. Why this is "max risk"

The §1.4 envelope `max_move × dt + funding + liq < mm` reserves
`slope_gap = mm - linear_budget` bps for the engine's exact-N
solvency proof. The proof guarantees that for every position size
N from 1 to MAX_ACCOUNT_NOTIONAL, the worst-case loss + liquidation
fee fits within `mm_req(N)` — but the proof relies on those
slope_gap bps to absorb integer ceil/floor rounding.

In `bounty_sol_20x`:
- `linear_budget = 47×10 + 1 + 25 = 496`, `slope_gap = 4`
- Rounding artifacts are sub-bps (1-2 atomic units), absorbed invisibly

In `bounty_sol_20x_max`:
- `linear_budget = 49×10 + 1 + 5  = 496`, `slope_gap = 4`
- Same slope_gap mathematically, BUT...
- The bigger `max_price_move × dt` (490 vs 470 bps) means each
  accrual processes 4.3% more state delta, which AMPLIFIES rounding
  artifacts when matured > residual fires
- The smaller liq_fee (5 vs 25 bps) means liquidations have less
  buffer to absorb the rounded difference

Empirically, this difference manifests as:
- 400× more `matured > residual` events at runtime
- Real haircut on user PnL (1.4% in worst-1% of runs)
- Insurance dip in 0.10% of runs ($21 cumulative across 2000 runs)

The engine never violates a top-level invariant — V≥C+I, K bounds,
F bounds, A floor, capital sum, reserve cap all hold. The runtime
overshoots are exactly what the haircut-then-insurance defense
ordering is designed for. But they ARE visible, unlike the safer
config.

---

## 6. Initial capital

```text
Insurance fund:    $1,000     ← BOUNTY TARGET
LP capital:      $100,000     ← Market-maker float
Initial oracle price: $200
```

---

## 7. Comparison

| | **bounty_sol_20x_max** | bounty_sol_20x | Hyperliquid |
|---|---|---|---|
| Max leverage | 20x | 20x | 40x (SOL) |
| Trading fee | **1 bps** | 2 bps | 1.5/4.5 maker/taker |
| Liquidation fee | **5 bps** | 25 bps | 0 explicit (~125 effective) |
| max_price_move/slot | **49 bps** (99.2% of ceiling) | 47 bps (95% of ceiling) | n/a (off-chain) |
| Per-accrual oracle tolerance | **4.90%** | 4.70% | n/a |
| Min deposit | $10 | $10 | $10 practical |
| Min order notional | $10 | $10 | $10 |
| Withdrawal | atomic same-tx | atomic same-tx | $1 + seconds |

This config undercuts everyone on every trader-facing dimension
except liquidation fee (where 5 bps is still 25× cheaper than HL's
effective 125 bps cost from maintenance margin forfeit).

---

## 8. Bounty mechanics

Same as `config.md` §8, but the bounty pool sees more action by
construction:

**Win condition:** cause `engine.insurance_fund.balance` to decrease
below its current value.

**Reward:** `max($1,000, 5 × $drained)`.

In simulated 2000-run stress, this config produces 2 runs (0.10%)
where insurance dips by economically negligible amounts ($1-$12
each). A real bounty hunter is more directed than random sampling —
they would target:

| Class | Spec section | What success looks like |
|---|---|---|
| **Crank-starvation bankruptcy** (most likely path) | §5.5 | Hold position to max_accrual_dt boundary, force liquidation at deepest clamped price, capture insurance dip |
| Sticky-set overflow | §4.3 step 4 | Engineer 256+ admit_h_max accounts in one instruction |
| Generation-rollover timing | §4.3 step 2 | Admit fresh PnL exactly when sweep_cursor wraps and consumption resets |
| ADL exact-K overflow | §10.22 | Drift K toward i128 boundaries via repeated small ADL |
| Same-slot exposed crank | §10.24 | dt=0 crank advancing slot_last with non-P_last price |
| Risk-notional ceil bypass | §10.8 | Open dust position whose floor notional evades MM check |
| Conservation violation | §0 #1 | Make V < C_tot + I — Tier 4 reward |
| Fee-credits sign flip | §1.1 #2 | Roll past i128::MIN, get free capital — Tier 5 reward |

---

## 9. When to use this vs `bounty_sol_20x`

**Use `bounty_sol_20x_max` when:**
- You explicitly want to maximize the bug bounty's attack surface
- The $1,000 prize is meant to actively attract scrutiny, not just
  satisfy a check-the-box requirement
- You're comfortable with sub-1.5% worst-case haircut on real users
  during stress
- You can afford small ($10s/2000 runs) routine insurance dip as
  the cost of advertising "even at the envelope ceiling, the engine
  doesn't break"

**Use `bounty_sol_20x` when:**
- You want the cleanest possible runtime metrics (no haircut, no
  insurance dip)
- The bounty exists primarily as a security-process item rather
  than an active recruiting tool
- You'd rather have engine slack absorb rounding silently

The difference for users:
- Trading fees: `1 bps` vs `2 bps` (50% cheaper here)
- Liquidation fees: `5 bps` vs `25 bps` (5× cheaper here)
- Worst-1% haircut: `1.4%` here vs `0.0001%` in the safer config

For 99% of users (those not in the worst-1% during stress events),
the two configs are indistinguishable. For the remaining 1%, this
config charges them a real (if small) cost.

---

## 10. Fuzz results — 2000 seeds × 1500 slots

```
Engine effective params (all matching this file):
  mm = 500, im = 500 (20x leverage, no opening buffer)
  trading_fee = 1 bps, liquidation_fee = 5 bps
  max_price_move = 49 bps/slot — §1.4 envelope ceiling
  threshold = Some(500) — spec-compliant for public wrapper

8-Invariant battery (4.8M assertions, all PASSED):
  V≥C+I:                       0 violations
  pnl_matured ≤ pnl_pos_tot:   0 violations
  K bounds (≤ i128::MAX/2):    0 violations
  F bounds:                    0 violations
  A floor (≥ MIN_A_SIDE):      0 violations
  neg_pnl_count consistency:   0 violations
  sum(capital) == c_tot:       0 violations
  sum(reserved_pnl) ≤ pos_pnl: 0 violations
  h_zero events:               0
  deficit events:              0

Insurance outflow audit:
  insurance_payout_runs:       2 / 2000 (0.10%)
  insurance_paid_out_total:    $21.20
  insurance_paid_out_max_per_run: $12.22
  insurance_end_p10:           $64,512
  insurance_end_mean:          $68,907

Cascade behavior:
  liquidations:                mean=581 / 2000 users (15.4%)
  drain_only_frac:             59%
  matured_overshoot total:     382,560 events
  affected runs:               1,690 / 2000 (84.5%)
  worst single run events:     266
  min_h_p01:                   0.985589  (1.4% haircut on worst-1%)
  min_h_p05:                   0.988652

Stress lanes:
  consumption-threshold trip:  100% of runs
  peak consumption p99:        4217 bps  (8.4× the 500 bps trigger)
  sweep_generations:           18 per run
```

### Comparison vs `bounty_sol_20x` (same fuzz, 2000 seeds)

| Metric | bounty_sol_20x | bounty_sol_20x_max | Δ |
|---|---|---|---|
| All 8 solvency invariants | ✅ | ✅ | — |
| h_zero events | 0 | 0 | — |
| deficit events | 0 | 0 | — |
| **insurance_paid_out_total** | $0 | **$21.20** | first non-zero |
| insurance payout runs | 0/2000 | 2/2000 | — |
| **min_h_p01** | 0.999999 | **0.985589** | -1.4% |
| **matured_overshoot events** | 905 | **382,560** | 400× |
| affected runs | 18 (0.9%) | 1,690 (84.5%) | 90× |
| drain_only_frac | 79% | 59% | -20pp |

The 0.2% extra per-accrual oracle tolerance (4.7% → 4.9%) costs:
1.4% worst-case haircut, $21 of routine insurance dip per 2000
runs, and 400× more visible rounding events. The engine still
holds — but it does so by the haircut + insurance machinery
working hard, not by sitting comfortably below the envelope.

---

## 11. Recommendation for the bounty

If the goal is "maximize hunter attention by maximizing visible
attack surface," deploy this config. The $21 of routine insurance
dip per 2000 simulated stress runs becomes a public signal: "the
engine sits close enough to its safety proof that small economic
events are observable." Hunters interpret this as "there's signal
worth poking at."

If the goal is "deploy with maximally clean metrics so any insurance
dip indicates a real bug," deploy `config.md`'s `bounty_sol_20x`
instead. There a $0.000001 dip is meaningful evidence; here a $0.50
dip is noise and you'd need a $50+ dip to be sure.

For a public bounty designed to attract actual security work, **this
config is the more honest invitation**: "we know there's runtime
slack at the envelope ceiling — find what's hiding in that slack."
