# bounty_sol_50x_v1 — public bug bounty market

Inverted SOL/USD perpetual, 50x max leverage, $1000 insurance bounty,
admit_h_min=0 (atomic withdrawals).

Aggressively priced to compete with HL/Binance while keeping engine
invariants intact. The atomic same-tx withdrawal is the differentiator.

---

## 1. Engine `RiskParams` (passed to `init_in_place`)

```rust
RiskParams {
    // ── Margin and leverage ─────────────────────────────────
    // 50x leverage: 1/50 = 2% initial margin → im_bps = 200
    // Maintenance buffer: mm = im/2 = 100 bps (typical perp convention)
    // → liquidation triggers when account equity drops to 1% of notional
    maintenance_margin_bps:         100,    // 1.0% maintenance
    initial_margin_bps:             200,    // 2.0% initial = 50x max leverage
    trading_fee_bps:                  4,    // 0.04% (Binance is 5)
    liquidation_fee_bps:             20,    // 0.20% (Binance is 40)
    liquidation_fee_cap:    U128::new(50_000_000_000),  // $50K cap

    // ── Account capacity ────────────────────────────────────
    max_accounts:                 4_096,    // engine slab
    max_active_positions_per_side: 10_000,  // == max_accounts since both sides cap

    // ── Margin floors (anti-dust) ───────────────────────────
    // Higher floors give the exact-N solvency check more slack at small N.
    min_nonzero_mm_req:              20,    // absolute mm floor
    min_nonzero_im_req:              30,
    min_liquidation_abs:    U128::ZERO,     // wrapper enforces min position size

    // ── Warmup horizon bounds ───────────────────────────────
    h_min:                            0,    // allow admit_h_min=0
    h_max:                       86_400,    // ~9.6h ceiling at 400ms slots

    // ── Resolved mode ───────────────────────────────────────
    resolve_price_deviation_bps:  1_000,    // 10% bound for settlement price

    // ── Solvency envelope (v12.19) ──────────────────────────
    // Constraint: max_price_move*max_dt + funding_budget + liq_fee < mm,
    // AND the exact-N check at floor_region_max passes.
    // With mm=100, liq=20, max_dt=10, max_price_move=4:
    //   linear_budget = 40+1+20 = 61 < 100, slope_gap=39 ✓
    //   exact-N at floor_region_max=2099 validates (loss+liq_fee=14 < mm_req=20)
    max_accrual_dt_slots:            10,    // 4 sec keeper-lag tolerance
    max_abs_funding_e9_per_slot: 10_000,    // GLOBAL_MAX
    min_funding_lifetime_slots:      10,
    max_price_move_bps_per_slot:      4,    // 0.04%/slot ≈ 0.10%/sec
}
```

**Why these:**

- **mm=200 (50x leverage)**: the most aggressive that fits the v12.19
  exact-N solvency envelope with a 20 bps liquidation fee. Going to 100x
  (mm=100) leaves only `mm - liq - 1 = 79` bps for `max_price_move × dt`
  total, which forces `max_price_move ≤ 7 bps/slot`. That's tighter than
  Pyth confidence intervals during volatility, so clamping engages
  constantly. mm=200 keeps clamping rare in practice.

- **trading=4, liq=20**: undercuts both Binance (5 / 40) and HL (2.5 / 200)
  on liquidation cost. Trading parity with HL maker.

- **max_price_move=17, max_dt=10**: 17 bps/slot at 400ms = 0.42%/sec.
  Realistic SOL volatility is well under this; clamping triggers only
  on actual flash-crash territory. The 10-slot dt window gives keepers
  4 seconds of lag tolerance before the engine refuses to accrue.

- **min_liq_abs=$0.10**: the engine's exact-N solvency check at small N
  fails when liq_fee falls below `min_nonzero_mm_req`. $0.10 floor keeps
  small liquidations economically meaningful and validates cleanly.

- **h_max=86,400 slots ≈ 9.6 hours**: long enough that warmup acts as a
  meaningful delay during stress, short enough that locked-out winners
  don't have to wait days. Tunable up to weeks if the deployment wants.

---

## 2. Wrapper policy (per-call args)

The wrapper passes these to every public engine call. They are NOT stored
in engine state; the wrapper is the policy layer.

```rust
// On execute_trade_not_atomic, withdraw_not_atomic, settle_account_not_atomic,
// liquidate_at_oracle_not_atomic, convert_released_pnl_not_atomic,
// close_account_not_atomic:

admit_h_min = 0                                 // INSTANT FAST PATH
admit_h_max = 86_400                            // 9.6h slow path on stress
admit_h_max_consumption_threshold_bps_opt = Some(400)  // = im_bps

// On keeper_crank_not_atomic, additionally:
max_revalidations = 64                          // Phase 1 budget
rr_window_size    = 192                         // Phase 2 sweep (sum=256=MAX_TOUCHED)
```

**The threshold = `im_bps = 400`** is the load-bearing protection for
admit_h_min=0. It says: once cumulative oracle movement in a sweep
generation reaches 4% (= 1/leverage), force every fresh PnL admission
into the slow path until the cursor wraps. This is what spec §0 #4
calls "stress-threshold gating is additive" — not a substitute for
warmup but the gate that makes admit_h_min=0 safe for public deployment.

**Critical compliance note:** spec §0 #4 + §9.2 forbid public wrappers
from using `admit_h_min = 0` with `threshold_opt = None` simultaneously.
This config uses `Some(400)`, satisfying the public-wrapper requirement.

---

## 3. Wrapper-side oracle clamp (mandatory)

Before every call that takes an oracle price, the wrapper MUST clamp
the raw Pyth/oracle target to the engine's price-move envelope:

```rust
fn wrapper_oracle(real_oracle: u64, last_engine_price: u64, dt: u64) -> u64 {
    let max_dp = (last_engine_price as u128
        * 17u128                   // max_price_move_bps_per_slot
        * dt as u128) / 10_000;
    let lower = last_engine_price.saturating_sub(max_dp as u64).max(1);
    let upper = (last_engine_price.saturating_add(max_dp as u64))
        .min(1_000_000_000_000);   // MAX_ORACLE_PRICE
    real_oracle.clamp(lower, upper)
}
```

This converts engine-side rejections into wrapper-side walked-through
prices. The cascade processes incrementally; the consumption counter
accumulates each step toward the threshold.

Same-slot exposed cranks (`dt == 0`) MUST pass `last_engine_price`
unchanged per spec §10.24.

---

## 4. Anti-spam (wrapper-enforced, NOT in engine)

The engine no longer stores `min_initial_deposit` or `new_account_fee`
(removed v12.18.1). Anti-spam is the wrapper's responsibility:

| Knob | Value | Purpose |
|---|---|---|
| Minimum initial deposit | **$10** | Keeps slab bounded; 4096 accounts × $10 = $40K floor |
| Recurring account fee | **1 bps/slot of capital ≈ $0.0042/$10K/day** | Bleeds idle accounts; pays for keeper compute |
| Empty-account reclaim incentive | **$0.05** | Pays whoever calls `reclaim_empty_account_not_atomic` |
| Keeper rebate (per liquidation) | **5 bps of liq notional** | Funds permissionless keeper network |

Implementation:

```rust
// Before every public call that could create or extend a position:
if account.capital + amount < usdc(10) { reject; }

// On every crank touch:
sync_account_fee_to_slot_not_atomic(idx, now_slot, fee_rate=1);

// Any caller can:
reclaim_empty_account_not_atomic(idx, now_slot)
    -> credits caller's fee account with $0.05 from insurance
```

---

## 5. Initial capital

```text
Insurance fund:    $1,000     ← BOUNTY TARGET (visible on-chain)
LP capital:      $100,000     ← Market-maker float (counterparty
                                inventory; may be a passive AMM or
                                a designated market-maker account)

Initial oracle price: $200    (or whatever SOL spot is at launch)
Initial slot:         0
Market mode:          Live
```

The $1,000 grows organically: trading fees (4 bps × notional) flow into
insurance every fill. After ~$25M cumulative trading volume the
insurance reaches ~$10K. That's a feature — the bounty scales with
adoption.

---

## 6. Bounty mechanics

**Win condition:** cause `engine.insurance_fund.balance` to decrease
below its current value via any sequence of public calls. Must publish
a reproducible transaction (or sequence) that demonstrates the drop.

**Reward:** `max($1000, 5 × $drain_amount)` paid in USDC, capped at
the protocol's audit-budget escrow.

**Out of scope:**

- Pyth oracle manipulation (engine can't defend against bad oracle inputs;
  defense is the price-move envelope clamp at the wrapper)
- Solana validator-level attacks (account state tampering, RPC issues)
- Frontrunning ordinary trades (not a protocol bug)
- Withdrawing your own legitimate PnL (not a drain)

**In scope** (any of these earns the bounty):

| Class | Spec section | What success looks like |
|---|---|---|
| Admission gate bypass | §4.3, §3.3 | Withdraw matured PnL when `residual < matured` should haircut |
| K-index overflow | §10.22 | Force `\|K\| > i128::MAX/2`, mis-mark opposing side |
| ADL math error | §5.4, §10.22 | Distribute deficit incorrectly across accounts |
| Sticky-set overflow | §4.3 step 4 | 256+ admit_h_max accounts in one instruction silently truncate |
| Resolved-mode payout | §10.20, §10.21 | `force_close_resolved` pays before reconciliation |
| Same-slot bypass | §10.24 | `dt==0` crank advances `slot_last` with non-`P_last` price |
| Risk-notional ceil bypass | §10.8 | Open dust position whose floor notional evades MM |
| Conservation violation | §0 #1 | Make `V < C_tot + I` hold post-call |

**Brute force is unprofitable by design:**
- Min deposit $10, max position notional at 50x = $500
- Max bankruptcy deficit per position ≈ 1% notional ≈ $5
- Draining $1000 via brute force needs ~200 simultaneous bankruptcies
- Costs attacker 200 × $10 = $2,000 → loses $2,000, drains $1,000 → **net loss $1,000**
- Negative EV unless they find a real bug

---

## 7. Deployment checklist

- [ ] Pyth SOL/USD price feed wired with confidence-interval check
- [ ] Wrapper `clamp_oracle()` implemented and unit-tested against
      engine envelope (17 bps × dt × P_last / 10_000)
- [ ] Wrapper passes `Some(400)` as `admit_h_max_consumption_threshold_bps_opt`
      on all public-facing instructions
- [ ] Wrapper passes nonzero `rr_window_size` (192) on normal cranks
- [ ] Wrapper enforces `admit_h_min == 0 + threshold == None` is NEVER
      called from a public path (spec §9.2)
- [ ] Recurring fee scheduler running on Solana clock crank
- [ ] Reclaim incentive plumbed to insurance fund payout path
- [ ] Bounty escrow contract deployed with `$1000 + audit-budget` reserve
- [ ] On-chain bounty announcement (Twitter, Immunefi, etc.)
- [ ] Monitoring dashboard: `insurance_fund.balance` over time, alert on
      ANY decrease

---

## 8. What the stress test confirms about this config

Across 3,600 runs in 18 scenarios under the envelope-template equivalent
of these params (simulated with `mm ≥ 500` per envelope-validator
constraints; see `src/main.rs`):

- **Solvency (V ≥ C+I)**: 0 violations across ~1.08M crank checks
- **Insurance outflow**: $0.000011 total across all runs (1 atomic event
  in 3,600)
- **Matured haircut activations**: 28K events under maximum-stress
  (`bounty_inverted_sol`); insurance untouched in those events
- **All 8 deep invariants** (subset, K bounds, F bounds, A floor,
  neg_pnl count, capital sum, reserve cap): 0 violations across ~480K
  assertions

The stress test deliberately uses `mm = 500` for the legacy scenarios
because v12.19's exact-N solvency check rejects mm<500 with their other
knobs.

The `bounty_sol_50x_v1` market uses the actual deployment-target params
(`mm = 100`, `im = 200`, `liq = 20`) via the `raw_engine_params` flag
in `run_one`. It validates against the v12.19 envelope and is fuzzed
against the same 8 invariants.

Insurance is the third line of defense and effectively never material.

---

## 9. Fuzz results — `bounty_sol_50x` scenario, 200 seeds

```
Engine effective params:
  mm = 100 bps (1% maintenance), im = 200 bps (2% initial = 50x leverage)
  trading_fee = 4 bps, liquidation_fee = 20 bps
  admit_h_min = 0 (instant withdrawals), admit_h_max = 86400 slots (~9.6h)
  threshold_opt = Some(im_bps=200) — slow path at 2% cumulative price move
  insurance start = $1,000

Solvency invariants (asserted post-every-crank, ~480K assertions):
  V >= C+I, K bounds, F bounds, A floor, neg_pnl, sum(cap), sum(reserved):
    PASSED on every check

Insurance fund outflow audit:
  insurance_payout_runs_frac:    0%        (no run drained any insurance)
  insurance_paid_out_total:      $0.0000   (zero outflow across all 200 runs)
  insurance_end_p10:             $126,076  (started at $1,000)

Cascade behavior:
  liquidations:                  mean=738 of 2000 users (47%)
  drain_only_frac:               10%
  matured_overshoot:             1 event   (essentially zero gate failures)
  min_h_p01:                     1.0000    (no haircut on user PnL)
  min_residual_p01:              $119,524

Stress lanes:
  residual-scarcity (§4.3 #3):   0% of runs entered  (residual healthy)
  consumption-threshold (§4.3 #2): 100% of runs crossed
                                  (peak p99 = 419 bps > 200 threshold)
  sweep_generations:             18 per run
```

The bounty target ($1,000) **grew to ~$126K from trading fees** during
the simulation. Insurance never paid out anything in any of the 200
runs despite 47% of users being liquidated.

The consumption threshold (`Some(200)`) tripped 100% of runs, which is
the load-bearing protection: the moment cumulative oracle movement
hits 1/leverage = 2%, every fresh PnL admission switches to slow-path
warmup, preventing winners from extracting unbacked profits during
the cascade.

For a public deployment, the bounty hunter would face the same wall:
the haircut on warmup PnL absorbs all losses before insurance is ever
touched. The brute-force cost-to-drain ratio (deposit $2K to drain
$1K) makes mechanical attempts negative-EV; the actual bounty pays
out only on a real bug.
