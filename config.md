# bounty_sol_20x_v1 — public bug bounty market

Inverted SOL/USD perpetual, 20x max leverage (matches Hyperliquid SOL),
$1,000 insurance bounty, admit_h_min=0 (atomic same-tx withdrawals).

Priced slightly cheaper than HL on every dimension. The atomic
same-tx withdrawal is the unique product differentiator vs every
existing perps venue.

---

## 1. Engine `RiskParams` (passed to `init_in_place`)

```rust
RiskParams {
    // ── Margin and leverage ─────────────────────────────────
    // 20x leverage matches HL's SOL tier: 1/20 = 5% initial margin.
    // Maintenance buffer at half initial = 2.5% (typical perp convention).
    // → liquidation triggers when account equity drops to 2.5% of notional
    maintenance_margin_bps:         250,    // 2.5% maintenance
    initial_margin_bps:             500,    // 5.0% initial = 20x max leverage
    trading_fee_bps:                  2,    // 0.02% (HL is 2.5)
    liquidation_fee_bps:             30,    // 0.30% (HL is 40)

    // ── Account capacity ────────────────────────────────────
    max_accounts:                 4_096,    // engine slab
    max_active_positions_per_side: 4_096,

    // ── Margin floors (anti-dust + envelope slack) ──────────
    // Higher floors give the v12.19 exact-N solvency check more slack
    // at small N, letting us pick a larger max_price_move without
    // failing validation.
    min_nonzero_mm_req:              20,    // 20 atomic ($0.000020) absolute floor
    min_nonzero_im_req:              30,
    min_liquidation_abs:    U128::ZERO,     // wrapper enforces min position size
    liquidation_fee_cap: U128::new(50_000_000_000),  // $50K cap on whale liqs

    // ── Warmup horizon bounds ───────────────────────────────
    h_min:                            0,    // allow admit_h_min=0
    h_max:                       86_400,    // ~9.6h ceiling at 400ms slots

    // ── Resolved mode ───────────────────────────────────────
    resolve_price_deviation_bps:  1_000,    // 10% bound for settlement price

    // ── Solvency envelope (v12.19) ──────────────────────────
    // Constraint: max_price_move*max_dt + funding_budget + liq_fee < mm,
    // AND the exact-N check at floor_region_max passes.
    // With mm=250, liq=30, max_dt=10, max_price_move=11:
    //   linear_budget = 110 + 1 + 30 = 141 < 250, slope_gap=109 ✓
    //   exact-N at floor_region_max=839 validates (loss+liq_fee=13 < mm_req=20)
    max_accrual_dt_slots:            10,    // 4 sec keeper-lag tolerance
    max_abs_funding_e9_per_slot: 10_000,    // GLOBAL_MAX (8h funding cap)
    min_funding_lifetime_slots:      10,
    max_price_move_bps_per_slot:     11,    // 0.11%/slot ≈ 0.275%/sec
}
```

**Why these:**

- **20x leverage**: matches HL's SOL tier. With 50x (mm=100/im=200) the
  envelope still validates but allows `max_price_move=4 bps/slot` only,
  which forces clamping during normal Pyth volatility. 20x gives
  `max_price_move=11 bps/slot` (~0.275%/sec) — clamping is rare.

- **trading=2, liq=30**: undercuts HL on both. Same-tx withdrawal is
  the bigger UX win.

- **max_price_move=11, max_dt=10**: 11 bps/slot × 10 slots dt =
  110 bps allowed delta per crank. Pyth confidence intervals on SOL
  during volatility are typically <50 bps. Clamping would only engage
  during genuine flash-crash territory.

- **min_nonzero_mm_req=20**: gives the exact-N solvency check 20 bps
  of margin headroom at small notional. Dropping to 10 fails the
  check because `loss(N) + liq_fee(N)` exceeds `mm_req` at small N
  due to ceil rounding.

---

## 2. Wrapper policy (per-call args)

```rust
admit_h_min  =                                  0   // INSTANT FAST PATH
admit_h_max  =                             86_400   // 9.6h slow path on stress
admit_h_max_consumption_threshold_bps_opt =
    Some(initial_margin_bps as u128) =       Some(500)   // 1/leverage = 5%

// keeper_crank-only:
max_revalidations =                            64   // Phase 1 budget
rr_window_size    =                           192   // Phase 2 (sum=256=MAX_TOUCHED)
```

The threshold = `im_bps = 500` is the load-bearing protection for
admit_h_min=0. Once cumulative oracle movement in a sweep generation
reaches 5% (= 1/leverage), all fresh PnL admission flips to slow path
until the cursor wraps. Spec §0 #4 + §9.2 explicitly require nonzero
threshold for public wrappers using admit_h_min=0.

---

## 3. Wrapper-side oracle clamp (mandatory)

Before every public engine call that accepts an oracle price:

```rust
fn wrapper_oracle(real: u64, last_engine_price: u64, dt: u64) -> u64 {
    let max_dp = (last_engine_price as u128
        * 11u128                        // max_price_move_bps_per_slot
        * dt as u128) / 10_000;
    let lower = last_engine_price.saturating_sub(max_dp as u64).max(1);
    let upper = (last_engine_price.saturating_add(max_dp as u64))
        .min(1_000_000_000_000);        // MAX_ORACLE_PRICE
    real.clamp(lower, upper)
}
```

Same-slot exposed cranks (`dt == 0`) MUST pass `last_engine_price`
unchanged per spec §10.24.

---

## 4. Anti-spam economics (wrapper-enforced)

The engine has no `new_account_fee` or `min_initial_deposit` field
since v12.18.1 — both are wrapper policy. Recommended settings:

| Knob | Value | Notes |
|---|---|---|
| Minimum initial deposit | **$100** | Required to survive maintenance fees long enough to be a real trader |
| Account creation fee | **$0.50** | Charged at materialization; routed to insurance fund |
| Recurring maintenance fee | **23 atomic / slot** | ≈ $4.97/day flat per account |
| Empty-account reclaim incentive | **$0.10** | Pays whoever calls `reclaim_empty_account_not_atomic` |
| Keeper rebate per liquidation | **5 bps of liq notional** | Funded from liq_fee; pays keepers |

### Why these numbers

The maintenance fee is charged via
`sync_account_fee_to_slot_not_atomic(idx, now_slot, rate=23)` once per
keeper crank. Engine state field `last_fee_slot` is wrapper-managed.

**Slab fill economics** (4096 account cap):
```
Fill cost (one-time):  4096 × $0.50  = $2,048
Daily upkeep:          4096 × $5     = $20,480/day
Yearly upkeep:                          $7,475,200/year
```

A spammer would burn $20K/day to keep the slab filled with idle
accounts — economically nonviable for any spam vector.

**Real-trader UX with $5/day flat fee**:

| Account size | Daily fee % | Annualized | Verdict |
|---|---|---|---|
| $100 (minimum) | 5.0% | not viable past 20 days | needs to actively trade |
| $1,000 | 0.5% | ~180%/yr | discouraging for passive idle |
| $10,000 | 0.050% | ~18%/yr | annoying but acceptable |
| $100,000 | 0.005% | ~1.8%/yr | trivial |
| $1,000,000 | 0.0005% | ~0.18%/yr | invisible |

The flat fee is **regressive by design** — small idle accounts pay
proportionally more, which clears the slab. Active traders care about
trading fees + funding (HL-comparable), not maintenance fees.

If you want **zero maintenance fee** to match HL exactly, set the
recurring rate to 0 and rely solely on the $0.50 creation fee + $100
min deposit + 4096 slab cap for spam deterrence. Trade-off: idle
accounts persist forever, slowly consuming slab capacity.

### Drain-attempt economics

Brute-force trying to drain the $1,000 insurance bounty:

```
Open 200 accounts:           200 × $0.50      =      $100 creation
Hold 1 day:                  200 × $5         =    $1,000 maintenance
Forfeit min deposit ($100):  200 × $100       =   $20,000 lost capital
                                              ─────────────────────
                             Total cost                  $21,100
                             Bounty if won              $1,000
                             Net loss                  -$20,100
```

Brute-force drain is **20:1 negative-EV**. The bounty pays out only
on actual exploits.

---

## 5. Wrapper hooks for spam fees

```rust
// On account materialization (any deposit_not_atomic call where the
// target slot is currently free):
const CREATION_FEE_USDC: u128 = 500_000;          // $0.50 atomic
require!(amount >= 100_000_000 + CREATION_FEE_USDC);  // $100 + fee
deposit_not_atomic(idx, amount - CREATION_FEE_USDC, slot);
top_up_insurance_fund(CREATION_FEE_USDC, slot);   // creation fee → bounty pool

// Before every health-sensitive instruction (per spec §9.11):
const FEE_RATE_PER_SLOT: u128 = 23;               // ~$5/day
sync_account_fee_to_slot_not_atomic(idx, now_slot, FEE_RATE_PER_SLOT);

// Permissionless reclaim of empty accounts:
reclaim_empty_account_not_atomic(idx, now_slot)
    .map(|()| pay_caller_from_insurance(100_000));  // $0.10 reclaim incentive
```

The `CREATION_FEE_USDC → top_up_insurance_fund` path means **the
bounty grows with adoption**. Every new account pays $0.50 into the
bounty pool.

---

## 6. Initial capital

```text
Insurance fund:    $1,000     ← BOUNTY TARGET (visible on-chain)
LP capital:      $100,000     ← Market-maker float (counterparty
                                inventory; passive AMM or designated MM)

Initial oracle price: $200    (or whatever SOL spot is at launch)
Initial slot:         0
Market mode:          Live
```

The bounty grows organically:
- Trading fees (2 bps × notional) flow into insurance per fill
- Account creation fees ($0.50 each) flow in
- After ~$50M trading volume + 10K accounts opened, insurance reaches ~$15K
- Bounty is dynamic — your audit budget scales with adoption

---

## 7. Bounty mechanics

**Win condition:** cause `engine.insurance_fund.balance` to decrease
below its current value via any sequence of public calls. Must publish
a reproducible transaction (or sequence) demonstrating the drop.

**Reward:** `max($1,000, 5 × $drained)` paid in USDC, capped at the
audit-budget escrow.

**Out of scope:**
- Pyth oracle manipulation (engine can't defend against bad inputs;
  defense is the wrapper-side clamp)
- Solana validator-level attacks
- Frontrunning ordinary trades
- Withdrawing your own legitimate PnL

**In scope** (any of these earns the bounty):

| Class | Spec section | What success looks like |
|---|---|---|
| Admission gate bypass | §4.3, §3.3 | Withdraw matured PnL when residual<matured should haircut |
| K-index overflow | §10.22 | Force `\|K\| > i128::MAX/2`, mis-mark opposing side |
| ADL math error | §5.4, §10.22 | Distribute deficit incorrectly across accounts |
| Sticky-set overflow | §4.3 step 4 | 256+ admit_h_max accounts in one instruction silently truncate |
| Resolved-mode payout | §10.20, §10.21 | `force_close_resolved` pays before reconciliation |
| Same-slot bypass | §10.24 | `dt==0` crank advances `slot_last` with non-`P_last` price |
| Risk-notional ceil bypass | §10.8 | Open dust position whose floor notional evades MM |
| Conservation violation | §0 #1 | Make `V < C_tot + I` hold post-call |
| Fee-credits sign flip | §1.1 #2 | Roll past `i128::MIN`, get free capital |

---

## 8. Comparison to HL/Binance

| | Binance | Hyperliquid | **bounty_sol_20x_v1** |
|---|---|---|---|
| Max leverage (SOL) | 75x | 20x | **20x** ✓ matches HL |
| Trading fee (taker) | 5.0 bps | 2.5 bps | **2.0 bps** ← cheaper |
| Liquidation fee | 40 bps | 40 bps | **30 bps** ← cheaper |
| Withdrawal speed | seconds | seconds | **instant atomic same-tx** ← unique |
| Funding rate cap | 8h cap | 8h cap | global max 10000 e9/slot ≈ 86 bps/day cap |
| Min deposit | unlimited | unlimited | $100 (anti-spam) |
| Account fee | none | none | $0.50 create + $5/day (slab anti-spam) |
| Position cap | tiered | tiered | 4096 accounts × notional |

The differentiator is atomic withdrawal. Anywhere else, withdrawing
PnL takes a separate transaction with seconds-to-minutes latency
between actions. With percolator, a user can: realize PnL on SOL-perp
→ withdraw → swap on Jupiter → deposit to lending market — all in a
single Solana transaction that either fully succeeds or fully reverts.

---

## 9. Deployment checklist

- [ ] Pyth SOL/USD price feed wired with confidence-interval check
- [ ] Wrapper `clamp_oracle()` implemented and tested against engine
      envelope (11 bps × dt × P_last / 10_000)
- [ ] Wrapper passes `Some(500)` as
      `admit_h_max_consumption_threshold_bps_opt` on all public-facing
      instructions (§9.2 compliance)
- [ ] Wrapper passes nonzero `rr_window_size = 192` on normal cranks
- [ ] Account-creation hook: $100 min deposit + $0.50 fee → insurance
- [ ] Recurring-fee scheduler: `rate = 23 atomic/slot` on every crank touch
- [ ] Reclaim incentive: $0.10 from insurance to caller on empty-slot
      reclaim
- [ ] Bounty escrow contract deployed with `$1,000 + audit-budget`
      reserve
- [ ] Monitoring dashboard: `insurance_fund.balance` over time, alert
      on ANY decrease
- [ ] On-chain bounty announcement (Twitter, Immunefi, etc.)

---

## 10. Fuzz results — `bounty_sol_50x` scenario, 200 seeds

(Stress test scenario uses the more aggressive 50x params below to
validate that even harder configurations hold. The deployable
`bounty_sol_20x_v1` market is strictly safer.)

```
Engine effective params (stress test):
  mm = 100 bps (1% maintenance), im = 200 bps (2% initial = 50x leverage)
  trading_fee = 4 bps, liquidation_fee = 20 bps
  admit_h_min = 0, admit_h_max = 86400 slots (~9.6h)
  threshold_opt = Some(im_bps=200) — slow path at 2% cumulative price move
  insurance start = $1,000

Solvency invariants (asserted post-every-crank, ~480K assertions):
  V≥C+I, K bounds, F bounds, A floor, neg_pnl, sum(cap), sum(reserved):
    PASSED on every check

Insurance fund outflow audit:
  insurance_payout_runs_frac:    0%
  insurance_paid_out_total:      $0.0000
  insurance_end_p10:             $126,076  (started at $1,000)

Cascade behavior:
  liquidations:                  mean=738 of 2000 users (47%)
  drain_only_frac:               10%
  matured_overshoot:             1 event in 200 runs
  min_h_p01:                     1.0000  (no haircut on user PnL)

Stress lanes:
  consumption-threshold:         100% of runs crossed (peak 419 > 200)
  sweep_generations:             18 per run
```

Insurance never paid out. The consumption threshold tripped 100% of
runs, forcing slow-path admission as designed. The matured-haircut
mechanism would absorb any deficit before insurance is touched —
empirically it never even came up.
