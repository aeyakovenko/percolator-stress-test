# Bounty deployment config — v13 engine

**Engine:** `percolator` branch `v13` at commit `816cc22`.
**Stress test:** `percolator-stress-test` branch `v13` (this repo).
**Status:** ready for $1,000 bug bounty after operator review.

This document is the v13 equivalent of v12's `max_risk.md`. It specifies the
deploy params, fee schedule, leverage limits, and audit posture.

---

## TL;DR

```
Leverage:     20x max (maintenance margin 5%, initial margin 5%)
Trading fee:  1 bps (0.01%)  — paid by both sides per trade
Liquidation:  5 bps (0.05%)  — keeper reward; min $0, cap $50,000
Funding:      symmetric, capped at ±0.001% per slot (= ±0.864% per day worst case)
Max move:     45 bps per slot (4.5% per 10-slot window = ~1.1%/sec)
Withdrawals:  admit_h_min = 0 (instant when system unstressed)
                admit_h_max = 30 slots (12-sec warmup under stress)
Insurance:    $1,000 starting balance (the bounty target)
```

## Full V13Config

```rust
V13Config {
    max_portfolio_assets:                1,    // single inverted-SOL market
    min_nonzero_mm_req:                 20,
    min_nonzero_im_req:                 30,
    h_min:                               0,    // INSTANT withdrawals
    h_max:                              30,    // 30-slot warmup under stress
    maintenance_margin_bps:            500,    // 5% — 20x max leverage
    initial_margin_bps:                500,    // 5% — same as MM (no margin cushion)
    max_trading_fee_bps:                 1,    // 0.01% per side per trade
    liquidation_fee_bps:                 5,    // 0.05% — keeper reward
    liquidation_fee_cap:    usdc(50_000),      // cap per liquidation
    min_liquidation_abs:                 0,
    max_accrual_dt_slots:               10,    // 10 slots ≈ 4 sec (Solana)
    max_abs_funding_e9_per_slot:         0,    // funding off in initial bounty deployment
    min_funding_lifetime_slots:         10,
    max_price_move_bps_per_slot:        45,    // **v13-specific** — v12 allowed 49
    max_account_b_settlement_chunks:     8,
    max_bankrupt_close_chunks:           8,
    public_b_chunk_atoms:    MAX_VAULT_TVL,
    permissionless_recovery_enabled:  true,
    stale_certificate_penalty_enabled: true,
    full_refresh_required_for_favorable_actions: true,
    public_liveness_profile_crank_forward: true,
}
```

## v13-specific notes vs v12

| Surface | v12 | v13 | Implication |
|---|---|---|---|
| `max_price_move_bps_per_slot` cliff | 49 | **45** | v13 reserves slightly more envelope headroom — more conservative. |
| `exec_price` PnL effect | Yes (bounded by post-trade IM check) | **None** (position marked at oracle via K-snap) | F1/F9 vulnerability fixed at engine layer; wrapper no longer responsible for price band. |
| Stress flag | Auto-tripped from `stress_consumed_bps_e9_since_envelope` | **Wrapper-policy `threshold_stress_active` bool** | Operator chooses when to pause favorable actions. |
| Portfolio model | One asset per engine | **Up to 16 legs per account** | Multi-asset stress-tested in `--test=multileg`. |
| `apply_position_delta` | Used `exec_price` for some accounting | **Pure K-snap attach** | Cleaner separation between trade transfer and PnL accrual. |

## Wrapper responsibilities

1. **`raw_oracle_target_price` upkeep.** Each `AssetStateV13` has both
   `raw_oracle_target_price` (what Pyth/Switchboard reported) and
   `effective_price` (what the engine has marked). The wrapper drives
   the effective price toward the target via `accrue_asset_to_not_atomic`,
   one envelope-step at a time. If they diverge,
   `target_effective_lag` is true and trades that increase risk are
   blocked (`LockActive`).

2. **Funding rate generation.** Set funding_rate_e9 per accrue call.
   Must obey `|funding_rate_e9| ≤ max_abs_funding_e9_per_slot`. For initial
   bounty deployment funding is disabled (`max_abs_funding_e9_per_slot=0`).

3. **Keeper crank loop.** Every slot (or every few slots):
   - Pull oracle from Pyth/Switchboard.
   - Compute the envelope-bounded `effective_price`.
   - Call `accrue_asset_to_not_atomic` with `protective_progress_committed=true`
     once accounts have been touched.
   - Refresh each at-risk account via `full_account_refresh`.
   - For any account with `health_cert.certified_liq_deficit > 0`, call
     `liquidate_account_not_atomic` with the most appropriate leg.

4. **Emergency pause.** Operator can set
   `MarketGroupV13.threshold_stress_active = true` to pause favorable
   actions (withdrawals, PnL conversions, account closes). User funds and
   open positions are preserved; only outflows are paused. Clear the flag
   to resume normal operation.

5. **Resolve path.** If the market needs to be wound down,
   `resolve_market_not_atomic(slot)` moves to `Resolved` mode. From there,
   each account exits via a loop:
   - `settle_account_side_effects_not_atomic` until no chunk remains
   - `apply_quantity_adl_after_residual_not_atomic` to flatten positions
   - `close_resolved_account_not_atomic(account, fee_rate_per_slot)` to finalize payout

## Operating envelope (for context)

| Property | Value |
|---|---|
| Max oracle move per slot | 45 bps = 0.45% (~1.1%/sec) |
| Max oracle move per 10-slot window | 4.5% |
| Max keeper-lag tolerance (slots) | 10 (= 4 seconds) |
| Max liquidation fee per close | 5 bps × notional, capped at $50,000 |
| Worst-case MM headroom | mm − loss_budget − liq_fee ≈ 4 bps (v13 verified) |

## Audit coverage

See `scripts/security_v13.md` for the full audit. Highlights:

- **14,000 fuzz seeds** across 7 scenarios (random, crash10, crash20, oracle_wick,
  funding_drain, high_lev, mega 20×3×mixed).
- **~4.2M slot-steps** with a 9-invariant battery per step (V≥C+I,
  matured≤pos_tot, K/F bounds, A_side floor, neg_pnl_count consistent,
  sum(capital)==c_tot, sum(reserved)≤sum(pos_pnl), F7 DrainOnly+opp).
- **38,866 liquidations triggered** cleanly across the suite.
- **20+ directed/adversarial probes** including exec_price, sybil_close,
  multi-asset crash, stale exploit, withdraw undercollateralize, hedge mask,
  8-leg saturation, 4-leg high-lev crash, ADL drain-reset, dust GC,
  adversarial-keeper ordering, account close, dt-gap, churn, market resolve,
  and boundary inputs.
- **Zero invariant failures, zero insurance payouts, zero residual booked,
  zero explicit loss, zero bankruptcy h-lock activations.**

## Bug bounty scope

Reward: **$1,000 to the first reproducible exploit** that causes any of:

1. `insurance` balance to decrease as a result of a legitimate-flow trade
   sequence (the test backdoor zombie-injection path doesn't qualify).
2. Any 9-invariant battery failure during a non-malformed call sequence.
3. A user being unable to exit their position when the engine reports
   `health_cert.certified_liq_deficit == 0` and no stress flag is set
   (must show a permanent block, not a wrapper-misuse case).
4. An `apply_position_delta` flow that produces account state inconsistent
   with the engine's own invariants.

Out of scope:
- Operator-level malicious wrapper (e.g., wrapper that lies about oracle to engine)
- Solana-runtime / SVM bugs unrelated to engine logic
- Slowdown / cost attacks not affecting fund safety
- Race conditions in wrapper code (engine is `not_atomic` per call)

Submission: PR or issue at https://github.com/aeyakovenko/percolator with
a reproducer (Rust test or script using this stress-test repo's helpers).

## Deploy checklist

- [ ] Engine `816cc22` audited by independent reviewer (gap noted in `security_v13.md`)
- [ ] `bounty_v13.md` reviewed and approved
- [ ] Solana deployment program reviewed for wrapper-side responsibilities (§ above)
- [ ] Oracle source (Pyth/Switchboard) chosen and tested for envelope compliance
- [ ] Keeper bot deployed and verified for at least 24h on devnet
- [ ] $1,000 insurance fund seeded
- [ ] Public announcement of bounty terms
