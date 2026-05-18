# v16 stress sweep — full results

**Date:** 2026-05-18
**Engine:** `v16.8.0 Realizable Full Shared Cross-Margin`
**Stress branch:** `v16`
**Probes run:** 31 / 31 pass execution; capital-efficiency probes show a regression vs the prior wiring.

## What changed in the engine since previous run

The K-pair settlement path now consults source-credit:

- `apply_haircut_bounded_close_loss_to_pnl` (v16.rs:7373): branches on `account_has_source_claims`. If true, uses `account_unliened_source_realizable_support` and `create_and_consume_account_source_credit_for_effective_not_atomic` instead of residual gating.
- `apply_signed_kf_delta_to_pnl` (v16.rs:7449): now accepts `source_domain: Option<usize>` and takes `MAX(global_effective_available, source_effective_available)` when supporting an existing loss.
- `settle_leg_kf_effects` (v16.rs:7833): passes `source_domain = Some(opposite_side(leg.side))` for `net > 0` settlement.
- `set_account_pnl_with_source` (v16.rs:8133): auto-tracks `account.source_claim_bound_num[domain]` and `source_credit[domain].positive_claim_bound_num` on positive PnL.

This is exactly the rewiring identified as missing in the previous report. The settle path is now correctly source-credit-aware — verified by `v16_pre_settle`:

```
v16_pre_settle: inject backing BEFORE settle, credit_rate=100%
Result: cap=$1000, pnl=$0, cert_eq=$1000
        → SPREAD FUNGIBILITY DELIVERED (HL-like)
```

This is a real improvement. **Manual backing now produces capital-preserving spread trades.**

## But: regression in healthy-market behavior

The new wiring uses source-credit **exclusively** when the account has source claims. It no longer falls back to residual-as-cushion. Since `full_account_refresh` still doesn't auto-create `BackingReservationPlan`, backing remains empty in healthy markets, and the account loses the residual-based protection it had under v15/old-v16.

Net effect on capital efficiency probes:

| Probe | Previous (old wiring) | Current (new wiring) | Change |
|---|---|---|---|
| Single-asset 2x lev, 30-day walk | 100% survival | **0% survival** | regression |
| Single-asset 5x lev, 30-day walk | 38% survival | **0% survival** | regression |
| Spread long SOL only (5x) | 39% survival | **0% survival** | regression |
| Spread long SOL + long ETH | 0.4% survival | **0% survival** | regression |
| Spread long SOL + short ETH | 2.0% survival | **0% survival** | regression |
| Diversification 1/2/3/4 ways | 100%/100%/100%/100% | **0%/0%/0%/0%** | regression |
| Mean-reversion ratchet ±10% | $0 lost | **−$1041 / −104%** | regression — worse than -100% (over-burn) |
| Round-trip spread (`spread_residual`) | $0 net change | $0 net change | unchanged |
| Manual backing pre-settle (`v16_pre_settle`) | $500 lost | **$0 lost** | new mechanism works |

## Root cause

In the new `apply_haircut_bounded_close_loss_to_pnl`:

```rust
let has_source_claims = Self::account_has_source_claims(account)?;
let effective_available = if has_source_claims {
    self.account_unliened_source_realizable_support(account, old_positive_face)?
} else {
    self.haircut_effective_support(old_positive_face, residual, junior_bound)?
};
```

When `set_account_pnl_with_source` auto-populates `account.source_claim_bound_num[domain]` from any positive PnL, `has_source_claims` becomes `true`. The branch then uses only `account_unliened_source_realizable_support`, which returns 0 when no backing was reserved.

**Previously**, even with residual = 0 globally, a single user's own prior losses would grow vault−c_tot and provide residual-as-cushion for their later gain leg. That path is now skipped.

Walked-out example at 5x lev, single seed (from `cap_eff` trace):

| Step | Price | Old cap | Old pnl | New cap | New pnl |
|---|---|---|---|---|---|
| 0    | $199 | $998 | $0     | $998 | $0     |
| 2000 | $189 | $739 | $0     | $739 | $0     |
| 3000 | $206 | $739 | +$424M | $739 | +$424M |
| 4000 | $209 | $739 | +$486M | $739 | +$486M |
| 5000 | $203 | $739 | +$234M | **$604** | **$0** |
| 6000 | $181 | $424 | $0     | **$55**  | $0 (deficit $171) |
| 6999 | $182 | $424 | +$15M (alive) | (already liquidated) | — |

At step 5000, the price drop should haircut the existing $486M positive PnL. Old wiring used residual ($254 = $998−$739 stuck in vault) to cushion → pnl falls to +$234M, cap unchanged. New wiring uses source-only (no backing) → entire $486M gain is burned, remaining loss settles $135 from cap.

## Fix

The fix is a one-line change in `apply_haircut_bounded_close_loss_to_pnl`: take the **MAX** of source and residual support, not the either-or branch.

```rust
let effective_available = std::cmp::max(
    self.account_unliened_source_realizable_support(account, old_positive_face)?,
    self.haircut_effective_support(old_positive_face, residual, junior_bound)?,
);
```

`apply_signed_kf_delta_to_pnl` already does this for the `account.pnl < 0` case (v16.rs:7490-7500). The same pattern is needed for `apply_haircut_bounded_close_loss_to_pnl`.

The second half of the spec promise — `BackingReservationPlan` auto-orchestration inside `full_account_refresh` — is still missing. Without it, the realizable-backing credit_rate stays at 0 unless a keeper/wrapper explicitly calls `add_fresh_counterparty_backing_not_atomic`. Until that lands, capital efficiency depends on the wrapper.

## Security probes (all still pass)

| Probe | Status |
|---|---|
| exec_price_attack | engine defended at 9999 bps deviation (LockActive) |
| sybil_close | 0 extraction across 100/1000/5000/9999 bps |
| hard_ext (2000-seed fuzz) | total withdrawn = $2,688,104 (= sum of deposits, no excess); max single withdraw = $2500 (= max deposit); 0 invariant fails |
| drift | bad-asset crash isolated to that asset; asset 1 (2x long) preserves cap=$499 (no contagion) |
| domain_attr | SOL bankruptcy charged $6.6M only to domain[1] (SOL long-side opp); BTC domains unaffected |
| corner_cases | adversarial keeper liquidates richest first; sum user cap $6,770 preserved |
| multileg | 4-leg user crash: insurance_used=$0, residual=$0, explicit_loss=$0 |
| f6 | conservative-pause wrapper-controlled (LockActive on stress=true) |
| boundary / config / resolve / pnl_trace | all paths exercised, no extraction |

## Manual-backing path — empirical proof of the spec mechanism

When the wrapper supplies backing (simulating what `BackingReservationPlan` would do automatically):

```
v16_pre_settle (inject before settle):
  Pre-settle inject (domain 2 = ETH Long): claim=true backing=true
  credit_rate = 100.00%
  After settle + refresh:
    cap=$1000, pnl=$0, cert_eq=$1000
  → SPREAD FUNGIBILITY DELIVERED (HL-like)
```

This is the v16 promise working end-to-end. The construction is correct. The missing pieces are:

1. **Auto-orchestration in `full_account_refresh`** — when an account refreshes with a loss, the engine should reserve that account's senior capital as backing for the opposing source domain. Currently the wrapper must call `add_fresh_counterparty_backing_not_atomic` manually.
2. **Residual-as-cushion fallback** — when source-credit gives zero support but global residual > 0, the engine should fall back to the residual mechanism rather than burning the gain leg entirely.

With those two changes, the spec's healthy-market capital efficiency claim should hold under stress.
