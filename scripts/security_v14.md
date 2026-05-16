# Percolator v14 security pass — Global Cross Margin

Engine: `percolator` branch `v14` at commit `a479f92`.
Stress test: `percolator-stress-test` branch `v14`.

This is a re-audit against v14, focused on the new attack surface
introduced by **Global Cross Margin** (v14.12.0 spec).

## What's new in v14

| Surface | v13 | v14 |
|---|---|---|
| Account equity | `capital + pnl - fee_debt` | `capital + min(pnl, 0) + haircut_effective_support(max(pnl, 0), residual, junior_bound) - fee_debt` |
| Cross-leg PnL aggregation | Per-leg risk, additive maintenance | **Shared portfolio_maintenance_req across all legs**, positive PnL haircut as support |
| Hedge credit | None | Configurable via SameUnderlyingExact / ExplicitFamilyWithGap |
| Insurance domains | Single fund | Domain-budgeted per (asset, side) — `V14_DOMAIN_COUNT = 2*max_portfolio_assets` |
| Bankruptcy residual | Single global pool | Per-leg attribution; no global bad-debt pool |
| Quantity ADL | Asset-scoped | Account-scoped with close_progress ledger |

The headline v14 invariant from spec §1: *"global_cross_margin_all_legs_support_maintenance"*  AND  *"global_cross_margin_does_not_create_global_B_domain"*. Cross-margin only mutualizes upside (positive PnL supports MM); bankruptcies stay leg-attributed.

## Empirical baseline (`--fuzz-all=2000`)

7 scenarios × 2000 seeds = 14,000 fuzz runs × ~200-400 slots = ~4M slot-steps.

| Scenario | Liquidations | Insurance Used | Residual | Explicit Loss | Invariant Fails |
|---|---|---|---|---|---|
| random         |  3,698 | **0** | **0** | **0** | **0** |
| crash10        |  5,047 | **0** | **0** | **0** | **0** |
| crash20        |  5,047 | **0** | **0** | **0** | **0** |
| funding_drain  |      0 | **0** | **0** | **0** | **0** |
| oracle_wick    |  5,047 | **0** | **0** | **0** | **0** |
| high_lev (18x) |  7,602 | **0** | **0** | **0** | **0** |
| **mega (20×3×mixed)** | **40,389** | **0** | **0** | **0** | **0** |

Mega scenario triggered **3.3× more liquidations than v13** (40,389 vs
12,215) — cross-margin makes shared-equity move faster, so the keeper
liquidates the largest leg more often as cumulative deficit grows.
Safety properties all hold.

## v14 cross-margin probes (`--test=xmargin`)

Three directed tests of the new surface:

### probe_xmargin_offset — offsetting legs in one account

- User: long $5k asset A + short $5k asset B
- Move both A and B up 10%
- Long profits +$500, short loses $500 → net pnl ≈ 0
- Account equity correctly nets the offsetting moves
- **Verified:** `cert.equity = $999` (cap minus fees), `mm_req = $550` (5% of $11k summed notional after 10% rise), `liq_deficit = 0`

The hedge correctly nets across legs without any insurance touch.

### probe_xmargin_asymmetric — one leg crashes, one stable

- User: long $8k asset A + long $8k asset B on $2k capital (8x portfolio)
- Asset A crashes 60%; B stays flat
- **Verified:** engine liquidates A leg only (1 liq, asset 0); B leg stays active; 0 insurance used
- User retains $704 of $2k — losses on A absorbed via shared capital pool

Cross-margin shared capital absorbs losses without cascading to B's leg.

### probe_xmargin_haircut — face positive PnL haircut under tight residual ★

- 10 users × $100 cap, all open longs at 5x leverage
- Small LP ($1M total vault) → `residual = vault - c_tot - insurance ≈ 0`
- Move oracle +10% → each user has face PnL +$50
- **Verified:**
  - `pnl_pos_tot = $500` (total face positive claims)
  - `sum(cert.equity) = $999.5` (≈$99.95 per user — basically just capital)
  - `sum(face positive pnl) = $500`

★ **The $500 face positive claim is haircut to ~$0 of senior value** because
`junior_claim_bound > residual`. This is v14's defining cross-margin
property: **no leg-local paper profit can become withdrawable senior
value** unless residual covers it. If users tried to convert + withdraw
their +$50 PnL, the haircut bounds it to whatever the residual supports.

## v13-port tests on v14 engine

All 30 stages of v13 stress tests run cleanly on v14:
- exec_price + sybil attacks — still defended (PnL is K-snap based, not exec_price)
- F6 stress pause — still wrapper-controlled
- Multi-leg probes (hedge mask, 8-leg saturation) — clean
- probe_drain (4 pathological cases) — clean
- v12 corner cases (adl_drain_reset, dust_gc, adversarial_keeper) — clean
- Account close / dt gap / churn — clean

The v13 audit findings carry over:
- F1/F9 (exec_price LP drain) — still fixed structurally
- F6 — same conservative pause via `threshold_stress_active`
- Lazy settlement — same wrapper-responsibility profile
- Keeper liveness — same deploy requirement

## v14-specific deploy considerations

1. **Cross-margin can speed liquidation cadence.** Multi-leg portfolios
   with mixed assets need keepers fast enough to handle the increased
   liquidation rate (3.3× observed in mega scenario). The §1.4 envelope
   still bounds per-slot loss, so per-slot keeper budget is unchanged.

2. **Positive PnL is haircut as support.** Users with profitable
   positions in an under-residual market cannot use the full face value
   of their PnL as MM cushion. Documenting this in user-facing UI
   prevents surprise during volatility events.

3. **Insurance is now domain-budgeted.** Each (asset, side) pair has
   its own insurance allocation. The current probe set doesn't exercise
   the domain-specific accounting deeply — worth a follow-up if the
   bounty deployment configures non-trivial domain budgets.

4. **Cross-margin DOES NOT create global B domain.** Bankruptcy on
   asset A's long side does not socialize loss across asset B's longs.
   Per-leg attribution is preserved by spec §5 design.

## Drift-style bad-asset attacks (`--test=drift`)

The Drift Protocol exploit (2021) extracted real value via oracle/asset
configuration vulnerabilities. v14's cross-margin design introduces a
new theoretical surface for similar attacks: a bad/manipulable asset in
the portfolio could be used to inflate equity that supports real losses
on other assets. Six probes target this surface:

| Probe | What it tests | Result |
|---|---|---|
| **A: thin-market** | Attacker holds majority of OI on asset 1; tries to manipulate its oracle | Open on attacker's "thin" leg failed — IM check blocks over-leverage across the portfolio |
| **B: phantom PnL** | Hedged long+short across 2 assets, correlated crash; tries to extract via withdraw | Hedge worked correctly during 36% correlated crash (cap preserved). Withdraw allowed when account was actually healthy. Subsequent reversal caused real loss with 0 insurance use. |
| **C: stale oracle ★** | Asset 0 crashes 60%; wrapper updates asset 1 oracle only every 20 slots (lazy keeper on asset 1) | **Withdraw blocked with `LockActive` due to oracle divergence between raw_oracle_target_price and effective_price.** The v14 `target_effective_lag` defense is what gates this. |
| **D: one-sided OI** | 10 longs on asset 1 (all against LP, no real shorts); 83% crash | 10 clean liquidations, 0 insurance used, 0 residual. Engine's §1.4 envelope handles one-sided OI via LP-as-counterparty. |
| **E: pump-and-withdraw** | Pump one leg, close+withdraw profit, let opposing leg blow up | Attacker withdrew $400 of capital when system was healthy; reverse move caused liquidation with 0 insurance use. Net attacker loss $330 on $1000 deposit. |
| **F: cross-asset contagion** | One user bankrupts on asset 0; verify another user on asset 1 is unaffected | asset1_user's `cert.equity_change = 0` after asset0_user's bankruptcy. **Per-leg attribution preserves the `does_not_create_global_B_domain` invariant.** |

**Headline finding (Probe C):** v14's `target_effective_lag` defense is
the primary protection against oracle-divergence attacks on the cross-margin
account equity. Any leg with a stale-vs-target oracle gate blocks all
favorable actions (withdraw, convert PnL, close-favorably) on the entire
account. This is the strongest defense against "Drift-style" extraction.

**Per-leg attribution (Probe F):** v14's bankruptcy residual booking
stays per-leg-attributed. Cross-margin gives equity benefits but NOT
loss-sharing. An attacker who bankrupts on a bad asset cannot drag down
healthy users on other assets.

## HARD stress: 10/10 + aggressive Drift-hack (`--test=hard`)

### probe_ten10_single_asset

50 users × 9x leverage long on a single 10x-leverage market. Oracle crashes
10% in 12 envelope-max slots, then continues to a total 50% crash.

| Metric | Value |
|---|---|
| Liquidations | 50 (one per user, in time) |
| Insurance used | **0** |
| Residual booked | **0** |
| Explicit loss | **0** |
| Sum user cap (initial $50k) | $41,670 (~17% from fees + position close) |
| Min user cap | $833 (no one fully bankrupted) |
| LP capital change | $9.999955M ≈ unchanged (just fees) |
| Side modes | both Normal throughout |

The §1.4 envelope at 10x leverage allows max_move=90 bps/slot → engine catches
all bankruptcies before deficit. No DrainOnly transitions, no ADL needed.

### probe_ten10_cross_margin

30 users × 3 legs on 3 assets at ~4.5x portfolio leverage. Only asset 0
crashes 10%.

| Metric | Value |
|---|---|
| Liquidations | **0** |
| Insurance used | **0** |
| Sum user cap loss | ~$9k of $60k (crash loss on asset 0 + fees) |
| Avg active legs per user | 3 (no legs liquidated) |

**Cross-margin diversification protects users.** A 10% drop on 1/3 of each
user's exposure is a 3.3% portfolio loss — well within the 10% MM buffer.
Zero liquidations triggered.

### probe_drift_hack_aggressive ★

Reconstructs the Drift Protocol attack pattern: pump a thin-market asset's
oracle to inflate cross-margin equity, then attempt extraction.

Setup:
- 5 bystanders with legitimate $5k longs on asset 0
- Attacker opens **$500 long on thin asset 1** (attacker is the only long; LP is sole counterparty)
- Attacker pumps asset 1 oracle envelope-max for 155 slots: **$200 → $400 (+100%)**
- Attacker's face PnL on asset 1 = **+$500** (100% of $500 notional)

Observed cross-margin behavior:
- `face_pnl = $500` (correctly tracked)
- `cert.equity = $1999.95` (≈ capital only — **PnL haircut to near zero**)
- `residual ≈ $0` (only ~$4 of vault excess over c_tot + insurance)
- `pnl_pos_tot = $500` (junior bound)
- haircut(500, 4, 500) ≈ $4 of effective support — **>99% of face PnL is haircut away**

Extraction attempts:
1. `withdraw $2000` → **`LockActive`** (insufficient equity beyond capital)
2. Close the profitable thin leg to realize PnL (which succeeded as a trade)
3. `withdraw $2000` post-close → **`LockActive`** still
4. **Final: attacker ended with $1996, having lost ~$4 to fees**

The defining v14 cross-margin guarantee — **no leg-local paper profit
becomes withdrawable senior value** — holds even against an attacker
who fully controls the oracle on a thin asset.

The attacker DID succeed in opening a $30k notional position on asset 0
(real capital backed the $1.5k IM at 5% IM), but could not extract any
value from the pumped asset.

## Empirical conclusion

v14's safety surface is at least as strong as v13's across all tested
flows. The cross-margin design adds upside mutualization (positive PnL
supports MM across legs) while preserving per-leg bankruptcy attribution.
The empirical baseline shows zero invariant failures and zero insurance
payouts on legitimate flows across all tested scenarios.
