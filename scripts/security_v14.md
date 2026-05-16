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

## Empirical conclusion

v14's safety surface is at least as strong as v13's across all tested
flows. The cross-margin design adds upside mutualization (positive PnL
supports MM across legs) while preserving per-leg bankruptcy attribution.
The empirical baseline shows zero invariant failures and zero insurance
payouts on legitimate flows across all tested scenarios.
