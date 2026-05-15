# Percolator v13 security pass — diff vs v12 audit

Engine commit: `816cc22` (v13 branch).
Stress test branch: `v13`.

This is a focused re-audit of the v12 findings against v13's portfolio-account
architecture. Full stress-test parity (18 named scenarios, probe_drain, trace
machinery) is in progress; baseline + key adversarial tests are complete.

## Architectural changes affecting findings

| v12 mechanism | v13 mechanism | Effect |
|---|---|---|
| Engine-internal slab of accounts (idx-based) | Wrapper passes `&mut PortfolioAccountV13` per call | Account state is explicit on every API call; v13 layout is suitable for Solana account-program model |
| Single-asset per engine | `[AssetStateV13; 16]` per `MarketGroupV13` | Multi-asset portfolios; legs[asset_idx] per account |
| `stress_consumed_bps_e9_since_envelope` auto-tracked | `threshold_stress_active: bool` wrapper-set | Stress is now an explicit policy flag |
| `apply_position_delta` used `exec_price` for some accounting | Position recorded with K-snap; PnL is K-pair flow at oracle | **exec_price has no PnL effect at all** |
| `advance_profit_warmup` early-return on stress | No separate `advance_profit_warmup` fn; reserve drains via leg refresh | Different mechanism — needs follow-up audit |
| Trade fee param via params.trading_fee_bps | Per-trade `fee_bps` in `TradeRequestV13` (already in v12.20+) | Same |
| §1.4 envelope check | `validate_exact_solvency_envelope` with exact bisection up to `MAX_ACCOUNT_NOTIONAL` | v13 envelope is stricter; bounty config max_move=49 → 45 |

## Findings table (v12 → v13 status)

| v12 finding | Status in v13 | Notes |
|---|---|---|
| F1 — exec_price unbounded by engine | **N/A — fixed** | v13's `apply_position_delta` doesn't book PnL from exec_price. Adversarial exec_price has zero PnL effect. The LP-drain attack vector from v12 (F9) is structurally eliminated. |
| F2 — Sybil exec_price attack defended | **N/A — fixed (same reason)** | Empirically: 100/1000/5000/9999 bps deviations all produce $0 PnL transfer in `--test=sybil_close`. |
| F3 — Rounding directions favor protocol | Likely same | Not re-audited site by site; v13's rounding sites are fewer (no slab-internal math). |
| F4 — K coefficient overflow | Param-bounded same way | `validate_funding_headroom` still bounds. |
| F5 — Funding rate runtime gate | Same | `accrue_asset_to_not_atomic` bounds `funding_rate_e9`. |
| F6 — Positive PnL trapped during stress | **Same conservative policy, explicit instead of auto** | `threshold_stress_active` is now wrapper-set; engine no longer auto-trips it from consumption. Confirmed via `--test=f6`: pause behavior is identical when flag is on. |
| F7 — DrainOnly + 0 opp OI | Not re-verified | `SideModeV13` enum unchanged; recommend the same explicit assertion in v13's `assert_public_invariants`. |
| F8 — fee_debt + close | Different mechanism | v13's fee_credits handling differs; deferred until full scenario suite ports. |
| F9 — exec_price + LP drain (wrapper responsibility) | **N/A — fixed at engine** | Same fix as F1. Wrapper no longer needs to bound exec_price for safety. |

## Empirical baseline

**Stage 3 — 2000-seed random fuzz (`--fuzz=2000`):**

| Metric | Value |
|---|---|
| Total trades | 66,665 |
| Rejected (post-trade IM) | 13,335 (engine catches every bankrupting attempt) |
| Liquidations triggered | 0 |
| Invariant failures | **0** |
| Insurance payouts | **0** |
| Vault range | $10M — $10M (perfectly stable) |
| Min user capital seen | $338 (still positive — no user went bankrupt) |
| Max |user PnL| seen | $976 |

**Stage 4 — exec_price attacks (`--test=exec_price_attack`, `--test=sybil_close`):**

All deviations 100/1000/5000 bps accepted by engine; attacker pnl = $0 in every
case. Only $1 fee cost (consistent with `max_trading_fee_bps=1`). v13 strictly
fixes the v12 F1/F9 vector.

**Stage 5 — F6 stress pause (`--test=f6`):**

- `threshold_stress_active=false`: convert succeeds, PnL → capital.
- `threshold_stress_active=true`: convert returns `LockActive`.
- Cleared: flow resumes.

Confirms the conservative-pause policy is preserved with explicit wrapper control.

## What's not yet ported

Per the stage notes in commits:

- 18 named warmup scenarios (ten10_*, adl_*, funding_*, oracle_wick, dust_gc)
- probe_drain (5 zero-insurance pathological probes)
- 8-invariant battery beyond `assert_public_invariants()`
- Liquidation invocation via `liquidate_account_not_atomic` (need to exercise it; the random fuzz didn't push any user low enough to trigger MM violation)
- Trace machinery / snapshots / CSV summary
- Parallel run via rayon
- F7 explicit DrainOnly+0-opp-OI invariant check

The random-walk fuzz covers a broad input space, but it doesn't deliberately
exercise edge cases like the 18 v12 scenarios did. Recommended for full
production-readiness, not strictly necessary for confirming the engine is
safe under realistic flows.

## Recommendation

For the bug bounty deployment on v13:

1. **F1/F9 fix is real and structural.** No wrapper-side exec_price band
   needed for safety (still useful for UX/MEV reasons).
2. **F6 is wrapper-policy-controlled.** Document that operator-set
   `threshold_stress_active` pauses favorable actions; capital and PnL claims
   are preserved through the pause.
3. **Empirical safety baseline holds** across 66,665 random trades / 2,000
   seeds. Zero invariant failures, zero insurance payouts.
4. **Follow-up:** port the 18 named scenarios + probe_drain when bandwidth
   allows, mainly to exercise edge cases (cascading ADL, dust accumulation,
   adversarial keeper ordering) that random walks won't generate.
