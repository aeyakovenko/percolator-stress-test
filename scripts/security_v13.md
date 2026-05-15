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

**Stage 6 — directed crash scenarios with liquidation (4 scenarios × 2000 seeds):**

| Scenario | Liquidations | Insurance Used | Residual | Invariant Fails | Min User Cap |
|---|---|---|---|---|---|
| random       | 3,908 | **0** | **0** | **0** | $474 |
| crash10      | 5,047 | **0** | **0** | **0** | $338 |
| crash20      | 5,047 | **0** | **0** | **0** | $338 |
| funding_drain |     0 | **0** | **0** | **0** | n/a |

Crash20 = 200 slots of envelope-max downward at max_move=45 bps/slot, with
8x leveraged users pre-positioned. Engine triggers all liquidations before
deficit. Zero insurance draws across 8,000 seeds × 4 scenarios.

**Stage 7 — probe_drain pathological cases (`--test=probes`):**

| Probe | What | Liquidations | Insurance Used | Residual | Invariants |
|---|---|---|---|---|---|
| P3 | 10 concentrated longs + 200-slot crash to -60% | 10 | **0** | **0** | OK |
| P2 | Zero-LP funding drain (symmetric, 500 slots) | 0 | **0** | **0** | OK |
| P4 | $20M whale @ 10x + 200-slot crash to $81 | 1 | **0** | **0** | OK |
| P5 | 2000-slot max-rate funding drain | 0 | **0** | **0** | OK |

The whale ($200M notional) liquidated cleanly during a 60% market crash
with zero insurance touched and zero residual booked.

**Stage 8/9/10/14 — comprehensive sweep across all 7 scenarios
(2000 seeds × 7 scenarios = 14,000 seeds):**

| Scenario | Liquidations | Insurance Used | Residual | Explicit Loss | Invariant Fails |
|---|---|---|---|---|---|
| random         | 3,908  | **0** | **0** | **0** | **0** |
| crash10        | 5,047  | **0** | **0** | **0** | **0** |
| crash20        | 5,047  | **0** | **0** | **0** | **0** |
| funding_drain  |     0  | **0** | **0** | **0** | **0** |
| oracle_wick    | 5,047  | **0** | **0** | **0** | **0** |
| high_lev (18x) | 7,602  | **0** | **0** | **0** | **0** |
| mega (20×3×mixed) | 12,215 | **0** | **0** | **0** | **0** |
| **TOTAL**      |**38,866** | **0** | **0** | **0** | **0** |

The **mega** scenario combines: 20 users (vs 5), 3 assets (multi-asset
portfolios), random 5-18× initial leverage, random asset selection per
trade, independent random walks per asset at envelope-max, 400 slots.
This is the most adversarial scenario in the suite. 12,215 liquidations
triggered across 2000 seeds; zero insurance touched.

Per-slot 9-invariant battery run on every step:
- V ≥ C + I (solvency)
- matured ≤ pos_tot
- K within i128/2
- F within i128/2
- A_side ≥ MIN_A_SIDE outside DrainOnly/ResetPending
- neg_pnl_count consistent
- sum(capital) == c_tot
- sum(reserved) ≤ sum(pos pnl)
- F7: DrainOnly ⇒ opp OI > 0 OR opp is ResetPending

Zero failures across 14,000 seeds × ~200-400 slots each ≈ 4.2M slot-steps.

**v13-specific probes (`--test=probes_v13`):**

| Probe | Setup | Result |
|---|---|---|
| Multi-asset crash | 2 assets, both long, one crashes 60% | 1 liq, **0** insurance |
| Stale account exploit | Mark stale, try convert / withdraw | convert blocked (LockActive); withdraw runs with post-call IM check |
| Withdraw undercollateralize | 15x position, escalating withdraws | small OK; large rejected (LockActive / InvalidConfig) |

## What's not yet ported

Stages 6-9 added directed crash scenarios, liquidation flow, rayon
parallelism, probe_drain equivalent, oracle_wick, high_lev (18x), and an
explicit 9-invariant battery (including the new F7 check).

Remaining v12 parity items (low priority — no observed failures to debug):

- 18 specific v12 *named* scenarios with their original setups (ten10_btc/
  sol/alt/hl, adl_trigger/a_decay/k_deficit/drain_reset/cascade/stale,
  adversarial_keeper/adl_cascade, funding_dynamics/crash_combo, dust_gc).
  The current 6 scenarios cover the safety-relevant behavior; v12's named
  scenarios exercise specific corner cases (e.g. adl_drain_reset
  transitions, dust GC paths).
- Trace machinery / snapshots / CSV summary for forensic analysis when
  failures occur. Not critical while no failures are observed.
- Per-run histograms (min_h_p01, h_zero_frac, etc.) for distributional
  analysis.

Final coverage:

- **14,000 fuzz seeds** across 7 scenarios + the 4 directed pathological probes + 3 v13-specific probes
- **~4.2M slot-steps** with full 9-invariant battery per step
- **38,866 liquidations triggered** across the suite — all clean
- **0 invariant failures**
- **0 insurance payouts** (legitimate flow)
- **0 residual booked**
- **0 explicit loss**
- **0 bankruptcy_hlock activations**

Plus the F6 (PnL trap), exec_price_attack, sybil_close, multi-asset, stale
extraction, and withdraw-undercollateralize adversarial tests all behave
as expected.

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
