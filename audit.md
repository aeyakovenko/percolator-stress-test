# Percolator Stress Test Audit Results

**Spec version:** v11.26
**Total scenarios:** 31
**Runs per scenario:** 200
**Total simulations:** 6200
**Solvency violations:** 0
**Deficit fraction:** 0.0 across all runs

## Summary Table

| Scenario | Crash | min_h p01 | true_h p01 | final_h | Liqs | ADL A_red | Users Liq% | Capital p01 | deficit |
|---|---|---|---|---|---|---|---|---|---|
| 10_whale | 30% + whale | 1.000 | 1.000 | 1.000 | 6 | 5.9 | 2.6% | 0.091 | 0.0 |
| 11_funding | 30% + fund | 1.000 | 1.000 | 1.000 | 5 | 5.1 | 2.8% | 0.091 | 0.0 |
| 12_armageddon | 50% no bnce | 1.000 | 1.000 | 1.000 | 2 | 1.7 | 2.5% | 0.067 | 0.0 |
| 13_skew_lag5 | 30% lag5 | 1.000 | 1.000 | 1.000 | 1 | 1.1 | 2.1% | 0.068 | 0.0 |
| 14_armageddon_lag5 | stair lag5 | 1.000 | 1.000 | 1.000 | 2 | 1.6 | 2.9% | 0.060 | 0.0 |
| 15_armageddon_lag20 | stair lag20 | 1.000 | 1.000 | 1.000 | 2 | 1.4 | 6.1% | 0.000 | 0.0 |
| 1_baseline | 30% bounce | 1.000 | 1.000 | 1.000 | 5 | 5.1 | 2.8% | 0.091 | 0.0 |
| 2_flash | 40% bounce | 1.000 | 1.000 | 1.000 | 6 | 4.4 | 23.5% | 0.000 | 0.0 |
| 3_slowbleed | 50%/500s | 1.000 | 1.000 | 1.000 | 7 | 7.0 | 10.7% | 0.048 | 0.0 |
| 4_noinsurance | 30% no ins | 1.000 | 1.000 | 1.000 | 5 | 5.1 | 2.8% | 0.091 | 0.0 |
| 5_tinylp | 30% small LP | 1.000 | 1.000 | 1.000 | 5 | 5.1 | 2.8% | 0.091 | 0.0 |
| 6_degens | 20% 2.5%IM | 1.000 | 1.000 | 1.000 | 12 | 10.4 | 17.4% | 0.021 | 0.0 |
| 7_skew90 | 30% 90%skew | 1.000 | 1.000 | 1.000 | 1 | 1.1 | 0.7% | 0.110 | 0.0 |
| 8_staircase | 3x15% steps | 1.000 | 1.000 | 1.000 | 6 | 5.8 | 7.8% | 0.065 | 0.0 |
| 9_oracle | 20% distort | 1.000 | 1.000 | 1.000 | 4 | 1.0 | 19.6% | 0.000 | 0.0 |
| adl_a_decay | 70% 2%IM | 1.000 | 0.979 | 1.000 | 4 | 2.7 | 9.1% | 0.000 | 0.0 |
| adl_cascade | 70% batch | 1.000 | 0.833 | 1.000 | 2 | 1.1 | 4.7% | 0.000 | 0.0 |
| adl_drain_reset | 80% 1.5%IM | 1.000 | 0.911 | 1.000 | 1 | 0.4 | 1.7% | 0.000 | 0.0 |
| adl_k_deficit | 50% K-path | 1.000 | 0.876 | 1.000 | 5 | 2.5 | 13.9% | 0.000 | 0.0 |
| adl_stale | 60% bounce | 1.000 | 0.844 | 1.000 | 2 | 1.0 | 4.6% | 0.000 | 0.0 |
| adl_trigger | 60% no ins | 1.000 | 0.844 | 1.000 | 2 | 1.0 | 4.6% | 0.000 | 0.0 |
| adversarial_adl_cascade | 70% advers | 1.000 | 0.292 | 1.000 | 74 | 2.9 | 4.9% | 0.000 | 0.0 |
| adversarial_adl_cascade_honest | 70% honest | 1.000 | 0.101 | 1.000 | 74 | 2.9 | 4.9% | 0.000 | 0.0 |
| adversarial_keeper | 50% advers | 1.000 | 0.886 | 1.000 | 192 | 7.0 | 13.7% | 0.000 | 0.0 |
| adversarial_keeper_honest | 50% honest | 1.000 | 0.561 | 1.000 | 192 | 7.0 | 13.7% | 0.000 | 0.0 |
| dust_gc | 40% dust/GC | 1.000 | 1.000 | 1.000 | 222 | 13.7 | 13.3% | 0.016 | 0.0 |
| funding_dynamics | 20% rate flip | 1.000 | 0.365 | 1.000 | 237 | 1.1 | 49.9% | 0.000 | 0.0 |
| ten10_alt | Alt 80% | 1.000 | 1.000 | 1.000 | 5 | 4.5 | 8.9% | 0.000 | 0.0 |
| ten10_btc | BTC 14% | 1.000 | 1.000 | 1.000 | 1 | 0.9 | 0.2% | 0.128 | 0.0 |
| ten10_hl | BTC 14% noI | 1.000 | 1.000 | 1.000 | 2 | 1.6 | 0.8% | 0.113 | 0.0 |
| ten10_sol | SOL 40% | 1.000 | 1.000 | 1.000 | 3 | 3.1 | 5.9% | 0.020 | 0.0 |

## Focused Tests

| Test | Result |
|---|---|
| --test=audit | Fee asymmetry FIXED (both a and b charged), risk-reducing exemption works, ADL pipeline OI balanced |
| --test=adl_fairness | Capital loss ratio exactly proportional to position size (1:2:4) |
| --test=zombie_haircut | All users same h on exit, overhang clears to h=1.0, market clean |
| --test=adl_saturation | 62 liquidations, A decays, K within i128, no overflow, solvency pass |
| --test=adl_fuzz (500 seeds) | 0.00%% fairness error, 0 solvency failures, K uses 0.000000%% of i128 |

## Key Properties Verified

1. **Solvency**: vault >= c_tot + insurance in every crank of every run
2. **A/K fairness**: deficit-ordered keeper gives 0.00%% fairness error across 500 fuzz seeds
3. **No overhang**: after ADL winds down positions and warmup converts PnL, pnl_pos_tot returns to 0, h returns to 1.0
4. **No overflow**: K uses <0.000001%% of i128 capacity even under max liquidation stress
5. **Protected principal**: flat accounts preserve capital through any crash
6. **OI balance**: OI_eff_long == OI_eff_short maintained after every crank, trade, and ADL event

## A/K vs Hyperliquid ADL (10/10 crash comparison)

| | Hyperliquid (actual 10/10) | Percolator A/K (simulation) |
|---|---|---|
| Mechanism | Queue-based ADL (sequential) | A/K proportional scaling (O(1)) |
| Profitable traders force-closed | $650M (28x excess over minimal bad debt) | 0 -- positions survive |
| Who absorbs deficit | Selected profitable shorts (queue order) | All opposing accounts equally (A-multiplier) |
| Deficit per unit of position | Varies by queue position | Identical for all (0.00%% error) |
| Insurance used | $188M (Binance), exhausted on HL | Not needed (h=1.0 via warmup gating) |
| Market recovery | Did not reclaim pre-crash levels | final_h = 1.0 (self-healing) |

## Fairness Properties

**A/K fairness is exact for open-position economics.** When a bankrupt account is liquidated, the A-multiplier reduces all opposing positions by the same ratio, and the K-index distributes quote deficit proportionally to position size.

**H fairness is exact only for the currently stored realized claim set**, not for the economically true claim set you would get after globally cranking everyone. This is an unavoidable compromise due to limitations in smart contracts.

## Kani Formal Verification

146 proofs across 6 files. Key proofs added during this audit:

| Proof | What it verifies | Time |
|---|---|---|
| proof_v1126_flat_close_uses_eq_maint_raw | Flat exit rejects negative Eq_maint_raw (v11.26 fix) | 19s |
| proof_v1126_risk_reducing_fee_neutral | Fee-neutral buffer comparison (v11.26 fix) | 18s |
| proof_v1126_min_nonzero_margin_floor | Margin floors prevent microscopic position evasion | 4.6s |
| proof_buffer_masking_blocked | Slippage extraction via MM_req drop blocked | 45s |
| proof_adl_pipeline_trade_liquidate_reopen | Full ADL pipeline: trade->liq->ADL->reset->reopen | 64s |
| proof_risk_reducing_exemption_path | I256 buffer comparison for below-MM trades | 29s |
| proof_fee_debt_sweep_consumes_released_pnl | Fee debt swept from matured PnL when capital=0 | 1.2s |
| proof_gc_reclaims_flat_dust_capital | Dust capital swept to insurance on GC | 1.7s |
| proof_junior_profit_backing | h * matured_pnl <= residual (both h branches) | 1.4s |
| bounded_haircut_ratio_bounded | h_num <= h_den, h < 1 branch exercised | 0.4s |
