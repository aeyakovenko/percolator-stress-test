# Percolator Stress Test Audit Results

**Spec version:** v11.26
**Total scenarios:** 31 (30 with liquidations)
**Runs per scenario:** 200
**Total simulations:** 6200
**Solvency violations:** 0
**Deficit fraction:** 0.0 across all runs

## Summary Table

| Scenario | Crash | min_h | true_h p01 | final_h | Liqs | ADL A | Liq%% | Cap p01 | deficit |
|---|---|---|---|---|---|---|---|---|---|
| 10_whale | 30% whale | 1.000 | 1.000 | 1.000 | 22 | 16.4 | 3.2%% | 0.090 | 0.0 |
| 11_funding | 30% fund | 1.000 | 1.000 | 1.000 | 20 | 15.8 | 3.4%% | 0.090 | 0.0 |
| 12_armageddon | 50% no bnce | 1.000 | 1.000 | 1.000 | 44 | 24.4 | 16.0%% | 0.039 | 0.0 |
| 13_skew_lag5 | 30% lag5 | 1.000 | 1.000 | 1.000 | 37 | 9.1 | 16.7%% | 0.039 | 0.0 |
| 14_armageddon_lag5 | stair lag5 | 1.000 | 1.000 | 1.000 | 43 | 17.6 | 19.0%% | 0.038 | 0.0 |
| 15_armageddon_lag20 | stair lag20 | 1.000 | 1.000 | 1.000 | 43 | 7.8 | 63.7%% | 0.000 | 0.0 |
| 1_baseline | 30% bounce | 1.000 | 1.000 | 1.000 | 20 | 15.8 | 3.4%% | 0.090 | 0.0 |
| 2_flash | 40% bounce | 1.000 | 1.000 | 1.000 | 23 | 7.3 | 42.6%% | 0.000 | 0.0 |
| 3_slowbleed | 50%/500s | 1.000 | 1.000 | 1.000 | 24 | 22.5 | 13.7%% | 0.047 | 0.0 |
| 4_noinsurance | 30% no ins | 1.000 | 1.000 | 1.000 | 20 | 15.8 | 3.4%% | 0.090 | 0.0 |
| 5_tinylp | 30% sm LP | 1.000 | 1.000 | 1.000 | 20 | 15.8 | 3.4%% | 0.090 | 0.0 |
| 6_degens | 20% 2.5%IM | 1.000 | 1.000 | 1.000 | 15 | 13.1 | 47.3%% | 0.017 | 0.0 |
| 7_skew90 | 30% 90%skew | 1.000 | 1.000 | 1.000 | 37 | 24.0 | 3.4%% | 0.090 | 0.0 |
| 8_staircase | 3x15% | 1.000 | 1.000 | 1.000 | 22 | 16.2 | 10.9%% | 0.064 | 0.0 |
| 9_oracle | 20% distort | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 0.995 | 0.0 |
| adl_a_decay | 70% 2%IM | 1.000 | 0.780 | 1.000 | 45 | 11.5 | 89.0%% | 0.000 | 0.0 |
| adl_cascade | 70% batch | 1.000 | 0.203 | 1.000 | 47 | 3.0 | 90.6%% | 0.000 | 0.0 |
| adl_drain_reset | 80% 1.5%IM | 1.000 | 0.068 | 1.000 | 35 | 1.0 | 89.6%% | 0.000 | 0.0 |
| adl_k_deficit | 50% K | 1.000 | 0.582 | 1.000 | 40 | 4.9 | 85.1%% | 0.000 | 0.0 |
| adl_stale | 60% bnce | 1.000 | 0.191 | 1.000 | 46 | 2.0 | 89.0%% | 0.000 | 0.0 |
| adl_trigger | 60% no ins | 1.000 | 0.191 | 1.000 | 46 | 2.0 | 89.0%% | 0.000 | 0.0 |
| adversarial_adl_cascade | 70% adv | 1.000 | 0.203 | 1.000 | 47 | 3.0 | 90.6%% | 0.000 | 0.0 |
| adversarial_adl_cascade_honest | 70% honest | 1.000 | 1.000 | 1.000 | 47 | 3.0 | 90.6%% | 0.000 | 0.0 |
| adversarial_keeper | 50% adv | 1.000 | 0.646 | 1.000 | 40 | 6.5 | 84.8%% | 0.000 | 0.0 |
| adversarial_keeper_honest | 50% honest | 1.000 | 1.000 | 1.000 | 40 | 6.5 | 84.8%% | 0.000 | 0.0 |
| dust_gc | 40% GC | 1.000 | 1.000 | 1.000 | 32 | 10.3 | 36.3%% | 0.007 | 0.0 |
| funding_dynamics | 20% rate | 1.000 | 1.000 | 1.000 | 14 | 4.3 | 48.4%% | 0.000 | 0.0 |
| ten10_alt | Alt 80% | 1.000 | 1.000 | 1.000 | 45 | 5.3 | 91.7%% | 0.000 | 0.0 |
| ten10_btc | BTC 14% | 1.000 | 1.000 | 1.000 | 19 | 9.6 | 38.6%% | 0.000 | 0.0 |
| ten10_hl | BTC noI | 1.000 | 1.000 | 1.000 | 18 | 6.6 | 36.6%% | 0.000 | 0.0 |
| ten10_sol | SOL 40% | 1.000 | 1.000 | 1.000 | 40 | 11.9 | 80.3%% | 0.000 | 0.0 |

## 10/10 Crash: Percolator A/K vs Hyperliquid Queue ADL

| | Hyperliquid (actual) | Percolator A/K (sim) |
|---|---|---|
| Mechanism | Queue ADL (sequential) | A/K proportional (O(1)) |
| Force-closed profitable traders | $650M (28x excess) | 0 — positions scale equally via A |
| Liquidations | ~$10B | 18/run (36.6% of users) |
| ADL events | Queue overtriggered | 6.6 A-reductions/run |
| Deficit | Unknown bad debt | 0.0 (zero) |
| Recovery | Did not reclaim | final_h = 1.0000 |

## Focused Tests

| Test | Result |
|---|---|
| --test=audit | Bilateral fee, risk-reducing exemption, ADL pipeline all pass |
| --test=adl_fairness | Loss exactly proportional to position size (1:2:4) |
| --test=zombie_haircut | Uniform h, overhang clears to 1.0, market clean |
| --test=adl_saturation | Max liq stress, K within i128, no overflow |
| --test=adl_fuzz (100 seeds) | 0.00%% fairness error, 0 solvency failures |

## Key Properties

1. **Solvency**: vault >= c_tot + insurance in every crank of every run
2. **A/K fairness**: 0.00%% error with deficit-ordered keeper (500 fuzz seeds)
3. **No overhang**: pnl_pos_tot returns to 0 after ADL wind-down
4. **No overflow**: K uses <0.000001%% of i128 capacity
5. **Protected principal**: flat accounts preserve capital
6. **OI balance**: OI_eff_long == OI_eff_short maintained throughout
