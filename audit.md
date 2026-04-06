# Percolator Stress Test Audit Results

**Spec:** v11.26  **Engine:** production (MAX_ACCOUNTS=4096)
**Scenarios:** 31 (31 with liquidations)  **Runs:** 6200  **Solvency violations:** 0

## Summary Table

| Scenario | Crash | Liqs | ADL A | Liq%% | true_h p01 | Cap p01 | deficit |
|---|---|---|---|---|---|---|---|
| 10_whale | 30% whale | 793 | 50.0 | 3.8%% | 1.000 | 0.090 | 0.0 |
| 11_funding | 30% fund | 792 | 50.0 | 3.8%% | 1.000 | 0.089 | 0.0 |
| 12_armageddon | 50% no bnce | 1574 | 54.0 | 17.6%% | 1.000 | 0.038 | 0.0 |
| 13_skew_lag5 | 30% lag5 | 1425 | 10.0 | 18.1%% | 1.000 | 0.039 | 0.0 |
| 14_armageddon_lag5 | stair lag5 | 1567 | 33.0 | 20.7%% | 1.000 | 0.037 | 0.0 |
| 15_armageddon_lag20 | stair lag20 | 1567 | 9.0 | 68.8%% | 0.990 | 0.000 | 0.0 |
| 1_baseline | 30% bounce | 792 | 50.0 | 3.8%% | 1.000 | 0.089 | 0.0 |
| 2_flash | 40% bounce | 869 | 9.0 | 44.3%% | 1.000 | 0.000 | 0.0 |
| 3_slowbleed | 50%/500s | 914 | 309.4 | 14.2%% | 1.000 | 0.047 | 0.0 |
| 4_noinsurance | 30% no ins | 792 | 50.0 | 3.8%% | 1.000 | 0.089 | 0.0 |
| 5_tinylp | 30% sm LP | 1 | 0.0 | 0.0%% | 1.000 | 0.995 | 0.0 |
| 6_degens | 20% 2.5%IM | 585 | 55.8 | 50.0%% | 1.000 | 0.017 | 0.0 |
| 7_skew90 | 30% 90%skew | 1425 | 50.0 | 3.8%% | 1.000 | 0.090 | 0.0 |
| 8_staircase | 3x15% | 860 | 52.8 | 11.3%% | 1.000 | 0.063 | 0.0 |
| 9_oracle | 20% distort | 0 | 0.0 | 0.0%% | 1.000 | 0.995 | 0.0 |
| adl_a_decay | 70% 2%IM | 1724 | 15.0 | 93.9%% | 0.852 | 0.000 | 0.0 |
| adl_cascade | 70% batch | 1096 | 3.0 | 95.7%% | 0.334 | 0.000 | 0.0 |
| adl_drain_reset | 80% 1.5%IM | 447 | 1.0 | 91.2%% | 0.118 | 0.000 | 0.0 |
| adl_k_deficit | 50% K | 1544 | 5.0 | 90.0%% | 0.710 | 0.000 | 0.0 |
| adl_stale | 60% bnce | 1638 | 2.0 | 93.9%% | 0.309 | 0.000 | 0.0 |
| adl_trigger | 60% no ins | 1638 | 2.0 | 93.9%% | 0.309 | 0.000 | 0.0 |
| adversarial_adl_cascade | 70% adv | 1096 | 3.0 | 95.7%% | 0.397 | 0.000 | 0.0 |
| adversarial_adl_cascade_honest | 70% honest | 1096 | 3.0 | 95.7%% | 0.449 | 0.000 | 0.0 |
| adversarial_keeper | 50% adv | 1486 | 7.0 | 89.4%% | 0.781 | 0.000 | 0.0 |
| adversarial_keeper_honest | 50% honest | 1486 | 7.0 | 89.4%% | 1.000 | 0.000 | 0.0 |
| dust_gc | 40% GC | 1131 | 13.9 | 37.7%% | 1.000 | 0.007 | 0.0 |
| funding_dynamics | 20% rate | 297 | 5.0 | 51.2%% | 1.000 | 0.000 | 0.0 |
| ten10_alt | Alt 80% | 1743 | 6.0 | 96.8%% | 0.595 | 0.000 | 0.0 |
| ten10_btc | BTC 14% | 728 | 13.0 | 41.1%% | 1.000 | 0.000 | 0.0 |
| ten10_hl | BTC noI | 714 | 8.0 | 39.1%% | 1.000 | 0.000 | 0.0 |
| ten10_sol | SOL 40% | 1538 | 14.0 | 85.1%% | 1.000 | 0.000 | 0.0 |

## 10/10 Crash Simulation (Oct 10, 2025)

BTC \$122K -> \$105K (14%%), 87%% long bias, 50-66x leverage, crank lag 3-5 slots.

| | Hyperliquid (actual) | Percolator A/K (simulation) |
|---|---|---|
| Mechanism | Queue ADL (sequential) | A/K proportional (O(1)) |
| Force-closed profitable traders | \$650M (28x excess) | 0 - positions scale equally via A |
| Liquidations | ~\$10B | 714/run (39.1% of users) |
| ADL events | Queue overtriggered | 8.0 A-reductions/run |
| Vault deficit | Unknown bad debt | 0.0 (zero across 200 runs) |
| Recovery | Did not reclaim | final_h = 1.0000 |

## Focused Tests

| Test | Result |
|---|---|
| adl_fairness | Loss exactly proportional to position size (1:2:4) |
| zombie_haircut | Uniform h, overhang clears to 1.0, market clean |
| adl_saturation | 4094 shorts bankrupt, K within i128, no overflow |
| adl_fuzz (100 seeds) | 0.00%% fairness error, 492 max liqs/seed, 0 solvency failures |
| audit | Bilateral fee, risk-reducing exemption, full ADL pipeline all pass |

## Key Properties

1. **Solvency**: vault >= c_tot + insurance in every crank of every run
2. **A/K fairness**: 0.00%% error with deficit-ordered keeper
3. **No overhang**: pnl_pos_tot returns to 0 after ADL wind-down
4. **No overflow**: K uses <0.000001%% of i128
5. **Protected principal**: flat accounts keep deposits
6. **OI balance**: OI_eff_long == OI_eff_short maintained throughout
