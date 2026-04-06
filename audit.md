# Percolator Stress Test Audit Results

**Spec version:** v11.26
**Engine:** force_close_resolved + _not_atomic API
**Total scenarios:** 31
**Runs per scenario:** 200
**Total simulations:** 6200
**Solvency violations:** 0
**Deficit fraction:** 0.0 across all runs

## Summary Table

| Scenario | Crash | min_h | true_h p01 | final_h | Liqs | ADL A | Liq%% | Cap p01 | deficit |
|---|---|---|---|---|---|---|---|---|---|
| 10_whale | 30% whale | 1.000 | 1.000 | 1.000 | 1 | 0.0 | 0.0%% | 1.264 | 0.0 |
| 11_funding | 30% fund | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.359 | 0.0 |
| 12_armageddon | 50% no bnce | 1.000 | 1.000 | 1.000 | 1 | 0.0 | 0.0%% | 1.277 | 0.0 |
| 13_skew_lag5 | 30% lag5 | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.387 | 0.0 |
| 14_armageddon_lag5 | stair lag5 | 1.000 | 1.000 | 1.000 | 1 | 0.0 | 0.0%% | 1.284 | 0.0 |
| 15_armageddon_lag20 | stair lag20 | 1.000 | 1.000 | 1.000 | 1 | 0.0 | 0.0%% | 1.311 | 0.0 |
| 1_baseline | 30% bounce | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.359 | 0.0 |
| 2_flash | 40% bounce | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.324 | 0.0 |
| 3_slowbleed | 50%/500s | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.737 | 0.0 |
| 4_noinsurance | 30% no ins | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.359 | 0.0 |
| 5_tinylp | 30% sm LP | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.359 | 0.0 |
| 6_degens | 20% 2.5%IM | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.245 | 0.0 |
| 7_skew90 | 30% 90%skew | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.387 | 0.0 |
| 8_staircase | 3x15% | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.568 | 0.0 |
| 9_oracle | 20% distort | 1.000 | 1.000 | 1.000 | 6 | 0.9 | 67.5%% | 0.000 | 0.0 |
| adl_a_decay | 70% 2%IM | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 2.298 | 0.0 |
| adl_cascade | 70% batch | 1.000 | 1.000 | 1.000 | 1 | 0.0 | 0.0%% | 1.533 | 0.0 |
| adl_drain_reset | 80% 1.5%IM | 1.000 | 1.000 | 1.000 | 1 | 0.0 | 0.0%% | 1.661 | 0.0 |
| adl_k_deficit | 50% K | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.896 | 0.0 |
| adl_stale | 60% bnce | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.864 | 0.0 |
| adl_trigger | 60% no ins | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.955 | 0.0 |
| adversarial_adl_cascade | 70% adv | 1.000 | 0.726 | 1.000 | 0 | 0.0 | 0.0%% | 1.658 | 0.0 |
| adversarial_adl_cascade_honest | 70% honest | 1.000 | 0.906 | 1.000 | 0 | 0.0 | 0.0%% | 1.658 | 0.0 |
| adversarial_keeper | 50% adv | 1.000 | 0.997 | 1.000 | 0 | 0.0 | 0.0%% | 1.547 | 0.0 |
| adversarial_keeper_honest | 50% honest | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.547 | 0.0 |
| dust_gc | 40% GC | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.372 | 0.0 |
| funding_dynamics | 20% rate | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.136 | 0.0 |
| ten10_alt | Alt 80% | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.904 | 0.0 |
| ten10_btc | BTC 14% | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.274 | 0.0 |
| ten10_hl | BTC noI | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.286 | 0.0 |
| ten10_sol | SOL 40% | 1.000 | 1.000 | 1.000 | 0 | 0.0 | 0.0%% | 1.498 | 0.0 |

## Focused Tests

| Test | Result |
|---|---|
| --test=audit | Fee asymmetry FIXED, risk-reducing exemption works, ADL pipeline pass |
| --test=adl_fairness | Capital loss exactly proportional to position size (1:2:4) |
| --test=zombie_haircut | All users same h, overhang clears to h=1.0, market clean |
| --test=adl_saturation | Max liquidations, K within i128, no overflow, solvency pass |
| --test=adl_fuzz (100 seeds) | 0.00%% fairness error, 0 solvency failures |

## Key Properties Verified

1. **Solvency**: vault >= c_tot + insurance in every crank of every run
2. **A/K fairness**: deficit-ordered keeper gives 0.00%% fairness error
3. **No overhang**: pnl_pos_tot returns to 0, h returns to 1.0 after ADL
4. **No overflow**: K uses <0.000001%% of i128 capacity
5. **Protected principal**: flat accounts preserve capital through any crash
6. **OI balance**: OI_eff_long == OI_eff_short maintained throughout

## A/K vs Hyperliquid ADL (10/10 crash)

| | Hyperliquid (actual) | Percolator A/K |
|---|---|---|
| Mechanism | Queue ADL (sequential) | A/K proportional (O(1)) |
| Force-closed profitable traders | $650M (28x excess) | 0 |
| Deficit distribution | Varies by queue position | Identical per unit |
| Insurance | Exhausted | Not needed (h=1.0) |
| Recovery | Did not reclaim | final_h = 1.0 |

## Fairness Properties

**A/K fairness is exact for open-position economics.** The A-multiplier reduces all
opposing positions by the same ratio; K distributes deficit proportionally to position size.

**H fairness is exact only for the currently stored realized claim set**, not the
economically true set after globally cranking everyone. Unavoidable smart contract compromise.
