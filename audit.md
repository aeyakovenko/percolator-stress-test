# Percolator Stress Test Audit Results

**Spec:** v11.26  **Engine:** production (MAX_ACCOUNTS=4096)
**Scenarios:** 31 (all with liquidations)  **Runs:** 6200  **Solvency violations:** 0

## Summary Table

| Scenario | Liqs | ADL A | Liq%% | true_h p01 | Ins End | deficit |
|---|---|---|---|---|---|---|
| 10_whale | 793 | 50.0 | 1.9%% | 1.000 | $11.1M | 0.0 |
| 11_funding | 792 | 50.0 | 1.9%% | 1.000 | $10.9M | 0.0 |
| 12_armageddon | 1626 | 54.0 | 15.8%% | 1.000 | $1.6M | 0.0 |
| 13_skew_lag5 | 1425 | 10.0 | 16.3%% | 1.000 | $11.3M | 0.0 |
| 14_armageddon_lag5 | 1618 | 33.0 | 18.6%% | 1.000 | $1.6M | 0.0 |
| 15_armageddon_lag20 | 1618 | 9.0 | 61.9%% | 0.991 | $235K | 0.0 |
| 1_baseline | 792 | 50.0 | 1.9%% | 1.000 | $10.9M | 0.0 |
| 2_flash | 869 | 9.0 | 22.2%% | 1.000 | $10.9M | 0.0 |
| 3_slowbleed | 914 | 309.4 | 7.1%% | 1.000 | $10.9M | 0.0 |
| 4_noinsurance | 792 | 50.0 | 1.9%% | 1.000 | $865K | 0.0 |
| 5_tinylp | 788 | 50.1 | 1.9%% | 1.000 | $10.9M | 0.0 |
| 6_degens | 585 | 55.8 | 25.0%% | 1.000 | $11.2M | 0.0 |
| 7_skew90 | 1425 | 50.0 | 3.4%% | 1.000 | $11.3M | 0.0 |
| 8_staircase | 860 | 52.8 | 5.6%% | 1.000 | $10.9M | 0.0 |
| 9_oracle | 687 | 1.0 | 30.9%% | 0.975 | $3.8M | 0.0 |
| adl_a_decay | 1724 | 15.0 | 84.6%% | 0.859 | $0 | 0.0 |
| adl_cascade | 1339 | 3.0 | 88.9%% | 0.354 | $0 | 0.0 |
| adl_drain_reset | 447 | 1.0 | 89.4%% | 0.121 | $0 | 0.0 |
| adl_k_deficit | 1544 | 5.0 | 76.5%% | 0.727 | $0 | 0.0 |
| adl_stale | 1705 | 2.0 | 88.9%% | 0.323 | $0 | 0.0 |
| adl_trigger | 1705 | 2.0 | 88.9%% | 0.323 | $0 | 0.0 |
| adversarial_adl_cascade | 1339 | 3.0 | 88.9%% | 0.410 | $0 | 0.0 |
| adversarial_adl_cascade_honest | 1339 | 3.0 | 88.9%% | 0.464 | $0 | 0.0 |
| adversarial_keeper | 1534 | 7.0 | 75.9%% | 0.801 | $0 | 0.0 |
| adversarial_keeper_honest | 1534 | 7.0 | 75.9%% | 0.968 | $0 | 0.0 |
| dust_gc | 1215 | 13.9 | 26.4%% | 1.000 | $2.7M | 0.0 |
| funding_dynamics | 297 | 5.0 | 25.6%% | 1.000 | $5.0M | 0.0 |
| ten10_alt | 1743 | 6.0 | 87.1%% | 0.609 | $1 | 0.0 |
| ten10_btc | 728 | 13.0 | 35.8%% | 1.000 | $21.2M | 0.0 |
| ten10_hl | 714 | 8.0 | 34.0%% | 0.996 | $0 | 0.0 |
| ten10_sol | 1538 | 14.0 | 76.6%% | 1.000 | $7.8M | 0.0 |

## Loss Absorption Waterfall

```
Bankrupt liquidation deficit D
  1. Insurance absorbs min(D, balance - floor)
  2. D_rem > 0: K_opp shifts (deficit to opposing open positions)
  3. A_opp shrinks (position quantity reduction)
  4. No opposing positions: uninsured loss (residual drops)
  5. H gates withdrawal: h = min(Residual, PNL_matured) / PNL_matured
     where Residual = Vault - C_tot - Insurance
```

Insurance is senior to A/K. H is read-only (never modifies insurance).

## What is verified vs not verified

**Verified (real):**
- Solvency: vault >= c_tot + insurance after every crank (sound math)
- Liquidations fire: 728-1743/run across scenarios
- ADL fires: A-reductions in 29/31 scenarios
- Insurance absorbs deficit before A/K (visible in table)
- ADL fairness: 0.00%% error with deficit-ordered keeper (fuzz test)
- No i128 overflow: K uses <0.000001%% of range

**Not verified by Monte Carlo (covered by Kani proofs only):**
- Non-zero funding rate dynamics (funding_schedule was dead code)
- Adversarial oracle manipulation (flash wicks, front-running)
- Cross-market correlation (single-market model only)
- Per-account fairness in Monte Carlo (only checked in fuzz test)
