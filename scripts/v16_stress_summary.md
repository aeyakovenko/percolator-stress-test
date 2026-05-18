# v16 stress sweep — full results

**Date:** 2026-05-18
**Engine:** `v16.8.0 Realizable Full Shared Cross-Margin`
**Stress branch:** `v16` (commit `8d96d36` + this commit)
**Probes run:** 31 / 31 pass

## Build & probe summary

All 31 probes ported cleanly via mechanical V14→V16 rename + V16Config
constructor delegation. All probes execute on v16 without errors.

## Key findings

### 1. Spec promise NOT yet delivered

The `v16_pre_settle` probe injects full counterparty backing (credit_rate=100%)
BEFORE settlement. The user's spread trade still loses $500 of $1000 capital.

**Root cause:** `apply_signed_kf_delta_to_pnl` (v16.rs:7334) and
`apply_haircut_bounded_close_loss_to_pnl` (v16.rs:7269) — the K-pair
settlement path — still use only `self.residual()` and
`self.junior_claim_bound()` (v15 gating). They never consult the new
`source_credit[domain].credit_rate_num`.

The new `account_haircut_equity` (v16.rs:7211) correctly takes
`MIN(global_support, source_realizable)`, but this runs AFTER per-leg
settlement has already drained capital.

### 2. No auto-orchestration of BackingReservationPlan

`full_account_refresh` calls `reconcile_account_source_credit_liens_not_atomic`,
but that function only handles expiry/impairment — it never creates new
backing reservations. The spec's "A full account refresh computes ... a
deterministic BackingReservationPlan" is not implemented.

`SourceCreditStateV16` slots remain `EMPTY` after refreshing an account
with losses — backing must be reserved manually by the wrapper via
`add_fresh_counterparty_backing_not_atomic`.

### 3. Healthy-market behavior matches v15

| Scenario | v15 result | v16 result |
|---|---|---|
| Spread profit, residual=0 | Stuck paper PnL, $0 cash recovery | **Same** — `spread_realize` log unchanged |
| Long SOL + short ETH 30-day walk | 2.6% survival | **2.0% survival** |
| Single 5x leverage 30-day walk | 44.6% survival | **38.2% survival** |
| Long SOL + long ETH 10x portfolio | 0.8% survival | **0.4% survival** |
| Diversification (4 split vs 1) | -11% vs -21% | **-11% vs -21%** |
| h_min/h_max ratchet test | identical across configs | **identical across configs** |
| Per-domain bankruptcy attribution | SOL bankruptcy isolated | **SOL bankruptcy isolated** |
| Round-trip spread (loss feeds residual) | Nets to $0 | **Same — `spread_residual` log unchanged** |

## Capital efficiency probes (key data)

### Single-asset random walk (500 seeds, ~30 day window, 30 bps daily vol)

```
leverage | survival% | avg pnl %
---------|-----------|----------
  2x     | 100.0%    | -21.79%
  5x     |  38.2%    | -21.75%
 10x     |   0.0%    | (liquidated)
 15x     |   0.0%    | (liquidated)
```

### Diversification (same $10k notional split across N assets, $5k cap)

```
config            | survival% | avg net pnl %
------------------|-----------|--------------
concentrated (1)  | 100.0%    | -21.03%
split 2 ways      | 100.0%    | -15.81%
split 3 ways      | 100.0%    | -12.97%
split 4 ways      | 100.0%    | -11.44%
```

### Spread trade (long SOL + short ETH, $5k each, $1k cap)

```
config                                       | survival% | avg P&L%
---------------------------------------------|-----------|----------
long SOL only ($5k notional, 5x)             |   39.0%   | -23.67%
long SOL + long ETH ($5k each, 10x portfolio)|    0.4%   | -22.89%
long SOL + short ETH (RELATIVE VALUE spread) |    2.0%   | -12.70%
```

The "hedged" spread liquidates 98% of the time — same as the unhedged 10x
portfolio. No realized cross-margin benefit in healthy markets.

### Mean-reversion ratchet (oracle round-trips)

```
±2.0%, ±5.0%, ±10.0%, ±15.0% round trips on 2x long position:
total_lost = $0 (0.0%) in every case
```

When the LP refreshes between accruals, the ratchet does NOT bite. (The
asymmetric capital absorption is bounded by residual, which is reset on
round-trip moves.)

### h-lock comparison (no effect)

```
instant     (h_min=0, h_max=1)  : cap=$860 pnl=$135  total_lost=$4 (0.5%)
default     (h_min=0, h_max=30) : cap=$860 pnl=$135  total_lost=$4 (0.5%)
warmup      (h_min=5, h_max=30) : cap=$860 pnl=$135  total_lost=$4 (0.5%)
```

Identical results across all h-lock configurations — confirms the ratchet
is residual-bounded, not h-lock bounded.

## Security probes (all pass)

| Probe | Status |
|---|---|
| exec_price_attack | engine defended (LockActive at 9999bps deviation) |
| sybil_close       | $0 attacker extraction across 100/1000/5000/9999 bps |
| drift             | bad-asset isolated; no contagion to healthy asset |
| hard / hard_ext   | 2000-seed fuzz: all 2000 withdraws limited to deposited cap, no insurance drain |
| domain_attr       | bankruptcy residual charged only to source domain (SOL spent=$0 when budget=$0) |
| corner_cases      | adversarial keeper: no extraction across 20 liquidations |
| multileg          | 4-leg user crash: insurance_used=$0, residual=$0, explicit_loss=$0 |
| f6                | conservative-pause wrapper-controlled (returns LockActive when stress=true) |
| boundary / config | configuration envelopes validated; rebalance path works |
| resolve           | market resolve + emergency exit path works |
| pnl_trace         | per-leg PnL materialization correct; liquidation outcomes valid |

## Conclusion

The v16 implementation is **structurally complete** (all data structures,
primitives, and proofs present) but the **K-pair settlement path is not
yet wired to the new source-credit gating mechanism**. In healthy markets
where `residual = 0`, v16 currently behaves identically to v15 — the
"Realizable Full Shared Cross-Margin" UX from the spec is structurally
unreachable until two changes land in the engine:

1. `apply_signed_kf_delta_to_pnl` and `apply_haircut_bounded_close_loss_to_pnl`
   consult `source_credit[domain].credit_rate_num` rather than (or in addition
   to) `residual / junior_claim_bound`.
2. `full_account_refresh` computes a `BackingReservationPlan` and calls
   `add_fresh_counterparty_backing_not_atomic` for every domain where
   the account owes loss — so backing exists by the time the next account
   tries to claim against it.

Without these, the spread-trade survival in normal markets remains <3%
and gain legs are paper-only.
