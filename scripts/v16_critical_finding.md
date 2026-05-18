# v16 — critical: oracle pump + convert + partial withdraw extracts LP capital

**Date:** 2026-05-18
**Engine:** `v16.8.0 Realizable Full Shared Cross-Margin` (engine commit `a37eb5f`)
**Reproducible via:** `cargo run --release -- --test=v16_instant` (probe `[A2-instant]` and `[A2-default]`)

## Summary

An attacker with $1,000 cap extracts $4,904 of real USDC from the LP by:

1. Opening a long position on asset 0 at oracle truth ($200)
2. Pumping the asset 0 oracle to $400 over ~150 slots at the engine's max permitted rate (45 bps/slot)
3. Calling `convert_released_pnl_to_capital_not_atomic` — succeeds, $5,000 of inflated PnL moves into capital
4. Calling `withdraw_not_atomic($5,449)` — succeeds because the engine's IM check sees post-withdraw cap ($550) ≥ IM req ($500)
5. Oracle reverts to truth ($200); the open position takes a $5,000 loss; cap drains to $0; bankruptcy residual $4,450 is booked against the LP-side source domain

The engine's `assert_public_invariants` reports `Ok` throughout. No invariant is violated. Every accounting step is internally consistent. But real value flows from LP to attacker.

## Reproduction output

```
[A2-instant] Drift-style oracle pump → convert → PARTIAL withdraw → revert  (h_min=0, h_max=1)
  convert returned Ok(5000)  cap=5999
  partial-withdraw $5449 → Ok("Ok"), actually $5449
  after revert: attacker_cap=$455 pnl=$0 | LP_cap_Δ=$-5000 | NET to attacker: $4904 ★★ EXTRACTION ★★
  engine invariants: Ok(())

[A2-default] Drift-style oracle pump → convert → PARTIAL withdraw → revert  (h_min=0, h_max=30)
  convert returned Ok(5000)  cap=5999
  partial-withdraw $5449 → Ok("Ok"), actually $5449
  after revert: attacker_cap=$455 pnl=$0 | LP_cap_Δ=$-5000 | NET to attacker: $4904 ★★ EXTRACTION ★★
  engine invariants: Ok(())
```

## Why the earlier `v16_drift` probe missed it

My original `v16_drift` probe attempted to withdraw the FULL converted cap ($5,999). The engine correctly rejected that because post-withdraw cap would be $0 (below IM). The probe stopped there with `Withdrawn: $0`.

The attack only works with a PARTIAL withdraw that leaves enough cap to satisfy IM. My probe didn't try that. With the partial withdraw, the engine permits the extraction.

## Why the engine permits it

In `convert_released_pnl_to_capital_not_atomic` (v16.rs:3367):

```
let converted = if account_has_source_claims(account) {
    global_support.min(account_source_realizable_support(account, released))
} else {
    global_support
};
```

When the LP has refreshed during the pump, the engine has reserved LP capital as backing for `(SOL, Short)` source domain (via `reserve_new_capital_backed_loss_for_source_domain_not_atomic`). `account_source_realizable_support` returns the full inflated PnL because backing exists at 100% credit rate.

Convert burns the face claim and **consumes** the backing — both are written down atomically. So at convert time the LP's USDC is committed irreversibly to the attacker, even though the underlying position is still open at an inflated price.

Then `withdraw_not_atomic` performs only the standard IM check:
```
post_withdraw_equity >= IM_requirement
```

At pump peak, IM_req = 5% × $5k notional = $250 (×2 for asset 0/1 if applicable). Post-withdraw equity = $550 cap. Check passes, withdrawal succeeds.

When the oracle returns to truth:
- Long position MTM = (200 − 400) × 25 = −$5,000
- settle drains all $550 of remaining cap
- Bankruptcy residual = $4,450, booked against the source domain
- Insurance / social loss absorbs it (in this case the LP)

## What v16 is missing

The convert path consumes counterparty backing for an unrealized gain. The gain comes from an oracle that can revert. The engine has no mechanism to:

1. **Lock the converted capital** to the position's continued solvency (e.g., reserved_pnl) — once converted, it's regular cap and can be withdrawn.
2. **Reverse the convert** when the position takes an offsetting loss — instead the loss drains cap normally and the LP eats the residual.
3. **Detect rapid sustained oracle moves** (the spec mentions `target_effective_lag`, but the per-slot max-move envelope is respected so no lag is detected).
4. **Cooldown convert/withdraw** after large MTM swings (the h-lock would do this if it triggered, but `bankruptcy_hlock_active` only flips on actual bankruptcy events, not on large favorable MTM).

## Severity

For a bounty-relevant deployment, this is a working capital-extraction primitive that scales with:
- Time to walk the oracle to a profitable extreme (~150 slots at 45 bps each — Solana slots are ~400ms, so ~60 seconds)
- LP capitalization (the attacker can extract up to LP's reserved backing capacity)
- The attacker's own deposit relative to position size (the higher the leverage, the larger the multiple)

At 5x leverage with $1k cap, the attacker extracted ~5x their deposit in a single attack cycle. At 20x leverage (which the spec envelope allows with smaller IM_bps), the multiplier scales accordingly.

## Mitigation ideas (not implemented in engine)

1. **Lock converted capital to position lifetime:** when `convert_released_pnl_to_capital` runs, add the converted amount to `reserved_pnl`-equivalent on `capital`, preventing withdrawal until the position is closed at an actual settled price.
2. **Defer realization until close:** disallow converting unrealized PnL on open positions; require close-trade or rebalance-reduce to flow PnL into the realized bucket first.
3. **Rate-limit convert by recent oracle drift:** if the asset's effective_price has moved by more than X bps over the last N slots, gate `convert_released_pnl_to_capital` for that source domain.
4. **Oracle-revert reversal of convert:** track converted amounts attributed to specific source-domain MTM, and reverse them if the source domain's MTM drops below the converted face.

Option (1) is the simplest and matches HL's withdrawal-lock semantics. Option (2) is the most conservative and matches v14/v15's safer (but less capital-efficient) behavior.

## Engine commits in scope

- `a37eb5f Update v16 source-credit spec`
- `b1ee2a1 Verify v16 source credit engine`
- Previously: rewiring of `apply_haircut_bounded_close_loss_to_pnl` and `apply_signed_kf_delta_to_pnl` to consult source-credit, plus `reserve_new_capital_backed_loss_for_source_domain_not_atomic` in `settle_leg_kf_effects`

## Reproduction

```
cargo run --release -- --test=v16_instant
```

Look for the `[A2-*]` lines showing `NET to attacker: $4904 ★★ EXTRACTION ★★`.
