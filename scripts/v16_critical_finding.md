# v16 — partial fix verified, new corruption state on retry path

**Date:** 2026-05-18
**Engine:** v16 `4f1ff20 Lock source-backed converted capital` (on top of `a37eb5f`)
**Stress branch:** `v16`
**Reproducible via:** `cargo run --release -- --test=v16_instant`

## TL;DR

The new commit `4f1ff20` adds `source_converted_capital_lock[V16_DOMAIN_COUNT]` and gates `withdraw_not_atomic` so the converted capital cannot be withdrawn while the source domain has active exposure. This **prevents the original LP→attacker cash extraction** but leaves the account in a corrupt state where `capital < lock`, with the attacker holding $4,973 of LP capital that they **cannot withdraw** and the LP cannot **recover**.

Net economic flow under the attack with the new fix:

```
LP_cap_Δ:              -$5000
Attacker cash out:     $0           ← fix works for immediate extraction
Attacker account cap:  $4973        ← phantom locked capital, unrecoverable
Engine top-level invariants:  Ok    ← per-account inconsistency invisible to global check
```

The attacker doesn't profit, but the LP still loses $5000 — a permanent denial-of-funds rather than a theft.

## The probe output

```
[A2-instant] h_min=0, h_max=1:
  convert returned Ok(5000)  cap=5999
  partial-withdraw $5449 → Err("LockActive"), actually $0          ← fix blocks this
  after revert (before close): cap=$4973 pnl=$0
  after close+convert+withdraw: cap=$4973 pnl=$0 | withdraw2=Err("InvalidLeg")
  debug: source_converted_capital_lock sum = $5000 | active legs = [0]
  engine invariants: Ok(())                                         ← top-level passes
  after second refresh: lock sum = $5000 (refresh=Err("InvalidLeg")) ← per-account fails
  retry withdraw $4973: Err("InvalidLeg")
```

The same outcome under `[A2-default]` with `h_max=30`.

## Sequence of events

1. Attacker deposits $1,000, opens long $5k SOL at oracle $200 (5x, IM=$500).
2. Pumps oracle $200 → $400 over ~150 slots at the engine's max-permitted 45 bps/slot.
3. `convert_released_pnl_to_capital_not_atomic`: **Ok(5000)**.
   - cap: $999 → $5999
   - `source_converted_capital_lock[(SOL, Short)]`: 0 → $5000
   - LP's reserved backing for `(SOL, Short)`: consumed
4. `withdraw_not_atomic($5449)`: **Err(LockActive)** (the new fix).
5. Oracle reverts $400 → $200.
   - Settlement loop drains cap as the long position takes MTM loss
   - Settle (`settle_negative_pnl_from_principal`) ignores the lock and pulls from `capital` directly: `paid = capital.min(loss)`
   - When cap drops, the account state has `lock > capital` — the fix's invariant is violated, but `settle` doesn't enforce or re-balance it
6. Liquidation triggers (since the leg's MTM at the lower oracle exceeds remaining unlocked cap)
   - Leg remains `active = true` (clearing is presumably pending in the engine's state)
   - The lock is NOT released (`release_inactive_source_converted_capital_locks` only fires when the source domain has *no active exposure*)
7. Subsequent operations fail with `InvalidLeg`:
   - `full_account_refresh`: validates account shape → `source_converted_capital_lock > capital` → `Err(InvalidLeg)`
   - `withdraw_not_atomic`: same validation → `Err(InvalidLeg)`
   - `convert_released_pnl_to_capital_not_atomic`: same

The attacker now has $4,973 of capital they cannot move, the LP has lost $5,000, and the engine considers itself "Ok" at the global level because no top-level invariant tracks per-account lock-vs-cap.

## Why settle doesn't respect the lock

`settle_negative_pnl_from_principal` (v16.rs:3338):

```
let paid = account.capital.min(loss);  // <-- drains full cap, ignores lock
account.capital -= paid;
```

It treats the locked amount as just capital. When the oracle reverts and the position loses, that loss has to settle somewhere; settle pulls from `capital` regardless of which portion is locked.

`validate_source_converted_capital_locks` (v16.rs:3112) then says:

```
if locked > account.capital {
    return Err(V16Error::InvalidLeg);
}
```

This is only called from `validate_account_shape`, which runs at the start of mutations but doesn't re-balance the account when it would otherwise produce an inconsistent state. So the state is allowed to drift into `lock > capital`, after which all favorable actions on the account are blocked.

## What "extraction" means under this fix

The original attack flow ($4,904 net to attacker) is blocked — the attacker doesn't get LP funds in their pocket. But:

- LP capital change: **−$5,000** (real, irreversible)
- Attacker account cap: **$4,973** (stuck, irrecoverable)
- $27 disappears to fees and rounding

So no net theft, but no symmetric reversal either. The LP eats a $5,000 loss and the attacker eats a $1,000 deposit loss. The net result is **$6,000 destroyed**, $4,973 of which sits as phantom capital with broken validity.

This is "deny extraction at the cost of denying recovery." Whether that's acceptable depends on the deployment's risk tolerance — but it's not the clean defense the spec implies (§0.6 "Protected principal is senior" suggests LP capital should not be at risk from oracle manipulation within the per-slot envelope).

## What a complete fix needs

The half-fix locks **withdrawal** of the converted amount but doesn't tie that amount to the **position's continued solvency**. A complete fix needs one of:

1. **Settle respects the lock**: when settle would drain cap below the source-locked amount, the unpaid portion routes to bankruptcy residual immediately rather than waiting. The lock then unwinds atomically with the residual booking.
2. **Convert reserves against a reverse swing**: convert can only succeed if the account's remaining unlocked cap covers the worst-case MTM reversion to entry price (or some bounded envelope). This is closer to v14/v15 behavior — less capital efficient but provably safe.
3. **Defer realization until close**: don't allow convert on open positions; require the position to be closed (cleanly or via liquidation) before PnL realizes into cap. The close trade locks in the price at execution time, not the prevailing oracle.

Option (1) is the smallest delta and preserves convert's spec semantics. Option (3) is closest to a battle-tested approach.

## Coverage in other probes

The other probes in the v16 suite still pass clean:

- `v16_backing_fuzz`: 200,000 transitions, 0 invariant fails
- `v16_extract`: 500 adversarial seeds, max single-seed extraction $0
- `v16_extras`: 5 invariants × 1000 seeds, 0 fails
- `v16_buckets`: per-domain isolation holds
- Capital efficiency, ratchet, spread-realize: all clean under both h_max=1 and h_max=30

The extraction path requires DIRECTED oracle manipulation (pump → convert → withdraw) which random-walk fuzzes don't hit. The `v16_instant` probe ([A2-*]) is the targeted scenario.

## Reproduction

```
cargo run --release -- --test=v16_instant
```

Look at `[A2-instant]` / `[A2-default]` for the post-fix behavior.
