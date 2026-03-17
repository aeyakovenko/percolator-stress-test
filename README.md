# Percolator Stress Test

Monte Carlo stress simulator for the Percolator risk engine.

## Fairness Properties

**A/K fairness (ADL) is exact for open-position economics.**
When a bankrupt account is liquidated and the deficit is socialized via `enqueue_adl`, the A-multiplier reduces all opposing positions by the same ratio, and the K-index distributes quote deficit proportionally to position size. Every account on the receiving side absorbs the same loss per unit of notional.

**H fairness (haircut ratio) is exact only for the currently stored realized claim set**, not for the economically "true" claim set you would get after globally cranking everyone. This is an unavoidable compromise due to limitations in smart contracts — a global sweep of all accounts would be required to compute the fully settled h, but on-chain execution cannot touch every account atomically. The lazy settlement model (A/K side indices) ensures that when accounts are eventually touched, they settle at the correct historical rate.

## Usage

```
cargo build --release
./target/release/stress_test [OPTIONS]
```

### Scenarios

```bash
# Default (30% crash, 2000 users, 200 runs)
./target/release/stress_test

# ADL-specific scenarios
./target/release/stress_test --scenario=adl_trigger
./target/release/stress_test --scenario=adl_a_decay
./target/release/stress_test --scenario=adl_k_deficit
./target/release/stress_test --scenario=adl_drain_reset
./target/release/stress_test --scenario=adl_stale
./target/release/stress_test --scenario=adl_cascade

# Focused verification tests
./target/release/stress_test --test=zombie_haircut
./target/release/stress_test --test=adl_fairness
```

### Run all scenarios

```bash
bash run_all.sh          # 15 standard scenarios
bash run_all_adl.sh      # 6 ADL scenarios
```
