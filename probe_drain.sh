#!/bin/bash
# Probe insurance depletion paths.

run_probe() {
    local label="$1"
    shift
    local out="/tmp/drain/${label}"
    mkdir -p "$out"
    echo "=== $label ==="
    cargo run --release -- --runs=30 --out=$out "$@" 2>/dev/null > /dev/null
    python3 -c "
import json
d = json.load(open('$out/default/summary.json'))
# Insurance is in ATOMIC units (1 USDC = 1e6 atomic). Convert to USDC dollars.
ins_mean_usdc = d['insurance_end_mean'] / 1e6
ins_p10_usdc = d['insurance_end_p10'] / 1e6
res_p01_usdc = d['min_residual_p01'] / 1e6
print(f'  liqs:                 mean={d[\"liq_mean\"]:.0f}  p99={d[\"liq_p99\"]:.0f}')
print(f'  users_liq:            mean={d[\"users_liq_frac_mean\"]:.1%}')
print(f'  insurance_end:        mean=\${ins_mean_usdc:,.0f}  p10=\${ins_p10_usdc:,.0f}')
print(f'  deficit_frac:         {d[\"deficit_frac\"]:.0%}  drain_only={d[\"drain_only_frac\"]:.0%}  h_zero_frac={d[\"h_zero_frac\"]:.0%}')
print(f'  min_h_p01:            {d[\"min_h_p01\"]:.4f}  min_true_h_p01: {d[\"min_true_h_p01\"]:.4f}')
print(f'  min_residual_p01:     \${res_p01_usdc:,.0f}')
print(f'  matured_overshoot:    {d[\"matured_overshoot_total\"]} events  (gate enforced upper bound)')
"
    echo
}

mkdir -p /tmp/drain

# Most aggressive pathological constructions
run_probe "P1_max_zombies_zero_ins" --scenario=ten10_alt --insurance=0 --lp_capital=1000000 --n_zombies=500 --zombie_pnl=500000 --zombie_fee_debt=10000

# Long-running funding drain with no cushion, no LP
run_probe "P2_funding_no_cushion" --scenario=funding_extreme --insurance=0 --lp_capital=1000000 --total_slots=1500

# Cascade with insurance ZERO and high turnover
run_probe "P3_cascade_no_insurance" --scenario=adl_drain_reset --insurance=0 --lp_capital=2000000

# Whale + crash + zero insurance
run_probe "P4_whale_drain" --scenario=ten10_alt --insurance=0 --whale=true --whale_capital=20000000 --whale_leverage=10 --lp_capital=2000000

# Extreme: everything at boundary
run_probe "P5_perfect_storm" --scenario=ten10_alt --insurance=0 --lp_capital=500000 --n_zombies=500 --zombie_pnl=1000000 --zombie_fee_debt=20000 --total_slots=400

echo "=== ALL DONE ==="
