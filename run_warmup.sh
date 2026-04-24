#!/bin/bash
set -e

SCENARIOS="ten10_btc ten10_sol ten10_alt ten10_hl adl_trigger adl_a_decay adl_k_deficit adl_drain_reset adl_cascade adl_stale adversarial_keeper adversarial_adl_cascade funding_dynamics funding_extreme funding_crash_combo oracle_wick oracle_wick_adl dust_gc"

BASE=stress_out_warmup

for scenario in $SCENARIOS; do
    OUT="$BASE/$scenario"
    mkdir -p "$OUT"
    echo "=== $scenario ==="
    cargo run --release -- --scenario=$scenario --out=$OUT --runs=200 2>/dev/null > /dev/null
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(f'  min_h:   mean={d[\"min_h_mean\"]:.4f}  p01={d[\"min_h_p01\"]:.4f}  p50={d[\"min_h_p50\"]:.4f}')
print(f'  final_h: mean={d[\"final_h_mean\"]:.4f}  p50={d[\"final_h_p50\"]:.4f}')
print(f'  h_zero_frac={d[\"h_zero_frac\"]:.2f}  deficit_frac={d[\"deficit_frac\"]:.2f}  drain_only={d[\"drain_only_frac\"]:.2f}')
print(f'  residual-scarcity stress: mean={d[\"stress_slots_mean\"]:.1f}  p99={d[\"stress_slots_p99\"]:.0f}  entered={d[\"stress_entered_frac\"]:.0%}')
print(f'  consumption stress:       mean={d[\"consumption_stress_slots_mean\"]:.1f}  p99={d[\"consumption_stress_slots_p99\"]:.0f}  entered={d[\"consumption_stress_entered_frac\"]:.0%}')
print(f'  peak consumption p99:     {d[\"max_consumption_bps_p99\"]:.0f} bps')
print(f'  sweep generations:        p50={d[\"sweep_generations_p50\"]:.0f}  p99={d[\"sweep_generations_p99\"]:.0f}')
print(f'  matured overshoot events: {d[\"matured_overshoot_total\"]}  (must be 0)')
print(f'  min_headroom: p01=\${d[\"min_headroom_p01\"]/1e6:.0f}  p50=\${d[\"min_headroom_p50\"]/1e6:.0f}')
print(f'  liqs: mean={d[\"liq_mean\"]:.0f}  p99={d[\"liq_p99\"]:.0f}')
" "$OUT/default/summary.json"
    echo
done

echo "=== ALL DONE ==="
