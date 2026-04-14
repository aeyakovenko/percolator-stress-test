#!/bin/bash
set -e

SCENARIOS="ten10_btc ten10_sol ten10_alt ten10_hl adl_trigger adl_a_decay adl_k_deficit adl_drain_reset adl_cascade adl_stale adversarial_keeper adversarial_adl_cascade funding_dynamics funding_extreme funding_crash_combo oracle_wick oracle_wick_adl dust_gc"

BASE=stress_out_warmup

for scenario in $SCENARIOS; do
    OUT="$BASE/$scenario"
    mkdir -p "$OUT"
    echo "=== $scenario ==="
    cargo run --release -- --scenario=$scenario --warmup=true --out=$OUT --runs=200 2>/dev/null
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(f'  min_h:  mean={d[\"min_h_mean\"]:.4f}  p01={d[\"min_h_p01\"]:.4f}  p50={d[\"min_h_p50\"]:.4f}')
print(f'  final_h: mean={d[\"final_h_mean\"]:.4f}  p50={d[\"final_h_p50\"]:.4f}')
print(f'  h_zero_frac={d[\"h_zero_frac\"]:.2f}  deficit_frac={d[\"deficit_frac\"]:.2f}  drain_only={d[\"drain_only_frac\"]:.2f}')
print(f'  max_h_lock: p99={d[\"max_h_lock_p99\"]:.0f}  max={d[\"max_h_lock_max\"]:.0f}')
print(f'  ins_ema_ratio: p01={d[\"min_ins_ema_ratio_p01\"]:.4f}  p50={d[\"min_ins_ema_ratio_p50\"]:.4f}')
print(f'  liqs: mean={d[\"liq_mean\"]:.0f}  p99={d[\"liq_p99\"]:.0f}')
" "$OUT/default/summary.json"
    echo
done

echo "=== ALL DONE ==="
