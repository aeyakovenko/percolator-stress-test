#!/bin/bash
set -e
B=/home/anatoly/percolator-stress/target/release/stress_test

echo "=== adl_trigger ==="
$B --scenario=adl_trigger --out=stress_out/adl_trigger 2>&1 | head -3

echo "=== adl_a_decay ==="
$B --scenario=adl_a_decay --out=stress_out/adl_a_decay 2>&1 | head -3

echo "=== adl_k_deficit ==="
$B --scenario=adl_k_deficit --out=stress_out/adl_k_deficit 2>&1 | head -3

echo "=== adl_drain_reset ==="
$B --scenario=adl_drain_reset --out=stress_out/adl_drain_reset 2>&1 | head -3

echo "=== adl_stale ==="
$B --scenario=adl_stale --out=stress_out/adl_stale 2>&1 | head -3

echo "=== adl_cascade ==="
$B --scenario=adl_cascade --out=stress_out/adl_cascade 2>&1 | head -3

echo "=== ALL ADL DONE ==="
