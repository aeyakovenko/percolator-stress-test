#!/bin/bash
set -e
B=/home/anatoly/percolator-stress/target/release/stress_test
S="--slippage=50"
D=stress_out_slippage50

echo "=== 1_baseline ==="
$B $S --out=$D/1_baseline 2>&1 | head -3

echo "=== 2_flash ==="
$B $S --crash_pct=4000 --crash_len=10 --bounce_pct=3000 --bounce_len=20 --total_slots=200 --out=$D/2_flash 2>&1 | head -3

echo "=== 3_slowbleed ==="
$B $S --crash_pct=5000 --crash_len=500 --bounce_pct=0 --bounce_len=1 --total_slots=600 --out=$D/3_slowbleed 2>&1 | head -3

echo "=== 4_noinsurance ==="
$B $S --insurance=0 --crash_pct=3000 --out=$D/4_noinsurance 2>&1 | head -3

echo "=== 5_tinylp ==="
$B $S --lp_capital=5000000 --crash_pct=3000 --out=$D/5_tinylp 2>&1 | head -3

echo "=== 6_degens ==="
$B $S --im_bps=250 --mm_bps=125 --crash_pct=2000 --out=$D/6_degens 2>&1 | head -3

echo "=== 7_skew90 ==="
$B $S --long_bias=0.9 --crash_pct=3000 --out=$D/7_skew90 2>&1 | head -3

echo "=== 8_staircase ==="
$B $S --price_path=staircase --crash_pct=1500 --crash_len=20 --staircase_steps=3 --staircase_flat=30 --total_slots=400 --out=$D/8_staircase 2>&1 | head -3

echo "=== 9_oracle ==="
$B $S --price_path=oracle_distortion --distortion_pct=2000 --distortion_start=30 --distortion_len=5 --total_slots=200 --out=$D/9_oracle 2>&1 | head -3

echo "=== 10_whale ==="
$B $S --whale=true --whale_capital=25000000 --whale_leverage=10 --crash_pct=3000 --out=$D/10_whale 2>&1 | head -3

echo "=== 11_funding ==="
$B $S --funding_rate=10 --crash_pct=3000 --out=$D/11_funding 2>&1 | head -3

echo "=== 12_armageddon ==="
$B $S --long_bias=0.9 --whale=true --whale_capital=25000000 --whale_leverage=10 --insurance=0 --crash_pct=5000 --out=$D/12_armageddon 2>&1 | head -3

echo "=== 13_skew_lag5 ==="
$B $S --long_bias=0.9 --crank_interval=5 --out=$D/13_skew_lag5 2>&1 | head -3

echo "=== 14_armageddon_lag5 ==="
$B $S --long_bias=0.9 --price_path=staircase --crash_pct=2000 --staircase_steps=3 --staircase_flat=20 --bounce_pct=0 --whale=true --whale_capital=25000000 --whale_leverage=10.0 --funding_rate=10 --insurance=0 --crank_interval=5 --out=$D/14_armageddon_lag5 2>&1 | head -3

echo "=== 15_armageddon_lag20 ==="
$B $S --long_bias=0.9 --price_path=staircase --crash_pct=2000 --staircase_steps=3 --staircase_flat=20 --bounce_pct=0 --whale=true --whale_capital=25000000 --whale_leverage=10.0 --funding_rate=10 --insurance=0 --crank_interval=20 --out=$D/15_armageddon_lag20 2>&1 | head -3

echo "=== ALL DONE ==="
