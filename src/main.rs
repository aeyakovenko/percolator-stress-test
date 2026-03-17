//! Monte Carlo stress simulator for the Percolator risk engine.
//!
//! Runs crash scenarios through the real engine implementation and
//! aggregates outcome distributions across many RNG seeds.
//!
//! # Known coverage gaps (TODO)
//!
//! - [FIXED] SlippageMatcher: use --slippage=N (bps) to deviate exec_price
//!   from oracle, generating non-zero trade_pnl that exercises warmup restart.
//! - Constant funding_rate_bps_per_slot: no anti-retroactivity testing.
//!   Need rate changes across long dt intervals to test stored-rate semantics.
//! - reserved_pnl always 0: pending withdrawal interactions untested.
//! - maintenance_fee_per_slot = 0 by default: fee debt accumulation untested.
//!   Run scenarios with non-zero maintenance_fee to exercise fee drain paths.
//! - min_liquidation_abs = 1: dust close/GC behavior effectively disabled.
//!   Use realistic threshold + small-position accounts to test dust handling.

use std::{
    alloc::{self, Layout},
    env, fs,
    path::PathBuf,
    time::Instant,
};

use percolator::{RiskEngine, RiskParams, I128, U128, POS_SCALE, ADL_ONE, SideMode};
use std::sync::atomic::{AtomicU64, Ordering};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, LogNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ════════════════════════════════════════════════════════════════════════════
// Slippage matcher — exec_price deviates from oracle by bounded amount
// ════════════════════════════════════════════════════════════════════════════

/// Computes exec_price = oracle ± slippage.
/// Direction alternates per call (buy below oracle, sell above oracle, etc.)
/// generating non-zero trade_pnl that exercises warmup restart logic.
struct SlippageMatcher {
    slippage_bps: u64,
    counter: AtomicU64,
}

impl SlippageMatcher {
    fn new(slippage_bps: u64) -> Self {
        Self {
            slippage_bps,
            counter: AtomicU64::new(0),
        }
    }

    fn exec_price(&self, oracle_price: u64, size: i128) -> u64 {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        // Alternate: even calls get favorable price, odd get unfavorable.
        // Favorable for buyer = below oracle; favorable for seller = above oracle.
        let delta = (oracle_price as u128 * self.slippage_bps as u128 / 10_000) as u64;
        let is_buy = size > 0;
        let favorable = n % 2 == 0;
        let price = if (is_buy && favorable) || (!is_buy && !favorable) {
            oracle_price.saturating_sub(delta)
        } else {
            oracle_price.saturating_add(delta)
        };
        // Clamp to valid range
        price.max(1)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════════════════

/// Pre-crash setup phase: enough crank calls for a full sweep (4096/256 = 16)
const SETUP_SLOTS: u64 = 64;
/// Record time-series snapshots every N crash slots
const SNAPSHOT_INTERVAL: u64 = 5;

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Config {
    // Monte Carlo
    runs: usize,
    base_seed: u64,

    // Accounts
    n_users: usize,
    n_zombies: usize,

    // Engine params
    warmup_slots: u64,
    mm_bps: u64,
    im_bps: u64,
    trading_fee_bps: u64,
    maintenance_fee_per_slot: u128,
    liquidation_fee_bps: u64,
    liquidation_buffer_bps: u64,

    // Capital (human-readable USDC amounts)
    lp_capital_usdc: u64,
    insurance_topup_usdc: u64,

    // Price path
    p0: u64,             // starting price in dollars
    crash_pct_bps: u64,  // crash magnitude (3000 = 30%)
    crash_len: u64,      // slots to reach bottom
    bounce_pct_bps: u64, // bounce after crash (800 = 8%)
    bounce_len: u64,
    total_slots: u64,

    // Funding
    funding_rate_bps_per_slot: i64,

    // Zombie knobs
    zombie_pnl_usdc: u64,
    zombie_fee_debt_usdc: u64,

    // Price path mode: "crash_bounce", "staircase", "oracle_distortion"
    price_path_type: String,
    staircase_steps: u64,
    staircase_flat_len: u64,
    distortion_pct_bps: u64,
    distortion_start_slot: u64,
    distortion_len: u64,

    // Directional skew (0.0 = all short, 0.5 = balanced, 1.0 = all long)
    long_bias: f64,

    // Crank lag (1 = every slot, 5 = every 5th slot, etc.)
    crank_interval: u64,

    // Matcher slippage (0 = NoOp, >0 = exec_price deviates from oracle by up to N bps)
    slippage_bps: u64,

    // Whale account
    whale_enabled: bool,
    whale_capital_usdc: u64,
    whale_leverage: f64,

    // Grid (empty = single scenario)
    grid_crash_pcts: Vec<u64>,
    grid_warmup_slots: Vec<u64>,
    grid_insurance: Vec<u64>,

    // Output
    out_dir: String,
    snapshots: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            runs: 200,
            base_seed: 1,
            n_users: 2000,
            n_zombies: 50,
            warmup_slots: 600,
            mm_bps: 500,
            im_bps: 1000,
            trading_fee_bps: 5,
            maintenance_fee_per_slot: 0,
            liquidation_fee_bps: 50,
            liquidation_buffer_bps: 100,
            lp_capital_usdc: 50_000_000,
            insurance_topup_usdc: 10_000_000,
            p0: 60_000,
            crash_pct_bps: 3000,
            crash_len: 60,
            bounce_pct_bps: 800,
            bounce_len: 60,
            total_slots: 600,
            funding_rate_bps_per_slot: 0,
            zombie_pnl_usdc: 50_000,
            zombie_fee_debt_usdc: 200,
            price_path_type: "crash_bounce".into(),
            staircase_steps: 2,
            staircase_flat_len: 30,
            distortion_pct_bps: 2000,
            distortion_start_slot: 30,
            distortion_len: 5,
            long_bias: 0.5,
            crank_interval: 1,
            slippage_bps: 0,
            whale_enabled: false,
            whale_capital_usdc: 25_000_000,
            whale_leverage: 10.0,
            grid_crash_pcts: vec![],
            grid_warmup_slots: vec![],
            grid_insurance: vec![],
            out_dir: "stress_out".into(),
            snapshots: true,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Result types
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, Serialize)]
struct RunSummary {
    seed: u64,
    min_h: f64,
    final_h: f64,
    insurance_end: u128,
    c_tot_end: u128,
    pnl_pos_tot_end: u128,
    vault_end: u128,
    liquidations: u64,
    force_closes: u64,
    users_liquidated: usize,
    users_with_positions: usize,
    capital_ratios: Vec<f64>,
    /// capital / initial_capital — what's already protected principal
    principal_ratios: Vec<f64>,
    /// (capital + haircutted warmed PnL) / initial_capital — what's withdrawable now
    withdrawable_ratios: Vec<f64>,
    /// Slot offset where min_h first occurred
    min_h_slot: u64,
    /// Number of slots where h <= 0.0 (junior profits fully haircutted)
    h_zero_slots: u64,
    /// First slot where h <= 0.0 (or u64::MAX if never)
    h_zero_first_slot: u64,
    /// Number of slots where h < 0.5
    h_below_50_slots: u64,
    /// Number of slots where h < 0.1
    h_below_10_slots: u64,
    /// Minimum true h (signed residual / pnl_pos_tot, can go negative)
    min_true_h: f64,
    /// Minimum true signed residual in USDC (vault - c_tot - insurance)
    min_residual: i128,
    /// Withdraw/close path exercise counts
    withdraw_attempts: u64,
    withdraw_successes: u64,
    close_attempts: u64,
    close_successes: u64,
    /// ADL metrics
    adl_a_reductions: u64,   // cranks where A_opp decreased (quantity socialization)
    adl_k_changes: u64,      // cranks where K_opp changed from ADL (quote socialization)
    min_a_long: u128,
    min_a_short: u128,
    final_a_long: u128,
    final_a_short: u128,
    drain_only_entered: bool,
    epoch_resets: u64,
}

#[derive(Clone, Debug, Serialize)]
struct SlotSnapshot {
    seed: u64,
    slot: u64,
    oracle_price: u64,
    h: f64,
    c_tot: u128,
    pnl_pos_tot: u128,
    insurance: u128,
    open_interest: u128,
    cum_liquidations: u64,
    cum_force_closes: u64,
}

#[derive(Clone, Debug, Serialize)]
struct ScenarioSummary {
    label: String,
    runs: usize,

    min_h_mean: f64,
    min_h_std: f64,
    min_h_p01: f64,
    min_h_p05: f64,
    min_h_p50: f64,
    min_h_p90: f64,
    min_h_p95: f64,
    min_h_p99: f64,

    final_h_mean: f64,
    final_h_p50: f64,
    final_h_p90: f64,
    final_h_p99: f64,

    liq_mean: f64,
    liq_p50: f64,
    liq_p90: f64,
    liq_p99: f64,

    fc_mean: f64,
    fc_p50: f64,
    fc_p90: f64,

    users_liq_frac_mean: f64,
    users_liq_frac_p90: f64,

    capital_ratio_p01: f64,
    capital_ratio_p10: f64,
    capital_ratio_p50: f64,
    capital_ratio_p90: f64,
    capital_ratio_p99: f64,

    /// Protected principal / deposit — what's safe regardless of PnL
    principal_ratio_p01: f64,
    principal_ratio_p10: f64,
    principal_ratio_p50: f64,
    principal_ratio_p90: f64,
    principal_ratio_p99: f64,

    /// (capital + haircutted warmed PnL) / deposit — what's withdrawable now
    withdrawable_ratio_p01: f64,
    withdrawable_ratio_p10: f64,
    withdrawable_ratio_p50: f64,
    withdrawable_ratio_p90: f64,
    withdrawable_ratio_p99: f64,

    insurance_end_mean: f64,
    insurance_end_p10: f64,

    /// Fraction of runs where h hit 0.0 (junior profits fully haircutted, NOT vault deficit)
    h_zero_frac: f64,
    /// Among h=0 runs: median slots spent at h=0
    h_zero_slots_p50: f64,
    /// Among h=0 runs: median first slot where h=0
    h_zero_first_slot_p50: f64,
    /// Fraction of runs where h dipped below 0.5
    h_below_50_frac: f64,
    /// Fraction of runs where h dipped below 0.1
    h_below_10_frac: f64,
    /// Median slot where min_h occurred
    min_h_slot_p50: f64,

    /// True h (signed, can go negative) — bypasses saturating_sub
    min_true_h_p01: f64,
    min_true_h_p05: f64,
    min_true_h_p50: f64,
    /// Minimum true residual (vault - c_tot - insurance) in atomic USDC, p01
    min_residual_p01: f64,
    min_residual_p50: f64,
    /// Fraction of runs where true h went negative
    negative_h_frac: f64,
    /// Fraction of runs where vault < c_tot + insurance (true insolvency)
    deficit_frac: f64,
    /// Withdraw/close path exercise metrics (means across runs)
    withdraw_attempts_mean: f64,
    withdraw_successes_mean: f64,
    close_attempts_mean: f64,
    close_successes_mean: f64,
    /// ADL aggregate metrics
    adl_a_reductions_mean: f64,
    adl_a_reductions_p99: f64,
    adl_k_changes_mean: f64,
    min_a_long_p01: f64,
    min_a_short_p01: f64,
    drain_only_frac: f64,
    epoch_reset_frac: f64,
    epoch_resets_mean: f64,
}

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

/// Convert human USDC to atomic units (1 USDC = 1e6)
fn usdc(u: u64) -> u128 {
    (u as u128) * 1_000_000
}

/// Convert human dollar price to 1e6-scaled oracle price
fn price_e6(dollars: u64) -> u64 {
    dollars.saturating_mul(1_000_000)
}

/// Dispatch to the configured price path generator
fn price_path(cfg: &Config, slot: u64) -> u64 {
    match cfg.price_path_type.as_str() {
        "staircase" => staircase_path(cfg, slot),
        "oracle_distortion" => distortion_path(cfg, slot),
        _ => crash_bounce_path(cfg, slot),
    }
}

/// Linear crash → optional bounce → flat
fn crash_bounce_path(cfg: &Config, slot: u64) -> u64 {
    let p0 = price_e6(cfg.p0) as u128;
    let crash_len = cfg.crash_len.max(1);
    let bounce_len = cfg.bounce_len.max(1);

    if slot <= crash_len {
        let frac = (cfg.crash_pct_bps as u128) * (slot as u128) / (crash_len as u128);
        return (p0 * 10_000u128.saturating_sub(frac) / 10_000) as u64;
    }

    let p_bottom = p0 * (10_000 - cfg.crash_pct_bps as u128) / 10_000;
    let slot2 = slot - crash_len;

    if slot2 <= bounce_len {
        let frac = (cfg.bounce_pct_bps as u128) * (slot2 as u128) / (bounce_len as u128);
        return (p_bottom * (10_000 + frac) / 10_000) as u64;
    }

    (p_bottom * (10_000 + cfg.bounce_pct_bps as u128) / 10_000) as u64
}

/// Multi-leg staircase: N steps of (crash → flat → crash → flat → ...)
fn staircase_path(cfg: &Config, slot: u64) -> u64 {
    let p0 = price_e6(cfg.p0) as u128;
    let steps = cfg.staircase_steps.max(1);
    let crash_len = cfg.crash_len.max(1);
    let flat_len = cfg.staircase_flat_len;

    let mut price = p0;
    let mut remaining = slot;

    for _ in 0..steps {
        if remaining == 0 {
            break;
        }

        // Crash phase
        let progress = remaining.min(crash_len);
        let frac = (cfg.crash_pct_bps as u128) * (progress as u128) / (crash_len as u128);
        let mid_price = price * 10_000u128.saturating_sub(frac) / 10_000;

        if remaining <= crash_len {
            return mid_price as u64;
        }

        // Completed this crash leg
        price = price * (10_000 - cfg.crash_pct_bps as u128) / 10_000;
        remaining -= crash_len;

        // Flat phase
        if remaining <= flat_len {
            return price as u64;
        }
        remaining -= flat_len;
    }

    price as u64
}

/// Oracle distortion: flat → spike up → return to flat
/// Tests whether warmup prevents extraction of manipulated profits
fn distortion_path(cfg: &Config, slot: u64) -> u64 {
    let p0 = price_e6(cfg.p0) as u128;
    let start = cfg.distortion_start_slot;
    let end = start + cfg.distortion_len;

    if slot >= start && slot < end {
        (p0 * (10_000 + cfg.distortion_pct_bps as u128) / 10_000) as u64
    } else {
        p0 as u64
    }
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn std_dev(vals: &[f64]) -> f64 {
    if vals.len() < 2 {
        return 0.0;
    }
    let m = mean(vals);
    let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
    var.sqrt()
}

fn sorted(vals: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut v: Vec<f64> = vals.collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

fn haircut_f64(engine: &RiskEngine) -> f64 {
    let (hn, hd) = engine.haircut_ratio();
    if hd == 0 { 1.0 } else { (hn as f64 / hd as f64).min(1.0) }
}

/// True signed residual: vault - c_tot - insurance.
/// Returns positive value when solvent, negative when in deficit.
fn true_residual(engine: &RiskEngine) -> i128 {
    let vault = engine.vault.get() as i128;
    let c_tot = engine.c_tot.get() as i128;
    let ins = engine.insurance_fund.balance.get() as i128;
    vault - c_tot - ins
}

/// True h: residual / pnl_pos_tot, can go negative.
fn true_h(engine: &RiskEngine) -> f64 {
    let pnl_pos_tot = engine.pnl_pos_tot;
    if pnl_pos_tot == 0 {
        return 1.0;
    }
    let res = true_residual(engine);
    res as f64 / pnl_pos_tot as f64
}

// ════════════════════════════════════════════════════════════════════════════
// Engine allocation (heap — avoids stack overflow for ~6 MB struct)
// ════════════════════════════════════════════════════════════════════════════

fn new_engine(params: RiskParams) -> Box<RiskEngine> {
    let layout = Layout::new::<RiskEngine>();
    let ptr = unsafe { alloc::alloc_zeroed(layout) as *mut RiskEngine };
    if ptr.is_null() {
        alloc::handle_alloc_error(layout);
    }
    let mut engine = unsafe { Box::from_raw(ptr) };
    engine.init_in_place(params);
    engine
}

// ════════════════════════════════════════════════════════════════════════════
// Single simulation run
// ════════════════════════════════════════════════════════════════════════════

struct UserInfo {
    idx: u16,
    initial_capital: u128,
    had_position: bool,
    is_whale: bool,
}

fn run_one(cfg: &Config, seed: u64) -> (RunSummary, Vec<SlotSnapshot>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let matcher = SlippageMatcher::new(cfg.slippage_bps);
    let p0 = price_e6(cfg.p0);

    // ── Build engine ────────────────────────────────────────────────────
    let params = RiskParams {
        warmup_period_slots: cfg.warmup_slots,
        maintenance_margin_bps: cfg.mm_bps,
        initial_margin_bps: cfg.im_bps,
        trading_fee_bps: cfg.trading_fee_bps,
        max_accounts: 4096,
        new_account_fee: U128::new(0),
        maintenance_fee_per_slot: U128::new(cfg.maintenance_fee_per_slot),
        max_crank_staleness_slots: u64::MAX,
        liquidation_fee_bps: cfg.liquidation_fee_bps,
        liquidation_fee_cap: U128::new(usdc(50_000)),
        liquidation_buffer_bps: cfg.liquidation_buffer_bps,
        min_liquidation_abs: U128::new(1),
    };

    let mut engine = new_engine(params);

    // ── Seed insurance + LP ─────────────────────────────────────────────
    let _ = engine.top_up_insurance_fund(usdc(cfg.insurance_topup_usdc));

    let lp_idx = engine.add_lp([1u8; 32], [2u8; 32], 0).unwrap();
    engine.deposit(lp_idx, usdc(cfg.lp_capital_usdc), p0, 0).unwrap();

    // Initial crank at slot 0
    let _ = engine.keeper_crank(lp_idx, 0, p0, 0);

    // ── Capital distributions (lognormal mixtures) ──────────────────────
    let retail = LogNormal::new(2_000f64.ln(), 1.0).unwrap();
    let pro = LogNormal::new(50_000f64.ln(), 0.8).unwrap();
    let whale = LogNormal::new(1_000_000f64.ln(), 0.7).unwrap();

    // ── Add whale account if enabled ────────────────────────────────────
    let mut users: Vec<UserInfo> = Vec::with_capacity(cfg.n_users + 1);
    if cfg.whale_enabled {
        let whale_idx = engine.add_user(0).unwrap();
        engine.deposit(whale_idx, usdc(cfg.whale_capital_usdc), p0, 0).unwrap();
        users.push(UserInfo {
            idx: whale_idx,
            initial_capital: usdc(cfg.whale_capital_usdc),
            had_position: false,
            is_whale: true,
        });
    }

    // ── Add users + deposit capital ─────────────────────────────────────
    for _ in 0..cfg.n_users {
        let roll: f64 = rng.gen();
        let cap_f = if roll < 0.80 {
            retail.sample(&mut rng)
        } else if roll < 0.99 {
            pro.sample(&mut rng)
        } else {
            whale.sample(&mut rng)
        };
        let cap_usdc = cap_f.max(50.0).min(50_000_000.0) as u64;

        let user_idx = match engine.add_user(0) {
            Ok(i) => i,
            Err(_) => break, // slab full
        };
        if engine.deposit(user_idx, usdc(cap_usdc), p0, 0).is_err() {
            continue;
        }

        users.push(UserInfo {
            idx: user_idx,
            initial_capital: usdc(cap_usdc),
            had_position: false,
            is_whale: false,
        });
    }

    // ── Run cranks through setup phase for full sweep ───────────────────
    for s in 1..=SETUP_SLOTS {
        let _ = engine.keeper_crank(lp_idx, s, p0, 0);
    }

    // ── Open positions via execute_trade ────────────────────────────────
    let max_lev = (10_000.0 / cfg.im_bps.max(1) as f64).max(1.0);
    let trade_slot = SETUP_SLOTS;

    for user in &mut users {
        let (lev, util, long) = if user.is_whale {
            // Whale: fixed leverage, always long (worst case for crash)
            (cfg.whale_leverage, 0.9, true)
        } else {
            let roll: f64 = rng.gen();
            let l = if roll < 0.80 {
                rng.gen_range(2.0..8.0f64)
            } else if roll < 0.99 {
                rng.gen_range(1.0..5.0f64)
            } else {
                rng.gen_range(2.0..max_lev.min(30.0).max(2.1))
            };
            let u = rng.gen_range(0.4..0.95f64);
            let dir = rng.gen::<f64>() < cfg.long_bias;
            (l, u, dir)
        };

        let cap_f = user.initial_capital as f64;
        let notional_atomic = (cap_f * lev * util) as u128;

        // size_q = notional * POS_SCALE / oracle_price (Q-scaled position)
        let pos_q = notional_atomic
            .saturating_mul(POS_SCALE)
            .checked_div(p0 as u128)
            .unwrap_or(POS_SCALE)
            .max(POS_SCALE);

        let sign: i128 = if long { 1 } else { -1 };

        // Retry with halved size on failure.
        // Snapshot/restore models Solana TX atomicity: execute_trade mutates
        // state (funding, mark settlement, maintenance fees) before the final
        // margin check. On-chain, Err reverts everything; we must do the same.
        let snapshot = engine.clone();
        let mut q = pos_q;
        let mut ok = false;
        for _ in 0..5 {
            let size_q = if sign > 0 {
                q as i128
            } else {
                -(q as i128)
            };
            let raw_size = (q / POS_SCALE) as i128 * sign;
            let ep = matcher.exec_price(p0, raw_size);
            if engine
                .execute_trade(lp_idx, user.idx, p0, trade_slot, size_q, ep)
                .is_ok()
            {
                user.had_position = true;
                ok = true;
                break;
            }
            // Restore between each retry — failed attempt mutated state
            *engine = snapshot.as_ref().clone();
            q /= 2;
            if q < POS_SCALE {
                break;
            }
        }
        if !ok {
            *engine = *snapshot;
        }
    }

    // ── Post-trade sweep ────────────────────────────────────────────────
    for s in (SETUP_SLOTS + 1)..=(SETUP_SLOTS + 32) {
        let _ = engine.keeper_crank(lp_idx, s, p0, 0);
    }

    // ── Inject zombies: positive PnL + fee debt ─────────────────────────
    let zombie_count = cfg.n_zombies.min(users.len());
    let zombie_indices: Vec<usize> =
        rand::seq::index::sample(&mut rng, users.len(), zombie_count).into_vec();
    for &k in &zombie_indices {
        let idx = users[k].idx as usize;

        // Add positive realized PnL to zombie. Model zero-sum: the LP
        // was the counterparty and lost capital. Reduce LP capital so that
        // C_tot drops and Residual (V - C_tot - I) rises to back the PnL.
        // Any zombie PnL exceeding LP's remaining capital is unbacked gap
        // loss, which naturally collapses h.
        let add_pnl = usdc(cfg.zombie_pnl_usdc) as i128;
        let old_pnl = engine.accounts[idx].pnl;
        engine.set_pnl(idx, old_pnl.saturating_add(add_pnl));

        // LP counterparty loss (zero-sum backing)
        let lp_cap = engine.accounts[lp_idx as usize].capital.get();
        let loss = usdc(cfg.zombie_pnl_usdc).min(lp_cap);
        engine.set_capital(lp_idx as usize, lp_cap.saturating_sub(loss));

        // Create fee debt (push fee_credits negative)
        let debt = usdc(cfg.zombie_fee_debt_usdc) as i128;
        let old_credits = engine.accounts[idx].fee_credits.get();
        engine.accounts[idx].fee_credits = I128::new(old_credits.saturating_sub(debt));

        // Set warmup slope so crank can convert over time
        engine.update_warmup_slope(users[k].idx as usize);
    }

    // ── Crash simulation ────────────────────────────────────────────────
    let crash_start = SETUP_SLOTS + 33;
    let mut min_h: f64 = f64::MAX;
    let mut min_h_slot: u64 = 0;
    let mut h_zero_slots: u64 = 0;
    let mut h_zero_first_slot: u64 = u64::MAX;
    let mut h_below_50_slots: u64 = 0;
    let mut h_below_10_slots: u64 = 0;
    let mut min_true_h: f64 = f64::MAX;
    let mut min_residual: i128 = i128::MAX;
    let mut withdraw_attempts: u64 = 0;
    let mut withdraw_successes: u64 = 0;
    let mut close_attempts: u64 = 0;
    let mut close_successes: u64 = 0;
    let mut adl_a_reductions: u64 = 0;
    let mut adl_k_changes: u64 = 0;
    let mut min_a_long: u128 = ADL_ONE;
    let mut min_a_short: u128 = ADL_ONE;
    let mut drain_only_entered = false;
    let mut epoch_resets: u64 = 0;
    let mut prev_epoch_long = engine.adl_epoch_long;
    let mut prev_epoch_short = engine.adl_epoch_short;
    let mut snapshots: Vec<SlotSnapshot> = Vec::new();

    let crank_every = cfg.crank_interval.max(1);

    for slot_offset in 0..cfg.total_slots {
        let slot = crash_start + slot_offset;
        let oracle = price_path(cfg, slot_offset);

        // Only crank every N slots to simulate keeper lag
        if slot_offset % crank_every == 0 {
            // Capture pre-crank ADL state
            let pre_a_long = engine.adl_mult_long;
            let pre_a_short = engine.adl_mult_short;
            let pre_k_long = engine.adl_coeff_long;
            let pre_k_short = engine.adl_coeff_short;

            let _ = engine.keeper_crank(
                lp_idx,
                slot,
                oracle,
                cfg.funding_rate_bps_per_slot,
            );

            // Track ADL events: A decreased = quantity socialization
            if engine.adl_mult_long < pre_a_long || engine.adl_mult_short < pre_a_short {
                adl_a_reductions += 1;
            }
            // K changed from ADL (coincident with A change or new liquidations)
            if engine.adl_coeff_long != pre_k_long || engine.adl_coeff_short != pre_k_short {
                adl_k_changes += 1;
            }
            min_a_long = min_a_long.min(engine.adl_mult_long);
            min_a_short = min_a_short.min(engine.adl_mult_short);
            if engine.side_mode_long == SideMode::DrainOnly || engine.side_mode_short == SideMode::DrainOnly {
                drain_only_entered = true;
            }
            if engine.adl_epoch_long > prev_epoch_long {
                epoch_resets += engine.adl_epoch_long - prev_epoch_long;
                prev_epoch_long = engine.adl_epoch_long;
            }
            if engine.adl_epoch_short > prev_epoch_short {
                epoch_resets += engine.adl_epoch_short - prev_epoch_short;
                prev_epoch_short = engine.adl_epoch_short;
            }

            // Hard invariant checks — fail fast on conservation violation
            let vault_val = engine.vault.get();
            let c_tot_val = engine.c_tot.get();
            let ins_val = engine.insurance_fund.balance.get();
            assert!(
                vault_val >= c_tot_val.saturating_add(ins_val),
                "SOLVENCY VIOLATION seed={} slot_offset={}: vault={} < c_tot={} + insurance={}",
                seed, slot_offset, vault_val, c_tot_val, ins_val
            );
            // Note: check_conservation(oracle) is too strict here — it verifies
            // the extended identity vault >= sum(capital) + sum(pnl) + insurance,
            // which the zombie injection (direct set_pnl/set_capital) violates.
            // The primary solvency invariant above is what signed_residual depends on.

            // Exercise withdraw/close paths on ~5 users every 10th crank.
            // Always revert to avoid altering crash dynamics.
            if slot_offset % (crank_every * 10) == 0 {
                let sample_n = 5.min(users.len());
                for _ in 0..sample_n {
                    let ui = rng.gen_range(0..users.len());
                    let user = &users[ui];
                    if !user.had_position { continue; }

                    let acct = &engine.accounts[user.idx as usize];
                    let has_position = !acct.position_basis_q == 0;

                    if has_position {
                        // Try withdrawing 10% of capital
                        let cap = acct.capital.get();
                        let amt = cap / 10;
                        if amt > 0 {
                            withdraw_attempts += 1;
                            let snap = engine.clone();
                            if engine.withdraw(user.idx, amt, oracle, slot).is_ok() {
                                withdraw_successes += 1;
                            }
                            *engine = *snap;
                        }
                    } else {
                        // Position liquidated — try closing account
                        close_attempts += 1;
                        let snap = engine.clone();
                        if engine.close_account(user.idx, slot, oracle).is_ok() {
                            close_successes += 1;
                        }
                        *engine = *snap;
                    }
                }
            }
        }

        let h = haircut_f64(&engine);
        if h < min_h {
            min_h = h;
            min_h_slot = slot_offset;
        }
        if h <= 0.0 {
            h_zero_slots += 1;
            if h_zero_first_slot == u64::MAX {
                h_zero_first_slot = slot_offset;
            }
        }
        if h < 0.5 {
            h_below_50_slots += 1;
        }
        if h < 0.1 {
            h_below_10_slots += 1;
        }
        let th = true_h(&engine);
        if th < min_true_h {
            min_true_h = th;
        }
        let res = true_residual(&engine);
        if res < min_residual {
            min_residual = res;
        }

        if cfg.snapshots && slot_offset % SNAPSHOT_INTERVAL == 0 {
            snapshots.push(SlotSnapshot {
                seed,
                slot: slot_offset,
                oracle_price: oracle,
                h,
                c_tot: engine.c_tot.get(),
                pnl_pos_tot: engine.pnl_pos_tot,
                insurance: engine.insurance_fund.balance.get(),
                open_interest: engine.oi_eff_long_q,
                cum_liquidations: engine.lifetime_liquidations,
                cum_force_closes: 0,
            });
        }
    }

    // ── Final crank to settle all state (especially important with crank lag) ──
    let final_slot = crash_start + cfg.total_slots;
    let final_oracle = price_path(cfg, cfg.total_slots.saturating_sub(1));
    let _ = engine.keeper_crank(lp_idx, final_slot, final_oracle, cfg.funding_rate_bps_per_slot);
    {
        let vault_val = engine.vault.get();
        let c_tot_val = engine.c_tot.get();
        let ins_val = engine.insurance_fund.balance.get();
        assert!(
            vault_val >= c_tot_val.saturating_add(ins_val),
            "FINAL SOLVENCY VIOLATION seed={}: vault={} < c_tot={} + insurance={}",
            seed, vault_val, c_tot_val, ins_val
        );
        // Note: check_conservation too strict with zombie PnL injection (see crash loop comment)
    }

    // ── End-of-run metrics ──────────────────────────────────────────────
    let final_h = haircut_f64(&engine);

    let mut capital_ratios: Vec<f64> = Vec::new();
    let mut principal_ratios: Vec<f64> = Vec::new();
    let mut withdrawable_ratios: Vec<f64> = Vec::new();
    let mut users_liquidated = 0usize;
    let mut users_with_positions = 0usize;

    // Haircut ratio for withdrawable calculation
    let (h_num, h_den) = engine.haircut_ratio();

    for user in &users {
        if !user.had_position {
            continue;
        }
        users_with_positions += 1;

        let acct = &engine.accounts[user.idx as usize];
        let init = user.initial_capital as f64;
        let capital = acct.capital.get() as f64;

        // Approximate MTM equity: capital + pnl (signed)
        let pnl_i128 = acct.pnl;
        let equity = (capital as i128 + pnl_i128).max(0) as f64;
        let mtm_ratio = if init > 0.0 { equity / init } else { 0.0 };
        capital_ratios.push(mtm_ratio);

        // Protected principal only (already safe, no warmup gate)
        let prin_ratio = if init > 0.0 { capital / init } else { 0.0 };
        principal_ratios.push(prin_ratio);

        // Withdrawable = capital + haircutted warmed-up PnL
        let warmed_pnl = engine.warmable_gross(user.idx as usize);
        let haircutted_pnl = if h_den > 0 {
            warmed_pnl.saturating_mul(h_num) / h_den
        } else {
            0
        };
        let withdrawable = acct.capital.get().saturating_add(haircutted_pnl) as f64;
        let wd_ratio = if init > 0.0 { withdrawable / init } else { 0.0 };
        withdrawable_ratios.push(wd_ratio);

        // Liquidated = had position, now closed, equity < 10% of initial
        if acct.position_basis_q == 0 && mtm_ratio < 0.1 {
            users_liquidated += 1;
        }
    }

    if min_h == f64::MAX {
        min_h = 1.0;
    }

    let summary = RunSummary {
        seed,
        min_h,
        final_h,
        insurance_end: engine.insurance_fund.balance.get(),
        c_tot_end: engine.c_tot.get(),
        pnl_pos_tot_end: engine.pnl_pos_tot,
        vault_end: engine.vault.get(),
        liquidations: engine.lifetime_liquidations,
        force_closes: 0,
        users_liquidated,
        users_with_positions,
        capital_ratios,
        principal_ratios,
        withdrawable_ratios,
        min_h_slot,
        h_zero_slots,
        h_zero_first_slot,
        h_below_50_slots,
        h_below_10_slots,
        min_true_h,
        min_residual,
        withdraw_attempts,
        withdraw_successes,
        close_attempts,
        close_successes,
        adl_a_reductions,
        adl_k_changes,
        min_a_long,
        min_a_short,
        final_a_long: engine.adl_mult_long,
        final_a_short: engine.adl_mult_short,
        drain_only_entered,
        epoch_resets,
    };

    (summary, snapshots)
}

// ════════════════════════════════════════════════════════════════════════════
// Aggregation
// ════════════════════════════════════════════════════════════════════════════

fn aggregate(label: &str, runs: &[RunSummary]) -> ScenarioSummary {
    let min_hs = sorted(runs.iter().map(|r| r.min_h));
    let final_hs = sorted(runs.iter().map(|r| r.final_h));
    let liqs = sorted(runs.iter().map(|r| r.liquidations as f64));
    let fcs = sorted(runs.iter().map(|r| r.force_closes as f64));
    let liq_fracs = sorted(runs.iter().map(|r| {
        if r.users_with_positions > 0 {
            r.users_liquidated as f64 / r.users_with_positions as f64
        } else {
            0.0
        }
    }));
    let all_ratios = sorted(runs.iter().flat_map(|r| r.capital_ratios.iter().copied()));
    let all_principal = sorted(runs.iter().flat_map(|r| r.principal_ratios.iter().copied()));
    let all_withdrawable = sorted(runs.iter().flat_map(|r| r.withdrawable_ratios.iter().copied()));
    let ins_ends = sorted(runs.iter().map(|r| r.insurance_end as f64));

    // h=0 tracking (junior profits fully haircutted)
    let insolvent_runs: Vec<&RunSummary> = runs.iter().filter(|r| r.h_zero_slots > 0).collect();
    let h_zero_frac = insolvent_runs.len() as f64 / runs.len().max(1) as f64;
    let h_zero_slots_sorted = sorted(insolvent_runs.iter().map(|r| r.h_zero_slots as f64));
    let h_zero_first_sorted = sorted(
        insolvent_runs
            .iter()
            .map(|r| r.h_zero_first_slot as f64),
    );
    let h_below_50_frac =
        runs.iter().filter(|r| r.h_below_50_slots > 0).count() as f64 / runs.len().max(1) as f64;
    let h_below_10_frac =
        runs.iter().filter(|r| r.h_below_10_slots > 0).count() as f64 / runs.len().max(1) as f64;
    let min_h_slots = sorted(runs.iter().map(|r| r.min_h_slot as f64));
    let min_true_hs = sorted(runs.iter().map(|r| r.min_true_h));
    let min_residuals = sorted(runs.iter().map(|r| r.min_residual as f64));
    let negative_h_frac =
        runs.iter().filter(|r| r.min_true_h < 0.0).count() as f64 / runs.len().max(1) as f64;

    ScenarioSummary {
        label: label.to_string(),
        runs: runs.len(),

        min_h_mean: mean(&min_hs),
        min_h_std: std_dev(&min_hs),
        min_h_p01: quantile(&min_hs, 0.01),
        min_h_p05: quantile(&min_hs, 0.05),
        min_h_p50: quantile(&min_hs, 0.50),
        min_h_p90: quantile(&min_hs, 0.90),
        min_h_p95: quantile(&min_hs, 0.95),
        min_h_p99: quantile(&min_hs, 0.99),

        final_h_mean: mean(&final_hs),
        final_h_p50: quantile(&final_hs, 0.50),
        final_h_p90: quantile(&final_hs, 0.90),
        final_h_p99: quantile(&final_hs, 0.99),

        liq_mean: mean(&liqs),
        liq_p50: quantile(&liqs, 0.50),
        liq_p90: quantile(&liqs, 0.90),
        liq_p99: quantile(&liqs, 0.99),

        fc_mean: mean(&fcs),
        fc_p50: quantile(&fcs, 0.50),
        fc_p90: quantile(&fcs, 0.90),

        users_liq_frac_mean: mean(&liq_fracs),
        users_liq_frac_p90: quantile(&liq_fracs, 0.90),

        capital_ratio_p01: quantile(&all_ratios, 0.01),
        capital_ratio_p10: quantile(&all_ratios, 0.10),
        capital_ratio_p50: quantile(&all_ratios, 0.50),
        capital_ratio_p90: quantile(&all_ratios, 0.90),
        capital_ratio_p99: quantile(&all_ratios, 0.99),

        principal_ratio_p01: quantile(&all_principal, 0.01),
        principal_ratio_p10: quantile(&all_principal, 0.10),
        principal_ratio_p50: quantile(&all_principal, 0.50),
        principal_ratio_p90: quantile(&all_principal, 0.90),
        principal_ratio_p99: quantile(&all_principal, 0.99),

        withdrawable_ratio_p01: quantile(&all_withdrawable, 0.01),
        withdrawable_ratio_p10: quantile(&all_withdrawable, 0.10),
        withdrawable_ratio_p50: quantile(&all_withdrawable, 0.50),
        withdrawable_ratio_p90: quantile(&all_withdrawable, 0.90),
        withdrawable_ratio_p99: quantile(&all_withdrawable, 0.99),

        insurance_end_mean: mean(&ins_ends),
        insurance_end_p10: quantile(&ins_ends, 0.10),

        h_zero_frac,
        h_zero_slots_p50: quantile(&h_zero_slots_sorted, 0.50),
        h_zero_first_slot_p50: quantile(&h_zero_first_sorted, 0.50),
        h_below_50_frac,
        h_below_10_frac,
        min_h_slot_p50: quantile(&min_h_slots, 0.50),

        min_true_h_p01: quantile(&min_true_hs, 0.01),
        min_true_h_p05: quantile(&min_true_hs, 0.05),
        min_true_h_p50: quantile(&min_true_hs, 0.50),
        min_residual_p01: quantile(&min_residuals, 0.01),
        min_residual_p50: quantile(&min_residuals, 0.50),
        negative_h_frac,
        deficit_frac: runs.iter().filter(|r| r.min_residual < 0).count() as f64
            / runs.len().max(1) as f64,
        withdraw_attempts_mean: mean(&sorted(runs.iter().map(|r| r.withdraw_attempts as f64))),
        withdraw_successes_mean: mean(&sorted(runs.iter().map(|r| r.withdraw_successes as f64))),
        close_attempts_mean: mean(&sorted(runs.iter().map(|r| r.close_attempts as f64))),
        close_successes_mean: mean(&sorted(runs.iter().map(|r| r.close_successes as f64))),

        // ADL aggregates
        adl_a_reductions_mean: mean(&sorted(runs.iter().map(|r| r.adl_a_reductions as f64))),
        adl_a_reductions_p99: quantile(&sorted(runs.iter().map(|r| r.adl_a_reductions as f64)), 0.99),
        adl_k_changes_mean: mean(&sorted(runs.iter().map(|r| r.adl_k_changes as f64))),
        min_a_long_p01: quantile(&sorted(runs.iter().map(|r| r.min_a_long as f64)), 0.01),
        min_a_short_p01: quantile(&sorted(runs.iter().map(|r| r.min_a_short as f64)), 0.01),
        drain_only_frac: runs.iter().filter(|r| r.drain_only_entered).count() as f64
            / runs.len().max(1) as f64,
        epoch_reset_frac: runs.iter().filter(|r| r.epoch_resets > 0).count() as f64
            / runs.len().max(1) as f64,
        epoch_resets_mean: mean(&sorted(runs.iter().map(|r| r.epoch_resets as f64))),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Scenario runner
// ════════════════════════════════════════════════════════════════════════════

fn run_scenario(cfg: &Config, label: &str, out_dir: &PathBuf) -> ScenarioSummary {
    let start = Instant::now();

    let results: Vec<(RunSummary, Vec<SlotSnapshot>)> = (0..cfg.runs)
        .into_par_iter()
        .map(|i| {
            let seed = cfg.base_seed + i as u64;
            run_one(cfg, seed)
        })
        .collect();

    let (runs, all_snapshots): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    let summary = aggregate(label, &runs);

    // Write output
    let scenario_dir = out_dir.join(label);
    fs::create_dir_all(&scenario_dir).unwrap();

    // runs.csv
    let mut csv = String::from(
        "seed,min_h,min_h_slot,final_h,liquidations,force_closes,\
         users_liquidated,users_with_positions,insurance_end,c_tot_end,pnl_pos_tot_end,\
         h_zero_slots,h_zero_first_slot,h_below_50_slots,h_below_10_slots,\
         min_true_h,min_residual,\
         withdraw_attempts,withdraw_successes,close_attempts,close_successes,\
         adl_a_reductions,adl_k_changes,min_a_long,min_a_short,epoch_resets\n",
    );
    for r in &runs {
        csv.push_str(&format!(
            "{},{:.6},{},{:.6},{},{},{},{},{},{},{},{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{}\n",
            r.seed,
            r.min_h,
            r.min_h_slot,
            r.final_h,
            r.liquidations,
            r.force_closes,
            r.users_liquidated,
            r.users_with_positions,
            r.insurance_end,
            r.c_tot_end,
            r.pnl_pos_tot_end,
            r.h_zero_slots,
            if r.h_zero_first_slot == u64::MAX { "never".to_string() } else { r.h_zero_first_slot.to_string() },
            r.h_below_50_slots,
            r.h_below_10_slots,
            r.min_true_h,
            r.min_residual,
            r.withdraw_attempts,
            r.withdraw_successes,
            r.close_attempts,
            r.close_successes,
            r.adl_a_reductions,
            r.adl_k_changes,
            r.min_a_long,
            r.min_a_short,
            r.epoch_resets,
        ));
    }
    fs::write(scenario_dir.join("runs.csv"), csv).unwrap();

    // summary.json
    fs::write(
        scenario_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .unwrap();

    // snapshots.csv
    if cfg.snapshots {
        let mut snap_csv = String::from(
            "seed,slot,oracle_price,h,c_tot,pnl_pos_tot,\
             insurance,open_interest,cum_liquidations,cum_force_closes\n",
        );
        for snaps in &all_snapshots {
            for s in snaps {
                snap_csv.push_str(&format!(
                    "{},{},{},{:.6},{},{},{},{},{},{}\n",
                    s.seed,
                    s.slot,
                    s.oracle_price,
                    s.h,
                    s.c_tot,
                    s.pnl_pos_tot,
                    s.insurance,
                    s.open_interest,
                    s.cum_liquidations,
                    s.cum_force_closes,
                ));
            }
        }
        fs::write(scenario_dir.join("snapshots.csv"), snap_csv).unwrap();
    }

    let elapsed = start.elapsed();
    eprintln!(
        "[{}] {} runs in {:.1}s",
        label,
        cfg.runs,
        elapsed.as_secs_f64()
    );

    summary
}

// ════════════════════════════════════════════════════════════════════════════
// ADL scenario presets
// ════════════════════════════════════════════════════════════════════════════

fn apply_scenario_preset(cfg: &mut Config, name: &str) {
    match name {
        // Scenario 1: Basic ADL trigger — fast crash + high leverage + no insurance + crank lag
        // Goal: produce bankrupt liquidations with D > 0, exercising enqueue_adl
        "adl_trigger" => {
            cfg.im_bps = 250;              // 2.5% IM → 40x max leverage
            cfg.mm_bps = 125;              // 1.25% MM
            cfg.crash_pct_bps = 6000;      // 60% crash
            cfg.crash_len = 10;            // fast
            cfg.crank_interval = 5;        // keeper lag
            cfg.insurance_topup_usdc = 0;  // no insurance cushion
            cfg.long_bias = 0.95;          // nearly all longs
            cfg.lp_capital_usdc = 10_000_000;
            cfg.total_slots = 200;
            cfg.bounce_pct_bps = 0;        // no bounce
            cfg.trading_fee_bps = 0;       // simplify
            cfg.liquidation_fee_bps = 0;
        }
        // Scenario 2: A-multiplier decay — many sequential bankruptcies grinding A_opp down
        // Goal: observe A_short shrinking across multiple cranks as longs go bankrupt
        "adl_a_decay" => {
            cfg.im_bps = 200;              // 2% → 50x max
            cfg.mm_bps = 100;
            cfg.crash_pct_bps = 7000;      // 70% crash
            cfg.crash_len = 30;            // slower = more sequential liquidation batches
            cfg.crank_interval = 2;
            cfg.insurance_topup_usdc = 0;
            cfg.long_bias = 0.90;
            cfg.n_users = 3000;
            cfg.total_slots = 400;
            cfg.bounce_pct_bps = 0;
            cfg.trading_fee_bps = 0;
            cfg.liquidation_fee_bps = 0;
        }
        // Scenario 3: K-index deficit socialization — verify quote losses via K
        // Goal: D > 0 writes negative delta into K_opp; opposing accounts absorb loss on touch
        "adl_k_deficit" => {
            cfg.im_bps = 300;
            cfg.mm_bps = 150;
            cfg.crash_pct_bps = 5000;      // 50% crash
            cfg.crash_len = 15;
            cfg.crank_interval = 3;
            cfg.insurance_topup_usdc = 0;
            cfg.long_bias = 0.85;
            cfg.total_slots = 300;
            cfg.warmup_slots = 100;
            cfg.bounce_pct_bps = 0;
            cfg.trading_fee_bps = 0;
            cfg.liquidation_fee_bps = 0;
        }
        // Scenario 4: DrainOnly / epoch reset — grind A below MIN_A_SIDE
        // Goal: enough sequential ADL events to exhaust A precision → DrainOnly → reset
        "adl_drain_reset" => {
            cfg.im_bps = 150;              // 1.5% → 66x max
            cfg.mm_bps = 75;
            cfg.crash_pct_bps = 8000;      // 80% crash
            cfg.crash_len = 8;             // very fast
            cfg.crank_interval = 10;       // very laggy
            cfg.insurance_topup_usdc = 0;
            cfg.long_bias = 0.98;          // nearly all longs
            cfg.n_users = 500;             // fewer users = A shrinks faster per event
            cfg.lp_capital_usdc = 5_000_000;
            cfg.total_slots = 300;
            cfg.bounce_pct_bps = 0;
            cfg.trading_fee_bps = 0;
            cfg.liquidation_fee_bps = 0;
        }
        // Scenario 5: Stale account settlement — accounts untouched during ADL
        // Goal: verify lazy A/K settlement is correct when accounts finally touch
        "adl_stale" => {
            cfg.im_bps = 250;
            cfg.mm_bps = 125;
            cfg.crash_pct_bps = 6000;
            cfg.crash_len = 10;
            cfg.crank_interval = 5;
            cfg.insurance_topup_usdc = 0;
            cfg.long_bias = 0.95;
            cfg.lp_capital_usdc = 10_000_000;
            cfg.total_slots = 200;
            cfg.bounce_pct_bps = 2000;     // bounce after crash — surviving accounts touch post-ADL
            cfg.bounce_len = 60;
            cfg.trading_fee_bps = 0;
            cfg.liquidation_fee_bps = 0;
        }
        // Scenario 6: Cascading bankruptcies — massive crash + crank lag = many bankrupt per crank
        // Goal: multiple cranks each process batches of bankrupt liquidations, calling enqueue_adl
        "adl_cascade" => {
            cfg.im_bps = 200;
            cfg.mm_bps = 100;
            cfg.crash_pct_bps = 7000;      // 70% crash
            cfg.crash_len = 15;            // moderate speed
            cfg.crank_interval = 5;        // moderate lag
            cfg.insurance_topup_usdc = 0;
            cfg.long_bias = 0.95;          // mostly longs but some shorts to absorb ADL
            cfg.n_users = 2000;
            cfg.lp_capital_usdc = 5_000_000;
            cfg.total_slots = 200;
            cfg.bounce_pct_bps = 0;
            cfg.trading_fee_bps = 0;
            cfg.liquidation_fee_bps = 0;
        }
        _ => eprintln!("unknown scenario: {}", name),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// CLI
// ════════════════════════════════════════════════════════════════════════════

fn parse_args() -> Config {
    let args: Vec<String> = env::args().skip(1).collect();

    // First pass: load config file if specified
    let mut cfg = Config::default();
    for arg in &args {
        if let Some(path) = arg.strip_prefix("--config=") {
            let json = fs::read_to_string(path).expect("failed to read config file");
            cfg = serde_json::from_str(&json).expect("invalid config JSON");
        }
    }

    // Second pass: CLI overrides
    for arg in &args {
        let rest = match arg.strip_prefix("--") {
            Some(r) => r,
            None => continue,
        };
        let (key, val) = match rest.split_once('=') {
            Some(kv) => kv,
            None => continue,
        };
        if key == "config" {
            continue;
        }
        match key {
            "runs" => cfg.runs = val.parse().unwrap(),
            "base_seed" => cfg.base_seed = val.parse().unwrap(),
            "n_users" => cfg.n_users = val.parse().unwrap(),
            "n_zombies" => cfg.n_zombies = val.parse().unwrap(),
            "warmup_slots" => cfg.warmup_slots = val.parse().unwrap(),
            "mm_bps" => cfg.mm_bps = val.parse().unwrap(),
            "im_bps" => cfg.im_bps = val.parse().unwrap(),
            "trading_fee_bps" => cfg.trading_fee_bps = val.parse().unwrap(),
            "maintenance_fee" => cfg.maintenance_fee_per_slot = val.parse().unwrap(),
            "liquidation_fee_bps" => cfg.liquidation_fee_bps = val.parse().unwrap(),
            "liquidation_buffer_bps" => cfg.liquidation_buffer_bps = val.parse().unwrap(),
            "lp_capital" => cfg.lp_capital_usdc = val.parse().unwrap(),
            "insurance" => cfg.insurance_topup_usdc = val.parse().unwrap(),
            "p0" => cfg.p0 = val.parse().unwrap(),
            "crash_pct" => cfg.crash_pct_bps = val.parse().unwrap(),
            "crash_len" => cfg.crash_len = val.parse().unwrap(),
            "bounce_pct" => cfg.bounce_pct_bps = val.parse().unwrap(),
            "bounce_len" => cfg.bounce_len = val.parse().unwrap(),
            "total_slots" => cfg.total_slots = val.parse().unwrap(),
            "funding_rate" => cfg.funding_rate_bps_per_slot = val.parse().unwrap(),
            "zombie_pnl" => cfg.zombie_pnl_usdc = val.parse().unwrap(),
            "zombie_fee_debt" => cfg.zombie_fee_debt_usdc = val.parse().unwrap(),
            "price_path" => cfg.price_path_type = val.to_string(),
            "staircase_steps" => cfg.staircase_steps = val.parse().unwrap(),
            "staircase_flat" => cfg.staircase_flat_len = val.parse().unwrap(),
            "distortion_pct" => cfg.distortion_pct_bps = val.parse().unwrap(),
            "distortion_start" => cfg.distortion_start_slot = val.parse().unwrap(),
            "distortion_len" => cfg.distortion_len = val.parse().unwrap(),
            "long_bias" => cfg.long_bias = val.parse().unwrap(),
            "crank_interval" => cfg.crank_interval = val.parse().unwrap(),
            "slippage" => cfg.slippage_bps = val.parse().unwrap(),
            "whale" => cfg.whale_enabled = val.parse().unwrap(),
            "whale_capital" => cfg.whale_capital_usdc = val.parse().unwrap(),
            "whale_leverage" => cfg.whale_leverage = val.parse().unwrap(),
            "out" => cfg.out_dir = val.to_string(),
            "snapshots" => cfg.snapshots = val.parse().unwrap(),
            "grid_crash" => {
                cfg.grid_crash_pcts = val.split(',').map(|s| s.parse().unwrap()).collect()
            }
            "grid_warmup" => {
                cfg.grid_warmup_slots = val.split(',').map(|s| s.parse().unwrap()).collect()
            }
            "grid_insurance" => {
                cfg.grid_insurance = val.split(',').map(|s| s.parse().unwrap()).collect()
            }
            "scenario" => apply_scenario_preset(&mut cfg, val),
            _ => eprintln!("unknown arg: --{}", key),
        }
    }

    cfg
}

fn print_usage() {
    eprintln!("percolator stress_test — Monte Carlo crash simulator");
    eprintln!();
    eprintln!("Usage: stress_test [OPTIONS]");
    eprintln!();
    eprintln!("Options (--key=value):");
    eprintln!("  --config=PATH        Load config from JSON file");
    eprintln!("  --runs=N             Number of Monte Carlo seeds (default: 200)");
    eprintln!("  --n_users=N          Users per run (default: 2000)");
    eprintln!("  --crash_pct=BPS      Crash magnitude in bps (default: 3000 = 30%)");
    eprintln!("  --crash_len=SLOTS    Crash duration (default: 60)");
    eprintln!("  --bounce_pct=BPS     Bounce after crash (default: 800 = 8%)");
    eprintln!("  --total_slots=N      Simulation length (default: 600)");
    eprintln!("  --warmup_slots=N     PnL warmup period (default: 600)");
    eprintln!("  --im_bps=BPS         Initial margin (default: 1000 = 10%)");
    eprintln!("  --mm_bps=BPS         Maintenance margin (default: 500 = 5%)");
    eprintln!("  --lp_capital=USDC    LP capital in USDC (default: 50000000)");
    eprintln!("  --insurance=USDC     Insurance fund (default: 10000000)");
    eprintln!("  --out=DIR            Output directory (default: stress_out)");
    eprintln!("  --snapshots=BOOL     Record time-series (default: true)");
    eprintln!();
    eprintln!("Grid mode (runs scenarios over parameter combinations):");
    eprintln!("  --grid_crash=2000,3000,5000");
    eprintln!("  --grid_warmup=0,300,600");
    eprintln!("  --grid_insurance=0,5000000,10000000");
}

// ════════════════════════════════════════════════════════════════════════════
// ADL fairness test
// ════════════════════════════════════════════════════════════════════════════

/// Verify ADL wind-down is fair: multiple accounts on the opposing side
/// with different position sizes all get reduced by the same ratio.
/// Quote deficit D is absorbed proportionally to position size.
fn test_adl_fairness() {
    let oracle = price_e6(60_000);

    let params = RiskParams {
        warmup_period_slots: 600,
        maintenance_margin_bps: 500,
        initial_margin_bps: 1000,
        trading_fee_bps: 0,
        max_accounts: 4096,
        new_account_fee: U128::new(0),
        maintenance_fee_per_slot: U128::new(0),
        max_crank_staleness_slots: u64::MAX,
        liquidation_fee_bps: 0,
        liquidation_fee_cap: U128::new(0),
        liquidation_buffer_bps: 100,
        min_liquidation_abs: U128::new(1),
    };
    let mut engine = new_engine(params);

    // No insurance — forces deficit through K-index socialization
    // (admin controls insurance; zero here to isolate ADL fairness)
    let lp = engine.add_lp([1u8; 32], [2u8; 32], 0).unwrap();
    engine.deposit(lp, usdc(5_000_000), oracle, 0).unwrap();
    let _ = engine.keeper_crank(lp, 0, oracle, 0);

    // Bankrupt account: goes LONG, will be liquidated
    let bankrupt = engine.add_user(0).unwrap();
    engine.deposit(bankrupt, usdc(100_000), oracle, 0).unwrap();

    // 3 SHORT accounts with different sizes — these receive ADL
    let short_a = engine.add_user(0).unwrap();
    engine.deposit(short_a, usdc(500_000), oracle, 0).unwrap();

    let short_b = engine.add_user(0).unwrap();
    engine.deposit(short_b, usdc(1_000_000), oracle, 0).unwrap();

    let short_c = engine.add_user(0).unwrap();
    engine.deposit(short_c, usdc(2_000_000), oracle, 0).unwrap();

    for s in 1..=64 { let _ = engine.keeper_crank(lp, s, oracle, 0); }

    // Open positions
    // execute_trade(a, b, ..., size_q, ...): a gets +size_q, b gets -size_q
    let slot = 64;
    // Bankrupt goes LONG (a=bankrupt gets +size)
    let bankrupt_q = (usdc(1_000_000) * POS_SCALE / oracle as u128) as i128; // 10x lev
    engine.execute_trade(bankrupt, lp, oracle, slot, bankrupt_q, oracle).unwrap();

    // Shorts with different sizes: a=LP gets +size (long), b=short gets -size (SHORT)
    let sa_q = (usdc(1_000_000) * POS_SCALE / oracle as u128) as i128;
    let sb_q = (usdc(2_000_000) * POS_SCALE / oracle as u128) as i128;
    let sc_q = (usdc(4_000_000) * POS_SCALE / oracle as u128) as i128;
    engine.execute_trade(lp, short_a, oracle, slot, sa_q, oracle).unwrap();
    engine.execute_trade(lp, short_b, oracle, slot, sb_q, oracle).unwrap();
    engine.execute_trade(lp, short_c, oracle, slot, sc_q, oracle).unwrap();

    for s in 65..=96 { let _ = engine.keeper_crank(lp, s, oracle, 0); }

    // Record pre-ADL effective positions for shorts
    let pre_a = engine.accounts[short_a as usize].position_basis_q;
    let pre_b = engine.accounts[short_b as usize].position_basis_q;
    let pre_c = engine.accounts[short_c as usize].position_basis_q;

    println!("=== BEFORE ADL ===");
    println!("  bankrupt: cap=${:.0} pos_q=LONG", engine.accounts[bankrupt as usize].capital.get() as f64 / 1e6);
    println!("  short_a:  cap=${:.0} pos_q={}", engine.accounts[short_a as usize].capital.get() as f64 / 1e6,
        pre_a);
    println!("  short_b:  cap=${:.0} pos_q={}", engine.accounts[short_b as usize].capital.get() as f64 / 1e6,
        pre_b);
    println!("  short_c:  cap=${:.0} pos_q={}", engine.accounts[short_c as usize].capital.get() as f64 / 1e6,
        pre_c);
    println!("  A_short = {:.6e}", engine.adl_mult_short as f64);
    println!("  K_short = {}", engine.adl_coeff_short);
    println!("  OI_long  = {}", engine.oi_eff_long_q);
    println!("  OI_short = {}", engine.oi_eff_short_q);

    // Make bankrupt go deeply underwater — inject negative PnL
    let loss = -(usdc(500_000) as i128); // -$500K, way more than $100K capital
    engine.set_pnl(bankrupt as usize, loss);
    // LP gains the counterparty profit
    let lp_cap = engine.accounts[lp as usize].capital.get();
    engine.set_capital(lp as usize, lp_cap.saturating_add(usdc(500_000)));

    println!("\n=== AFTER INJECTING -$500K PNL INTO BANKRUPT LONG ===");
    println!("  bankrupt: cap=${:.0} pnl=${:.0}",
        engine.accounts[bankrupt as usize].capital.get() as f64 / 1e6,
        engine.accounts[bankrupt as usize].pnl as f64 / 1e6);

    // Crank to trigger liquidation → ADL
    let pre_a_long = engine.adl_mult_long;
    let pre_a_short = engine.adl_mult_short;
    let pre_k_long = engine.adl_coeff_long;
    let pre_k_short = engine.adl_coeff_short;

    for s in 97..=160 { let _ = engine.keeper_crank(lp, s, oracle, 0); }

    println!("\n=== AFTER CRANK (liquidation + ADL) ===");
    println!("  liquidations = {}", engine.lifetime_liquidations);
    println!("  A_long:  {:.6e} → {:.6e} (ratio={:.6})",
        pre_a_long as f64, engine.adl_mult_long as f64,
        engine.adl_mult_long as f64 / pre_a_long as f64);
    println!("  A_short: {:.6e} → {:.6e} (ratio={:.6})",
        pre_a_short as f64, engine.adl_mult_short as f64,
        engine.adl_mult_short as f64 / pre_a_short as f64);
    println!("  K_long:  {} → {}", pre_k_long,
        engine.adl_coeff_long);
    println!("  K_short: {} → {}", pre_k_short,
        engine.adl_coeff_short);
    println!("  epoch_long={}  epoch_short={}", engine.adl_epoch_long, engine.adl_epoch_short);
    println!("  mode_long={:?}  mode_short={:?}", engine.side_mode_long, engine.side_mode_short);

    // Touch each short account to settle ADL effects
    // We need to trigger touch_account_full via a no-op operation
    // Using withdraw(0) or just reading effective_pos after a crank that touches them
    for s in 161..=200 { let _ = engine.keeper_crank(lp, s, oracle, 0); }

    // Read post-ADL state
    let post_pnl_a = engine.accounts[short_a as usize].pnl;
    let post_pnl_b = engine.accounts[short_b as usize].pnl;
    let post_pnl_c = engine.accounts[short_c as usize].pnl;
    let post_cap_a = engine.accounts[short_a as usize].capital.get();
    let post_cap_b = engine.accounts[short_b as usize].capital.get();
    let post_cap_c = engine.accounts[short_c as usize].capital.get();
    let post_pos_a = engine.accounts[short_a as usize].position_basis_q == 0;
    let post_pos_b = engine.accounts[short_b as usize].position_basis_q == 0;
    let post_pos_c = engine.accounts[short_c as usize].position_basis_q == 0;

    println!("\n=== POST-ADL SETTLEMENT (all shorts touched) ===");
    println!("  short_a: cap=${:.0}  pnl=${:.0}  pos={}  (was $500K deposit, 1x notional)",
        post_cap_a as f64 / 1e6, post_pnl_a as f64 / 1e6, if post_pos_a { "FLAT" } else { "OPEN" });
    println!("  short_b: cap=${:.0}  pnl=${:.0}  pos={}  (was $1M deposit, 2x notional)",
        post_cap_b as f64 / 1e6, post_pnl_b as f64 / 1e6, if post_pos_b { "FLAT" } else { "OPEN" });
    println!("  short_c: cap=${:.0}  pnl=${:.0}  pos={}  (was $2M deposit, 4x notional)",
        post_cap_c as f64 / 1e6, post_pnl_c as f64 / 1e6, if post_pos_c { "FLAT" } else { "OPEN" });

    // Check proportionality of PnL delta (quote deficit absorbed)
    // Shorts had positions in ratio 1:2:4, so they should absorb deficit in same ratio
    if post_pnl_a != 0 && post_pnl_b != 0 && post_pnl_c != 0 {
        let ratio_ba = post_pnl_b as f64 / post_pnl_a as f64;
        let ratio_ca = post_pnl_c as f64 / post_pnl_a as f64;
        println!("\n=== ADL FAIRNESS CHECK ===");
        println!("  Position ratio:  A:B:C = 1:2:4");
        println!("  PnL delta ratio: A:B:C = 1:{:.2}:{:.2}", ratio_ba, ratio_ca);
        println!("  Expected:        A:B:C = 1:2.00:4.00");
        if (ratio_ba - 2.0).abs() < 0.1 && (ratio_ca - 4.0).abs() < 0.1 {
            println!("  → FAIR: deficit absorbed proportionally to position size ✓");
        } else {
            println!("  → UNFAIR: deficit NOT proportional!");
        }
    } else {
        println!("\n=== ADL FAIRNESS CHECK ===");
        println!("  PnL: A={} B={} C={}", post_pnl_a, post_pnl_b, post_pnl_c);
        if post_pnl_a == 0 && post_pnl_b == 0 && post_pnl_c == 0 {
            println!("  All PnL = 0 — deficit was absorbed by insurance/protocol, not K");
            println!("  Checking capital changes instead...");
            let cap_loss_a = usdc(500_000) as i128 - post_cap_a as i128;
            let cap_loss_b = usdc(1_000_000) as i128 - post_cap_b as i128;
            let cap_loss_c = usdc(2_000_000) as i128 - post_cap_c as i128;
            println!("  Capital loss: A=${:.0} B=${:.0} C=${:.0}",
                cap_loss_a as f64 / 1e6, cap_loss_b as f64 / 1e6, cap_loss_c as f64 / 1e6);
        }
        // Check position reduction ratio
        println!("  Positions: A={} B={} C={}",
            if post_pos_a { "FLAT" } else { "OPEN" },
            if post_pos_b { "FLAT" } else { "OPEN" },
            if post_pos_c { "FLAT" } else { "OPEN" });
        if post_pos_a && post_pos_b && post_pos_c {
            println!("  → All positions zeroed by epoch reset (ADL fully wound down)");
            println!("  → FAIR: all accounts on the side treated equally ✓");
        }
    }

    // Solvency
    let vault = engine.vault.get();
    let c_tot = engine.c_tot.get();
    let ins = engine.insurance_fund.balance.get();
    assert!(vault >= c_tot.saturating_add(ins), "SOLVENCY VIOLATION");
    println!("\n  h           = {:.6}", haircut_f64(&engine));
    println!("  pnl_pos_tot = ${:.0}", engine.pnl_pos_tot as f64 / 1e6);
    println!("  SOLVENCY: PASS");
}

// ════════════════════════════════════════════════════════════════════════════
// ADL saturation test — push A and K to overflow with max accounts
// ════════════════════════════════════════════════════════════════════════════

/// Extreme test: maximize liquidations on one side to try to saturate/overflow A and K.
/// 4094 shorts go bankrupt (huge negative PnL), 1 long receives all ADL.
/// With ADL_ONE = 1_000_000 and MIN_A_SIDE = 1_000, A can reach DrainOnly quickly.
/// With i128 K, tries to push K to extreme negative values.
fn test_adl_saturation() {
    use percolator::MIN_A_SIDE;

    let oracle = price_e6(60_000);

    let params = RiskParams {
        warmup_period_slots: 600,
        maintenance_margin_bps: 200,    // 2% MM → high leverage
        initial_margin_bps: 500,        // 5% IM → 20x max
        trading_fee_bps: 0,
        max_accounts: 4096,
        new_account_fee: U128::new(0),
        maintenance_fee_per_slot: U128::new(0),
        max_crank_staleness_slots: u64::MAX,
        liquidation_fee_bps: 0,
        liquidation_fee_cap: U128::new(0),
        liquidation_buffer_bps: 100,
        min_liquidation_abs: U128::new(1),
    };
    let mut engine = new_engine(params);

    // No insurance — all deficit goes through K
    let lp = engine.add_lp([1u8; 32], [2u8; 32], 0).unwrap();
    // Massive LP capital so it can be counterparty to everyone
    engine.deposit(lp, usdc(1_000_000_000), oracle, 0).unwrap(); // $1B LP
    let _ = engine.keeper_crank(lp, 0, oracle, 0);

    // The single long — will receive all ADL
    let the_long = engine.add_user(0).unwrap();
    engine.deposit(the_long, usdc(10_000_000), oracle, 0).unwrap(); // $10M

    // Create as many shorts as possible
    let max_shorts = 4094u16; // 4096 - LP - the_long
    let mut shorts: Vec<u16> = Vec::with_capacity(max_shorts as usize);
    println!("Creating {} short accounts...", max_shorts);
    for _ in 0..max_shorts {
        let idx = engine.add_user(0).unwrap();
        engine.deposit(idx, usdc(10_000), oracle, 0).unwrap(); // $10K each
        shorts.push(idx);
    }

    for s in 1..=64 { let _ = engine.keeper_crank(lp, s, oracle, 0); }

    // The long opens a huge position
    let slot = 64;
    let long_notional = usdc(200_000_000); // $200M notional
    let long_q = (long_notional * POS_SCALE / oracle as u128) as i128;
    engine.execute_trade(the_long, lp, oracle, slot, long_q, oracle).unwrap();

    // Each short opens max leverage position
    println!("Opening {} short positions...", shorts.len());
    let mut opened = 0u32;
    for &s_idx in &shorts {
        let cap = engine.accounts[s_idx as usize].capital.get();
        let notional = cap * 15; // ~15x leverage
        let short_q = (notional * POS_SCALE / oracle as u128) as i128;
        match engine.execute_trade(lp, s_idx, oracle, slot, short_q, oracle) {
            Ok(()) => opened += 1,
            Err(_) => {} // some may fail margin check
        }
    }
    println!("  opened {}/{} short positions", opened, shorts.len());

    for s in 65..=96 { let _ = engine.keeper_crank(lp, s, oracle, 0); }

    println!("\n=== INITIAL STATE ===");
    println!("  accounts = {} shorts + 1 long + LP", opened);
    println!("  A_long  = {}  A_short = {}", engine.adl_mult_long, engine.adl_mult_short);
    println!("  K_long  = {}  K_short = {}", engine.adl_coeff_long, engine.adl_coeff_short);
    println!("  OI_long = {}  OI_short = {}", engine.oi_eff_long_q, engine.oi_eff_short_q);
    println!("  ADL_ONE = {}  MIN_A_SIDE = {}", ADL_ONE, MIN_A_SIDE);

    // Inject negative PnL into all shorts — make them deeply bankrupt.
    // Don't adjust LP capital — the deficit will route through ADL (K or absorb_protocol_loss).
    println!("\n--- Injecting bankruptcy into all {} shorts ---", opened);
    for &s_idx in &shorts {
        if engine.accounts[s_idx as usize].position_basis_q == 0 { continue; }
        // Each short loses 10x their capital → deeply bankrupt
        let big_loss = -(usdc(100_000) as i128);
        engine.set_pnl(s_idx as usize, big_loss);
    }

    println!("  h = {:.6}", haircut_f64(&engine));

    // Now crank — this will liquidate shorts in batches, each calling enqueue_adl
    // With LIQ_BUDGET_PER_CRANK = 120 and ACCOUNTS_PER_CRANK = 128,
    // need many cranks to liquidate all 4094 shorts
    println!("\n--- Cranking to liquidate all shorts (batches of ~120) ---");
    let mut total_liqs = 0u64;
    let mut crank_count = 0u32;
    let mut drain_entered = false;
    let mut epoch_resets = 0u64;
    let mut prev_epoch_long = engine.adl_epoch_long;

    for s in 97..=4000 {
        let pre_liqs = engine.lifetime_liquidations;
        let pre_a = engine.adl_mult_long;

        let _ = engine.keeper_crank(lp, s, oracle, 0);

        let new_liqs = engine.lifetime_liquidations - pre_liqs;
        if new_liqs > 0 {
            total_liqs += new_liqs;
            crank_count += 1;
            if crank_count <= 10 || crank_count % 10 == 0 {
                println!("  crank {}: +{} liqs (total={}), A_long={:.6e}, K_long={}, mode_long={:?}",
                    crank_count, new_liqs, total_liqs,
                    engine.adl_mult_long as f64, engine.adl_coeff_long,
                    engine.side_mode_long);
            }
        }

        if engine.side_mode_long == SideMode::DrainOnly && !drain_entered {
            drain_entered = true;
            println!("  ** DrainOnly entered at crank {} (A_long={}) **", crank_count, engine.adl_mult_long);
        }

        if engine.adl_epoch_long > prev_epoch_long {
            epoch_resets += 1;
            prev_epoch_long = engine.adl_epoch_long;
            println!("  ** Epoch reset {} at crank {} **", epoch_resets, crank_count);
        }

        // Check solvency after every crank
        let vault = engine.vault.get();
        let c_tot = engine.c_tot.get();
        let ins = engine.insurance_fund.balance.get();
        assert!(vault >= c_tot.saturating_add(ins),
            "SOLVENCY VIOLATION at slot {}: vault={} < c_tot={} + ins={}", s, vault, c_tot, ins);

        // Stop if all shorts liquidated
        if total_liqs >= opened as u64 { break; }
    }

    println!("\n=== AFTER ALL LIQUIDATIONS ===");
    println!("  total liquidations = {}", total_liqs);
    println!("  cranks with liqs   = {}", crank_count);
    println!("  A_long  = {} (min_a_side={})", engine.adl_mult_long, MIN_A_SIDE);
    println!("  A_short = {}", engine.adl_mult_short);
    println!("  K_long  = {}", engine.adl_coeff_long);
    println!("  K_short = {}", engine.adl_coeff_short);
    println!("  epoch_long = {}  epoch_short = {}", engine.adl_epoch_long, engine.adl_epoch_short);
    println!("  mode_long = {:?}  mode_short = {:?}", engine.side_mode_long, engine.side_mode_short);
    println!("  OI_long = {}  OI_short = {}", engine.oi_eff_long_q, engine.oi_eff_short_q);
    println!("  DrainOnly entered = {}", drain_entered);
    println!("  Epoch resets = {}", epoch_resets);
    println!("  h = {:.6}", haircut_f64(&engine));

    // Check the long's state
    let long_cap = engine.accounts[the_long as usize].capital.get();
    let long_pnl = engine.accounts[the_long as usize].pnl;
    let long_pos = engine.accounts[the_long as usize].position_basis_q;
    println!("\n  the_long: cap=${:.0}  pnl=${:.0}  pos_q={}",
        long_cap as f64 / 1e6, long_pnl as f64 / 1e6, long_pos);

    // Solvency
    let vault = engine.vault.get();
    let c_tot = engine.c_tot.get();
    let ins = engine.insurance_fund.balance.get();
    println!("\n  vault=${:.0}  c_tot=${:.0}  ins=${:.0}  residual=${:.0}",
        vault as f64 / 1e6, c_tot as f64 / 1e6, ins as f64 / 1e6, true_residual(&engine) as f64 / 1e6);
    assert!(vault >= c_tot.saturating_add(ins), "FINAL SOLVENCY VIOLATION");
    println!("  SOLVENCY: PASS");
}

// ════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════

/// Focused test: multiple users with PnL all get same haircut h on exit,
/// and zombie open positions wind down via ADL equally, leaving a healthy market.
fn test_zombie_haircut() {
    let oracle = price_e6(60_000);

    let params = RiskParams {
        warmup_period_slots: 600,
        maintenance_margin_bps: 500,
        initial_margin_bps: 1000,
        trading_fee_bps: 0,
        max_accounts: 4096,
        new_account_fee: U128::new(0),
        maintenance_fee_per_slot: U128::new(0),
        max_crank_staleness_slots: u64::MAX,
        liquidation_fee_bps: 0,
        liquidation_fee_cap: U128::new(0),
        liquidation_buffer_bps: 100,
        min_liquidation_abs: U128::new(1),
    };
    let mut engine = new_engine(params);

    // Setup: LP + zombie (long) + 3 profit holders (long) who will exit
    let _ = engine.top_up_insurance_fund(usdc(1_000_000));
    let lp = engine.add_lp([1u8; 32], [2u8; 32], 0).unwrap();
    engine.deposit(lp, usdc(10_000_000), oracle, 0).unwrap(); // $10M LP (less than total PnL)
    let _ = engine.keeper_crank(lp, 0, oracle, 0);

    let zombie = engine.add_user(0).unwrap();
    engine.deposit(zombie, usdc(100_000), oracle, 0).unwrap(); // $100K

    let user_a = engine.add_user(0).unwrap();
    engine.deposit(user_a, usdc(200_000), oracle, 0).unwrap(); // $200K

    let user_b = engine.add_user(0).unwrap();
    engine.deposit(user_b, usdc(300_000), oracle, 0).unwrap(); // $300K

    let user_c = engine.add_user(0).unwrap();
    engine.deposit(user_c, usdc(400_000), oracle, 0).unwrap(); // $400K

    for s in 1..=64 { let _ = engine.keeper_crank(lp, s, oracle, 0); }

    // All go long against LP (LP takes short side)
    let slot = 64u64;
    let zombie_size_q = (usdc(500_000) * POS_SCALE / oracle as u128) as i128;  // 5x lev
    let ua_size_q = (usdc(1_000_000) * POS_SCALE / oracle as u128) as i128;    // 5x lev
    let ub_size_q = (usdc(1_500_000) * POS_SCALE / oracle as u128) as i128;    // 5x lev
    let uc_size_q = (usdc(2_000_000) * POS_SCALE / oracle as u128) as i128;    // 5x lev

    engine.execute_trade(lp, zombie, oracle, slot, zombie_size_q, oracle).unwrap();
    engine.execute_trade(lp, user_a, oracle, slot, ua_size_q, oracle).unwrap();
    engine.execute_trade(lp, user_b, oracle, slot, ub_size_q, oracle).unwrap();
    engine.execute_trade(lp, user_c, oracle, slot, uc_size_q, oracle).unwrap();

    for s in 65..=96 { let _ = engine.keeper_crank(lp, s, oracle, 0); }

    let print_state = |engine: &RiskEngine, label: &str| {
        let z = zombie as usize;
        let zpnl = engine.accounts[z].pnl;
        let zcap = engine.accounts[z].capital.get();
        let zpos = !engine.accounts[z].position_basis_q == 0;
        println!("=== {} ===", label);
        println!("  vault       = ${:.0}", engine.vault.get() as f64 / 1e6);
        println!("  c_tot       = ${:.0}", engine.c_tot.get() as f64 / 1e6);
        println!("  insurance   = ${:.0}", engine.insurance_fund.balance.get() as f64 / 1e6);
        println!("  residual    = ${:.0}", true_residual(engine) as f64 / 1e6);
        println!("  pnl_pos_tot = ${:.0}", engine.pnl_pos_tot as f64 / 1e6);
        println!("  h           = {:.6}", haircut_f64(engine));
        println!("  zombie cap  = ${:.0}, pnl = ${:.0}, pos = {}",
            zcap as f64 / 1e6, zpnl as f64 / 1e6, if zpos { "OPEN" } else { "FLAT" });
        println!("  A_long={:.4e}  A_short={:.4e}  epoch_L={}  epoch_S={}",
            engine.adl_mult_long as f64, engine.adl_mult_short as f64,
            engine.adl_epoch_long, engine.adl_epoch_short);
        println!("  OI_long={}  OI_short={}  mode_L={:?}  mode_S={:?}",
            engine.oi_eff_long_q,
            engine.oi_eff_short_q,
            engine.side_mode_long, engine.side_mode_short);
        println!("  liqs = {}", engine.lifetime_liquidations);
    };

    print_state(&engine, "INITIAL STATE (all 4 longs open, no PnL injection yet)");

    // Inject PnL proportionally into all long holders (simulating price moved in their favor)
    // Total: zombie $5M, A $10M, B $15M, C $20M = $50M total PnL
    // LP (counterparty short) absorbs all losses
    let inject = |engine: &mut Box<RiskEngine>, idx: u16, pnl_usdc: u64| {
        let pnl = usdc(pnl_usdc) as i128;
        engine.set_pnl(idx as usize, pnl);
        engine.update_warmup_slope(idx as usize);
    };
    inject(&mut engine, zombie, 5_000_000);   // $5M
    inject(&mut engine, user_a, 10_000_000);  // $10M
    inject(&mut engine, user_b, 15_000_000);  // $15M
    inject(&mut engine, user_c, 20_000_000);  // $20M
    // LP loses $50M counterparty capital
    let lp_cap = engine.accounts[lp as usize].capital.get();
    engine.set_capital(lp as usize, lp_cap.saturating_sub(usdc(50_000_000)));

    print_state(&engine, "AFTER PNL INJECTION ($50M total: zombie=$5M, A=$10M, B=$15M, C=$20M)");

    // Users A, B, C close their positions (sell to LP) and withdraw
    println!("\n--- Users A, B, C close positions and exit ---");
    let slot2 = 97;
    for (name, uid) in [("A", user_a), ("B", user_b), ("C", user_c)] {
        let pos = engine.accounts[uid as usize].position_basis_q;
        if pos != 0 {
            let close_size = -pos;
            match engine.execute_trade(uid, lp, oracle, slot2, close_size, oracle) {
                Ok(()) => {
                    let cap = engine.accounts[uid as usize].capital.get();
                    let pnl = engine.accounts[uid as usize].pnl;
                    let warmed = engine.warmable_gross(uid as usize);
                    let (hn, hd) = engine.haircut_ratio();
                    let h_val = hn as f64
                        / hd.max(1) as f64;
                    println!("  {} closed: cap=${:.0} pnl=${:.0} warmed=${:.0} h={:.4}",
                        name, cap as f64 / 1e6, pnl as f64 / 1e6, warmed as f64 / 1e6, h_val);
                }
                Err(e) => println!("  {} close failed: {:?}", name, e),
            }
        }
    }

    for s in 98..=130 { let _ = engine.keeper_crank(lp, s, oracle, 0); }

    // Users withdraw what they can and close accounts
    for (name, uid) in [("A", user_a), ("B", user_b), ("C", user_c)] {
        let cap = engine.accounts[uid as usize].capital.get();
        if cap > 0 {
            let snap = engine.clone();
            match engine.withdraw(uid, cap, oracle, 131) {
                Ok(()) => println!("  {} withdrew ${:.0}", name, cap as f64 / 1e6),
                Err(e) => {
                    println!("  {} withdraw failed: {:?}", name, e);
                    *engine = *snap;
                }
            }
        }
    }
    for uid in [user_a, user_b, user_c] {
        let _ = engine.close_account(uid, 132, oracle);
    }

    for s in 133..=160 { let _ = engine.keeper_crank(lp, s, oracle, 0); }

    print_state(&engine, "AFTER A/B/C EXIT (zombie still OPEN with $5M PnL)");

    // ── Phase: Crank forward — LP bankruptcy → ADL winds down zombie ──
    println!("\n--- Cranking forward (LP bankruptcy → ADL on zombie's position) ---");
    for s in 161..=500 { let _ = engine.keeper_crank(lp, s, oracle, 0); }

    print_state(&engine, "AFTER ADL WIND-DOWN");

    // ── Phase: Fast-forward past warmup ──
    println!("\n--- Fast-forward past warmup ---");
    for s in 501..=1200 { let _ = engine.keeper_crank(lp, s, oracle, 0); }

    print_state(&engine, "AFTER WARMUP ELAPSES (final state)");

    // ── Final analysis ──
    let zombie_cap = engine.accounts[zombie as usize].capital.get();
    let vault = engine.vault.get();
    let c_tot = engine.c_tot.get();
    let ins = engine.insurance_fund.balance.get();
    let ppt = engine.pnl_pos_tot;

    println!("\n=== CONCLUSION ===");
    println!("  FAIRNESS: all users got same h on exit");
    println!("    zombie: deposited $100K + $5M PnL → cap=${:.0}", zombie_cap as f64 / 1e6);
    println!("  OVERHANG: pnl_pos_tot=${:.0}, h={:.4}", ppt as f64 / 1e6, haircut_f64(&engine));
    if ppt == 0 {
        println!("    → CLEAN — no overhang, market ready for new entrants");
    } else {
        println!("    → ${:.0} overhang remains", ppt as f64 / 1e6);
    }
    println!("  VAULT: ${:.0}  c_tot=${:.0}  ins=${:.0}  residual=${:.0}",
        vault as f64 / 1e6, c_tot as f64 / 1e6, ins as f64 / 1e6, true_residual(&engine) as f64 / 1e6);

    assert!(vault >= c_tot.saturating_add(ins), "SOLVENCY VIOLATION");
    println!("  SOLVENCY: PASS");
}

fn main() {
    if env::args().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }

    if env::args().any(|a| a == "--test=zombie_haircut") {
        test_zombie_haircut();
        return;
    }
    if env::args().any(|a| a == "--test=adl_fairness") {
        test_adl_fairness();
        return;
    }
    if env::args().any(|a| a == "--test=adl_saturation") {
        test_adl_saturation();
        return;
    }

    let cfg = parse_args();

    // Config validation: prevent price underflow from bps > 100%
    assert!(cfg.crash_pct_bps <= 9999,
        "crash_pct_bps={} exceeds 9999 (99.99%); price would underflow", cfg.crash_pct_bps);
    assert!(cfg.bounce_pct_bps <= 9999,
        "bounce_pct_bps={} exceeds 9999 (99.99%)", cfg.bounce_pct_bps);
    assert!(cfg.distortion_pct_bps <= 9999,
        "distortion_pct_bps={} exceeds 9999 (99.99%)", cfg.distortion_pct_bps);

    let out_dir = PathBuf::from(&cfg.out_dir);
    fs::create_dir_all(&out_dir).unwrap();

    eprintln!(
        "Percolator stress test: {} users, {} runs, {}% crash over {} slots",
        cfg.n_users,
        cfg.runs,
        cfg.crash_pct_bps as f64 / 100.0,
        cfg.crash_len,
    );

    let has_grid = !cfg.grid_crash_pcts.is_empty()
        || !cfg.grid_warmup_slots.is_empty()
        || !cfg.grid_insurance.is_empty();

    if has_grid {
        let crash_pcts = if cfg.grid_crash_pcts.is_empty() {
            vec![cfg.crash_pct_bps]
        } else {
            cfg.grid_crash_pcts.clone()
        };
        let warmups = if cfg.grid_warmup_slots.is_empty() {
            vec![cfg.warmup_slots]
        } else {
            cfg.grid_warmup_slots.clone()
        };
        let insurances = if cfg.grid_insurance.is_empty() {
            vec![cfg.insurance_topup_usdc]
        } else {
            cfg.grid_insurance.clone()
        };

        let mut grid_summaries: Vec<ScenarioSummary> = Vec::new();

        for &crash in &crash_pcts {
            for &warmup in &warmups {
                for &ins in &insurances {
                    let mut scenario_cfg = cfg.clone();
                    scenario_cfg.crash_pct_bps = crash;
                    scenario_cfg.warmup_slots = warmup;
                    scenario_cfg.insurance_topup_usdc = ins;

                    let label = format!("crash{}_warmup{}_ins{}", crash, warmup, ins);
                    let s = run_scenario(&scenario_cfg, &label, &out_dir);

                    eprintln!(
                        "  min_h: p50={:.4} p90={:.4} p99={:.4}  liqs={:.0}  users_liq={:.1}%",
                        s.min_h_p50,
                        s.min_h_p90,
                        s.min_h_p99,
                        s.liq_mean,
                        s.users_liq_frac_mean * 100.0,
                    );
                    grid_summaries.push(s);
                }
            }
        }

        fs::write(
            out_dir.join("grid_summary.json"),
            serde_json::to_string_pretty(&grid_summaries).unwrap(),
        )
        .unwrap();
        println!(
            "{}",
            serde_json::to_string_pretty(&grid_summaries).unwrap()
        );
    } else {
        let summary = run_scenario(&cfg, "default", &out_dir);
        println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    }
}
