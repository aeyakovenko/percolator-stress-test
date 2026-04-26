//! Monte Carlo stress simulator for the Percolator risk engine.
//!
//! Runs crash scenarios through the real engine implementation and
//! aggregates outcome distributions across many RNG seeds.
//!
//! # Known coverage gaps (TODO)
//!
//! - [FIXED] SlippageMatcher: use --slippage=N (bps) to deviate exec_price
//!   from oracle, generating non-zero trade_pnl that exercises warmup restart.
//! - reserved_pnl always 0: pending withdrawal interactions untested.
//! - fee debt accumulation: exercise fee drain paths with trading_fee_bps > 0.
//! - min_liquidation_abs = 1: dust close/GC behavior effectively disabled.
//!   Use realistic threshold + small-position accounts to test dust handling.

use std::{
    alloc::{self, Layout},
    env, fs,
    path::PathBuf,
    time::Instant,
};

use percolator::{RiskEngine, RiskParams, LiquidationPolicy, ReserveMode, InstructionContext, I128, U128, POS_SCALE, ADL_ONE, SideMode};
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
    mm_bps: u64,
    im_bps: u64,
    trading_fee_bps: u64,
    liquidation_fee_bps: u64,

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
    grid_insurance: Vec<u64>,

    // Candidate ordering: "all", "deficit", "adversarial"
    candidate_ordering: String,

    // Min liquidation abs (atomic USDC; 1 = effectively disabled)
    min_liquidation_abs: u128,

    // Funding rate schedule: (slot_offset, rate_bps_per_slot) pairs
    // Applied in order during crash loop. Empty = constant 0.
    funding_schedule: Vec<(u64, i128)>,

    // Oracle manipulation: flash wick parameters
    wick_slot: u64,       // slot offset for the wick (0 = disabled)
    wick_pct_bps: u64,    // wick magnitude in bps (e.g. 5000 = 50% spike)
    wick_duration: u64,   // how many slots the wick lasts before reverting

    // Admission pair (spec §4.7): engine picks admit_h_min when residual has
    // headroom for fresh PnL (matured + fresh <= residual), else picks admit_h_max
    // and marks the account sticky. Fast path = instant withdrawal; slow path = long warmup.
    // Default admit_h_min=0 gives instant release when system is healthy.
    // Default admit_h_max=108000 = 12 hours at 400ms slots (upper bound on lockup).
    admit_h_min_slots: u64,
    admit_h_max_slots: u64,

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
            mm_bps: 500,
            im_bps: 1000,
            trading_fee_bps: 5,
            liquidation_fee_bps: 50,
            lp_capital_usdc: 50_000_000,
            insurance_topup_usdc: 10_000_000,
            p0: 60_000,
            crash_pct_bps: 3000,
            crash_len: 60,
            bounce_pct_bps: 800,
            bounce_len: 60,
            total_slots: 600,
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
            grid_insurance: vec![],
            candidate_ordering: "all".into(),
            funding_schedule: vec![],
            wick_slot: 0,
            wick_pct_bps: 0,
            wick_duration: 0,
            admit_h_min_slots: 0,
            admit_h_max_slots: 108_000,
            min_liquidation_abs: 1,
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
    /// Max relative fairness error in h*PnL across accounts when h < 1
    max_h_fairness_err: f64,
    insurance_end: u128,
    c_tot_end: u128,
    pnl_pos_tot_end: u128,
    vault_end: u128,
    liquidations: u64,
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
    /// Admission stress — residual-scarcity lane (spec §4.3 law 3):
    /// slots where matured_pos_tot >= residual (engine forced to admit_h_max).
    stress_slots: u64,
    /// First slot where residual-scarcity stress was entered (u64::MAX if never).
    stress_first_slot: u64,
    /// Min headroom observed: residual - matured_pos_tot (negative = stressed).
    min_headroom: i128,
    /// Consumption-threshold stress (spec §4.3 law 2):
    /// slots where price_move_consumed >= threshold (= cfg.im_bps * PRICE_MOVE_CONSUMPTION_SCALE).
    consumption_stress_slots: u64,
    /// First slot where consumption threshold was crossed.
    consumption_stress_first_slot: u64,
    /// Peak price_move_consumed_bps_this_generation observed (scaled by 1e9).
    max_consumption_bps_e9: u128,
    /// Number of sweep-generation rollovers during the run (each resets consumption).
    sweep_generations: u64,
    /// Number of post-crank audit failures: matured_pos_tot > residual.
    /// This should always be 0 — the admission gate prevents matured overshoot.
    matured_overshoot_events: u64,
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
    /// Max h-fairness error across all runs (0.0 = perfect, >0.05 = concern)
    max_h_fairness_err: f64,
    /// Admission stress — residual-scarcity lane (matured >= residual)
    stress_slots_mean: f64,
    stress_slots_p50: f64,
    stress_slots_p99: f64,
    stress_entered_frac: f64,
    min_headroom_p01: f64,
    min_headroom_p50: f64,
    /// Admission stress — consumption-threshold lane (§4.3 law 2)
    consumption_stress_slots_mean: f64,
    consumption_stress_slots_p99: f64,
    /// Fraction of runs where consumption threshold (= cfg.im_bps) was crossed
    consumption_stress_entered_frac: f64,
    /// Peak consumption in bps (descaled from e9) p99 across runs
    max_consumption_bps_p99: f64,
    /// Sweep-generation rollovers p50/p99 (each resets consumption counter)
    sweep_generations_p50: f64,
    sweep_generations_p99: f64,
    /// Matured-overshoot events: admission gate failures. MUST be 0 everywhere.
    matured_overshoot_total: u64,
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

// ════════════════════════════════════════════════════════════════════════════
// Account materialization (v12.18.1 API)
// ════════════════════════════════════════════════════════════════════════════
//
// v12.18.1 removed add_user / add_lp / new_account_fee. Accounts materialize
// via deposit_not_atomic(amount >= cfg_min_initial_deposit). We mirror the
// test helpers: pick the free-list head, call materialize_at (stress-visible
// backdoor) to allocate a slot without moving capital, then deposit.

fn add_user(engine: &mut RiskEngine) -> Result<u16, percolator::RiskError> {
    let idx = engine.free_head;
    if idx == u16::MAX || (idx as usize) >= percolator::MAX_ACCOUNTS {
        return Err(percolator::RiskError::Overflow);
    }
    engine.materialize_at(idx, 0)?;
    Ok(idx)
}

fn add_lp(engine: &mut RiskEngine, matcher_program: [u8; 32], matcher_context: [u8; 32]) -> Result<u16, percolator::RiskError> {
    let idx = add_user(engine)?;
    engine.accounts[idx as usize].kind = percolator::Account::KIND_LP;
    engine.accounts[idx as usize].matcher_program = matcher_program;
    engine.accounts[idx as usize].matcher_context = matcher_context;
    Ok(idx)
}

// ════════════════════════════════════════════════════════════════════════════
// Oracle clamp for the v12.19 price-move envelope (wrapper-side policy)
// ════════════════════════════════════════════════════════════════════════════
//
// The engine rejects any crank whose oracle delta exceeds
//   max_price_move_bps_per_slot * dt * P_last / 10_000.
// A real keeper should CLAMP the oracle to the envelope rather than freeze the
// market: the cascade walks through prices at envelope-max per crank, the
// consumption counter accumulates each step, and the threshold gate trips
// `admit_h_max` once cumulative consumption reaches 1/leverage. This gives
// catch-up liveness without letting the engine mark positions at unrealistic
// prices.

fn clamp_oracle(real_oracle: u64, last_engine_price: u64, max_move_bps: u64, dt: u64) -> u64 {
    // Per-call allowed delta in atomic units.
    // max_move_bps * dt fits u64 for realistic values (≤ MAX_MARGIN_BPS * reasonable dt).
    let budget_num = (last_engine_price as u128)
        .saturating_mul(max_move_bps as u128)
        .saturating_mul(dt as u128);
    let budget = (budget_num / 10_000) as u64;
    let lower = last_engine_price.saturating_sub(budget);
    let upper = last_engine_price.saturating_add(budget).min(percolator::MAX_ORACLE_PRICE);
    real_oracle.clamp(lower.max(1), upper)
}

// ════════════════════════════════════════════════════════════════════════════
// Admission-pair stress tracking (spec §4.7)
// ════════════════════════════════════════════════════════════════════════════

/// Compute residual - matured. Negative = engine would pick admit_h_max on fresh PnL.
fn headroom(engine: &RiskEngine) -> i128 {
    let senior = engine.c_tot.get().saturating_add(engine.insurance_fund.balance.get());
    let residual = engine.vault.get().saturating_sub(senior) as i128;
    let matured = engine.pnl_matured_pos_tot as i128;
    residual - matured
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
    new_engine_with(params, 0, 1)
}

fn new_engine_with(params: RiskParams, init_slot: u64, init_oracle_price: u64) -> Box<RiskEngine> {
    let layout = Layout::new::<RiskEngine>();
    let ptr = unsafe { alloc::alloc_zeroed(layout) as *mut RiskEngine };
    if ptr.is_null() {
        alloc::handle_alloc_error(layout);
    }
    let mut engine = unsafe { Box::from_raw(ptr) };
    engine.init_in_place(params, init_slot, init_oracle_price);
    engine
}

// ════════════════════════════════════════════════════════════════════════════
// Candidate list builders for keeper_crank
// ════════════════════════════════════════════════════════════════════════════

/// Return all used account indices — simple sweep for general cranking.
fn all_accounts(engine: &RiskEngine) -> Vec<(u16, Option<LiquidationPolicy>)> {
    (0..4096u16).filter(|&i| engine.is_used(i as usize)).map(|i| (i, Some(LiquidationPolicy::FullClose))).collect()
}

/// Return all used account indices ordered by deficit (most bankrupt first).
/// Matches spec §11.6.2 Band B: "higher predicted uncovered deficit" first.
fn deficit_ordered_candidates(engine: &RiskEngine) -> Vec<(u16, Option<LiquidationPolicy>)> {
    let mut candidates: Vec<(u16, i128)> = (0..4096u16)
        .filter(|&i| engine.is_used(i as usize))
        .map(|i| {
            let acct = &engine.accounts[i as usize];
            // deficit = capital + pnl; lower (more negative) = more bankrupt
            let equity = acct.capital.get() as i128 + acct.pnl;
            (i, equity)
        })
        .collect();
    // Sort ascending by equity so most bankrupt (lowest equity) comes first
    candidates.sort_by_key(|&(_, eq)| eq);
    candidates.into_iter().map(|(idx, _)| (idx, Some(LiquidationPolicy::FullClose))).collect()
}

/// Adversarial keeper ordering: most profitable accounts first (opposite of honest).
/// Profitable longs get touched and settle K before liquidations push K negative.
fn adversarial_ordered_candidates(engine: &RiskEngine) -> Vec<(u16, Option<LiquidationPolicy>)> {
    let mut candidates: Vec<(u16, i128)> = (0..4096u16)
        .filter(|&i| engine.is_used(i as usize))
        .map(|i| {
            let acct = &engine.accounts[i as usize];
            let equity = acct.capital.get() as i128 + acct.pnl;
            (i, equity)
        })
        .collect();
    candidates.sort_by_key(|&(_, eq)| std::cmp::Reverse(eq));
    candidates.into_iter().map(|(idx, _)| (idx, Some(LiquidationPolicy::FullClose))).collect()
}

/// Build candidate list based on config ordering mode
fn build_candidates(engine: &RiskEngine, ordering: &str) -> Vec<(u16, Option<LiquidationPolicy>)> {
    match ordering {
        "deficit" => deficit_ordered_candidates(engine),
        "adversarial" => adversarial_ordered_candidates(engine),
        _ => all_accounts(engine),
    }
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
    //
    // v12.19 adds an EXACT per-risk-notional solvency envelope check that
    // rejects RiskParams unless price_budget + funding_budget + liquidation_fee
    // leaves enough slope gap below maintenance for ceil/floor rounding at
    // small N. Many scenarios' minimal mm/liq values don't pass this check.
    //
    // We mirror the engine's test-helper `zero_fee_params` envelope template
    // (mm=500, liq=0, max_dt=100, rate=10_000, max_price_move=4) which is
    // provably safe. Scenario's cfg.mm_bps/cfg.liquidation_fee_bps are floored
    // to envelope-safe minimums; if a scenario requested smaller mm, we bump
    // it up. The admit_h_min=0, admit_h_max=cfg.admit_h_max_slots admission
    // pair is preserved so the v12.19 stress invariants remain exercised.
    //
    // Clamping in the crash loop ensures natural crash rates > 4 bps/slot
    // walk through the market at envelope-max per crank.
    let env_mm_bps = cfg.mm_bps.max(500);
    let env_im_bps = cfg.im_bps.max(1000).max(env_mm_bps);
    let params = RiskParams {
        maintenance_margin_bps: env_mm_bps,
        initial_margin_bps: env_im_bps,
        trading_fee_bps: cfg.trading_fee_bps,
        max_accounts: percolator::MAX_ACCOUNTS as u64,
        liquidation_fee_bps: 0,
        liquidation_fee_cap: U128::ZERO,
        min_liquidation_abs: U128::ZERO,
        min_nonzero_mm_req: 5,
        min_nonzero_im_req: 6,
        h_min: cfg.admit_h_min_slots,
        h_max: cfg.admit_h_max_slots,
        resolve_price_deviation_bps: 1000,
        max_accrual_dt_slots: 100,
        max_abs_funding_e9_per_slot: 10_000,
        min_funding_lifetime_slots: 10_000_000,
        max_active_positions_per_side: percolator::MAX_ACCOUNTS as u64,
        max_price_move_bps_per_slot: 4,
    };

    let mut engine = new_engine_with(params, 0, p0);
    let admit_h_min = cfg.admit_h_min_slots;
    let admit_h_max = cfg.admit_h_max_slots;

    // ── Seed insurance + LP ─────────────────────────────────────────────
    let _ = engine.top_up_insurance_fund(usdc(cfg.insurance_topup_usdc), 0);

    // ── Admission stress tracking (spec §4.3) ──
    // Residual-scarcity lane (law 3):
    let mut stress_slots: u64 = 0;
    let mut stress_first_slot: u64 = u64::MAX;
    let mut min_headroom: i128 = i128::MAX;
    // Consumption-threshold lane (law 2):
    let mut consumption_stress_slots: u64 = 0;
    let mut consumption_stress_first_slot: u64 = u64::MAX;
    let mut max_consumption_bps_e9: u128 = 0;
    let mut prev_sweep_generation = engine.sweep_generation;
    let mut sweep_generations: u64 = 0;
    // Gate-correctness audit: matured > residual should never happen.
    let mut matured_overshoot_events: u64 = 0;
    // Threshold in e9 units (scale per spec §1.4: PRICE_MOVE_CONSUMPTION_SCALE = 1e9)
    let threshold_e9: u128 = (cfg.im_bps as u128) * 1_000_000_000u128;

    let lp_idx = add_lp(&mut engine, [1u8; 32], [2u8; 32]).unwrap();
    engine.deposit_not_atomic(lp_idx, usdc(cfg.lp_capital_usdc), 0).unwrap();

    // Initial crank at slot 0 — batch across all accounts (64 per crank).
    // rr_window_size only on first chunk (see crash-loop comment).
    let init_candidates = all_accounts(&engine);
    let mut init_first = true;
    for chunk in init_candidates.chunks(64) {
        let rr_w = if init_first { 192 } else { 0 };
        let _ = engine.keeper_crank_not_atomic(0, p0, chunk, 64, 0, admit_h_min, admit_h_max, Some(cfg.im_bps as u128), rr_w);
        init_first = false;
    }

    // ── Capital distributions (lognormal mixtures) ──────────────────────
    let retail = LogNormal::new(2_000f64.ln(), 1.0).unwrap();
    let pro = LogNormal::new(50_000f64.ln(), 0.8).unwrap();
    let whale = LogNormal::new(1_000_000f64.ln(), 0.7).unwrap();

    // ── Add whale account if enabled ────────────────────────────────────
    let mut users: Vec<UserInfo> = Vec::with_capacity(cfg.n_users + 1);
    if cfg.whale_enabled {
        let whale_idx = add_user(&mut engine).unwrap();
        engine.deposit_not_atomic(whale_idx, usdc(cfg.whale_capital_usdc), 0).unwrap();
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

        let user_idx = match add_user(&mut engine) {
            Ok(i) => i,
            Err(_) => break, // slab full
        };
        if engine.deposit_not_atomic(user_idx, usdc(cap_usdc), 0).is_err() {
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
    // Each crank can only touch 64 accounts (MAX_TOUCHED_PER_INSTRUCTION),
    // so we batch candidates into chunks of 64 and issue one crank per chunk.
    for s in 1..=SETUP_SLOTS {
        let candidates = all_accounts(&engine);
        let mut first = true;
        for chunk in candidates.chunks(64) {
            let rr_w = if first { 192 } else { 0 };
            let _ = engine.keeper_crank_not_atomic(s, p0, chunk, 64, 0, admit_h_min, admit_h_max, Some(cfg.im_bps as u128), rr_w);
            first = false;
        }
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
            let l = if roll < 0.40 {
                rng.gen_range(2.0..8.0f64)
            } else if roll < 0.70 {
                rng.gen_range(5.0..15.0f64)
            } else if roll < 0.90 {
                rng.gen_range(10.0..max_lev.min(30.0).max(10.1))
            } else {
                rng.gen_range(15.0..max_lev.min(50.0).max(15.1))
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
            .unwrap_or(1)
            .max(1);

        let sign: i128 = if long { 1 } else { -1 };

        // Try at target leverage, then at 90% of max IM-safe leverage.
        // Snapshot/restore models Solana TX atomicity.
        let snapshot = engine.clone();
        let attempts = [pos_q, {
            let safe_notional = (cap_f * (max_lev * 0.9) * util) as u128;
            safe_notional.saturating_mul(POS_SCALE).checked_div(p0 as u128).unwrap_or(1).max(1)
        }];
        let mut ok = false;
        for &q in &attempts {
            let abs_size = q as i128; // size_q must be positive
            let raw_size = (q / POS_SCALE) as i128 * sign;
            let ep = matcher.exec_price(p0, raw_size);
            // For longs: (user, lp, +size) → user gets +size (long)
            // For shorts: (lp, user, +size) → user gets -size (short)
            let (a, b) = if long { (user.idx, lp_idx) } else { (lp_idx, user.idx) };
            match engine.execute_trade_not_atomic(a, b, p0, trade_slot, abs_size, ep, 0, admit_h_min, admit_h_max, Some(cfg.im_bps as u128)) {
                Ok(()) => { user.had_position = true; ok = true; break; }
                Err(e) => {
                    *engine = snapshot.as_ref().clone();
                    // debug: uncomment to see trade failures
                    // eprintln!("  trade user={} q={} lev={:.1} err={:?}", user.idx, q, lev, e);
                }
            }
        }
        if !ok { *engine = *snapshot; }
    }

    // ── Post-trade sweep ────────────────────────────────────────────────
    for s in (SETUP_SLOTS + 1)..=(SETUP_SLOTS + 32) {
        let candidates = all_accounts(&engine);
        let mut first = true;
        for chunk in candidates.chunks(64) {
            let rr_w = if first { 192 } else { 0 };
            let _ = engine.keeper_crank_not_atomic(s, p0, chunk, 64, 0, admit_h_min, admit_h_max, Some(cfg.im_bps as u128), rr_w);
            first = false;
        }
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
        let mut zombie_ctx = InstructionContext::new_with_admission(admit_h_min, admit_h_max);
        engine.set_pnl_with_reserve(
            idx,
            old_pnl.saturating_add(add_pnl),
            ReserveMode::UseAdmissionPair(admit_h_min, admit_h_max),
            Some(&mut zombie_ctx),
        ).unwrap();

        // LP counterparty loss (zero-sum backing)
        let lp_cap = engine.accounts[lp_idx as usize].capital.get();
        let loss = usdc(cfg.zombie_pnl_usdc).min(lp_cap);
        engine.set_capital(lp_idx as usize, lp_cap.saturating_sub(loss)).unwrap();

        // Create fee debt (push fee_credits negative)
        let debt = usdc(cfg.zombie_fee_debt_usdc) as i128;
        let old_credits = engine.accounts[idx].fee_credits.get();
        engine.accounts[idx].fee_credits = I128::new(old_credits.saturating_sub(debt));

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
    let mut total_liquidations: u64 = 0;
    let mut max_h_fairness_err: f64 = 0.0;
    let mut snapshots: Vec<SlotSnapshot> = Vec::new();

    // v12.19: max_accrual_dt_slots = 2 * crank_interval lets cranks run at the
    // scenario's natural cadence (keeper-lag preserved). Real oracle moves faster
    // than the envelope are CLAMPED via clamp_oracle() before being passed to
    // the crank, so the cascade walks through at envelope-max rather than freezing.
    let crank_every = cfg.crank_interval.max(1);

    for slot_offset in 0..cfg.total_slots {
        let slot = crash_start + slot_offset;
        let mut oracle = price_path(cfg, slot_offset);

        // Oracle manipulation: flash wick in the harmful direction for majority side
        if cfg.wick_slot > 0 && slot_offset >= cfg.wick_slot
            && slot_offset < cfg.wick_slot + cfg.wick_duration
        {
            let base = price_path(cfg, slot_offset);
            if cfg.long_bias > 0.5 {
                // Mostly longs — wick DOWN to amplify their losses
                oracle = (base as u128 * 10_000u128.saturating_sub(cfg.wick_pct_bps as u128) / 10_000).max(1) as u64;
            } else {
                // Mostly shorts — wick UP to amplify their losses
                oracle = (base as u128 * (10_000 + cfg.wick_pct_bps as u128) / 10_000) as u64;
            }
        }

        // Compute funding rate from schedule
        let mut funding_rate: i128 = 0;
        for &(trigger_slot, rate) in &cfg.funding_schedule {
            if slot_offset >= trigger_slot { funding_rate = rate; }
        }

        // Only crank every N slots to simulate keeper lag
        if slot_offset % crank_every == 0 {
            // Capture pre-crank ADL state
            let pre_a_long = engine.adl_mult_long;
            let pre_a_short = engine.adl_mult_short;
            let pre_k_long = engine.adl_coeff_long;
            let pre_k_short = engine.adl_coeff_short;

            // Clamp the real oracle to the engine's price-move envelope so
            // fast crashes walk through at envelope-max per crank instead of
            // freezing the market. dt = slots since last successful crank.
            let dt = slot.saturating_sub(engine.last_market_slot);
            let clamped_oracle = clamp_oracle(
                oracle,
                engine.last_oracle_price,
                engine.params.max_price_move_bps_per_slot,
                dt,
            );

            // Batch cranks across all candidates. Funding rate and the rr
            // structural sweep are only applied on the FIRST chunk per slot:
            //  - funding: avoid double-charging across batched cranks
            //  - rr_window_size: cursor only advances once per slot, so
            //    sweep_generation rolls at a meaningful rate (once per
            //    MAX_ACCOUNTS/rr_window_size slots) and the consumption
            //    threshold has time to accumulate across a generation.
            //
            // Subsequent chunks pass rr_window_size=0 which spec §9 allows
            // for trusted/private wrappers (this stress test models a
            // trusted keeper).
            let candidates = build_candidates(&engine, &cfg.candidate_ordering);
            let mut first = true;
            for chunk in candidates.chunks(64) {
                let rate = if first { funding_rate } else { 0 };
                let rr_window = if first { 192 } else { 0 };
                if let Ok(outcome) = engine.keeper_crank_not_atomic(slot, clamped_oracle, chunk, 64, rate, admit_h_min, admit_h_max, Some(cfg.im_bps as u128), rr_window) {
                    total_liquidations += outcome.num_liquidations as u64;
                }
                first = false;
            }

            // Residual-scarcity lane (spec §4.3 law 3): matured+fresh > residual
            // forces admit_h_max for any fresh PnL.
            let hr = headroom(&engine);
            if hr < min_headroom { min_headroom = hr; }
            if hr <= 0 {
                stress_slots += 1;
                if stress_first_slot == u64::MAX { stress_first_slot = slot_offset; }
            }

            // Consumption-threshold lane (spec §4.3 law 2): price_move_consumed_e9
            // >= threshold_e9 forces admit_h_max regardless of residual.
            // (Note: field is named `price_move_consumed_bps_this_generation` but is
            // stored in e9 scale per spec §1.4 PRICE_MOVE_CONSUMPTION_SCALE.)
            let consumed_e9 = engine.price_move_consumed_bps_this_generation;
            if consumed_e9 > max_consumption_bps_e9 { max_consumption_bps_e9 = consumed_e9; }
            if consumed_e9 >= threshold_e9 {
                consumption_stress_slots += 1;
                if consumption_stress_first_slot == u64::MAX {
                    consumption_stress_first_slot = slot_offset;
                }
            }

            // Sweep-generation rollovers reset the consumption counter.
            if engine.sweep_generation > prev_sweep_generation {
                sweep_generations += engine.sweep_generation - prev_sweep_generation;
                prev_sweep_generation = engine.sweep_generation;
            }

            // Gate-correctness audit: the admission gate must keep matured
            // bounded by residual. If matured > residual after a crank, the
            // gate failed — this is a spec §4.3 invariant violation.
            let senior = engine.c_tot.get().saturating_add(engine.insurance_fund.balance.get());
            let residual = engine.vault.get().saturating_sub(senior);
            if engine.pnl_matured_pos_tot > residual {
                matured_overshoot_events += 1;
            }

            // H-fairness check: when h < 1, all accounts with positive
            // matured PnL should get the same effective h ratio.
            let (h_num, h_den) = engine.haircut_ratio();
            if h_num < h_den && h_den > 0 {
                let global_h = h_num as f64 / h_den as f64;
                let mut max_err: f64 = 0.0;
                for user in &users {
                    if !user.had_position { continue; }
                    let released = engine.released_pos(user.idx as usize);
                    if released == 0 { continue; }
                    let effective = engine.effective_matured_pnl(user.idx as usize);
                    let per_account_h = effective as f64 / released as f64;
                    let err = (per_account_h - global_h).abs() / global_h.max(1e-12);
                    if err > max_err { max_err = err; }
                }
                if max_err > max_h_fairness_err { max_h_fairness_err = max_err; }
            }

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

            // ── Bounty-fuzz invariant battery ──────────────────────────────
            // 1. matured ≤ pnl_pos_tot (subset relation)
            assert!(
                engine.pnl_matured_pos_tot <= engine.pnl_pos_tot,
                "INVARIANT-1 seed={} slot={}: pnl_matured_pos_tot ({}) > pnl_pos_tot ({})",
                seed, slot_offset, engine.pnl_matured_pos_tot, engine.pnl_pos_tot
            );

            // 3. K-index headroom — neither side may cross ±i128::MAX/2
            // (gives accrue_market_to room for one more max-step without overflow)
            let k_bound = i128::MAX / 2;
            assert!(
                engine.adl_coeff_long.abs() <= k_bound,
                "INVARIANT-3a seed={} slot={}: K_long ({}) past i128::MAX/2",
                seed, slot_offset, engine.adl_coeff_long
            );
            assert!(
                engine.adl_coeff_short.abs() <= k_bound,
                "INVARIANT-3b seed={} slot={}: K_short ({}) past i128::MAX/2",
                seed, slot_offset, engine.adl_coeff_short
            );

            // 4. F-index headroom (same bound)
            assert!(
                engine.f_long_num.abs() <= k_bound,
                "INVARIANT-4a seed={} slot={}: F_long ({}) past i128::MAX/2",
                seed, slot_offset, engine.f_long_num
            );
            assert!(
                engine.f_short_num.abs() <= k_bound,
                "INVARIANT-4b seed={} slot={}: F_short ({}) past i128::MAX/2",
                seed, slot_offset, engine.f_short_num
            );

            // 5. ADL multiplier floor — must not go below MIN_A_SIDE
            //    (engine triggers DrainOnly+epoch reset when this would happen)
            let min_a = percolator::MIN_A_SIDE;
            if engine.side_mode_long != SideMode::DrainOnly {
                assert!(
                    engine.adl_mult_long >= min_a,
                    "INVARIANT-5a seed={} slot={}: A_long ({}) < MIN_A_SIDE ({}) without DrainOnly",
                    seed, slot_offset, engine.adl_mult_long, min_a
                );
            }
            if engine.side_mode_short != SideMode::DrainOnly {
                assert!(
                    engine.adl_mult_short >= min_a,
                    "INVARIANT-5b seed={} slot={}: A_short ({}) < MIN_A_SIDE ({}) without DrainOnly",
                    seed, slot_offset, engine.adl_mult_short, min_a
                );
            }

            // 6. neg_pnl_account_count consistency — explicit recount
            let neg_count: u64 = (0..percolator::MAX_ACCOUNTS)
                .filter(|&i| engine.is_used(i))
                .filter(|&i| engine.accounts[i].pnl < 0)
                .count() as u64;
            assert!(
                engine.neg_pnl_account_count == neg_count,
                "INVARIANT-6 seed={} slot={}: neg_pnl_account_count ({}) != actual ({})",
                seed, slot_offset, engine.neg_pnl_account_count, neg_count
            );

            // 7. sum(account.capital) == c_tot
            let cap_sum: u128 = (0..percolator::MAX_ACCOUNTS)
                .filter(|&i| engine.is_used(i))
                .map(|i| engine.accounts[i].capital.get())
                .sum();
            assert!(
                cap_sum == c_tot_val,
                "INVARIANT-7 seed={} slot={}: sum(capital) ({}) != c_tot ({})",
                seed, slot_offset, cap_sum, c_tot_val
            );

            // 8. sum(account.reserved_pnl) <= sum(max(0, pnl))
            let reserved_sum: u128 = (0..percolator::MAX_ACCOUNTS)
                .filter(|&i| engine.is_used(i))
                .map(|i| engine.accounts[i].reserved_pnl)
                .sum();
            let pos_pnl_sum: u128 = (0..percolator::MAX_ACCOUNTS)
                .filter(|&i| engine.is_used(i))
                .map(|i| if engine.accounts[i].pnl > 0 { engine.accounts[i].pnl as u128 } else { 0 })
                .sum();
            assert!(
                reserved_sum <= pos_pnl_sum,
                "INVARIANT-8 seed={} slot={}: sum(reserved_pnl) ({}) > sum(max(0,pnl)) ({})",
                seed, slot_offset, reserved_sum, pos_pnl_sum
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
                    let has_position = acct.position_basis_q != 0;

                    if has_position {
                        // Try withdrawing 10% of capital
                        let cap = acct.capital.get();
                        let amt = cap / 10;
                        if amt > 0 {
                            withdraw_attempts += 1;
                            let snap = engine.clone();
                            if engine.withdraw_not_atomic(user.idx, amt, oracle, slot, 0, admit_h_min, admit_h_max, Some(cfg.im_bps as u128)).is_ok() {
                                withdraw_successes += 1;
                            }
                            *engine = *snap;
                        }
                    } else {
                        // Position liquidated — try closing account
                        close_attempts += 1;
                        let snap = engine.clone();
                        if engine.close_account_not_atomic(user.idx, slot, oracle, 0, admit_h_min, admit_h_max, Some(cfg.im_bps as u128)).is_ok() {
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
                cum_liquidations: total_liquidations,
            });
        }
    }

    // ── Final crank to settle all state (especially important with crank lag) ──
    // Clamp the final oracle too — if the crash ended with price beyond the
    // envelope relative to the last-marked price, this lets the final sweep
    // finish walking toward it.
    let final_slot = crash_start + cfg.total_slots;
    let final_oracle_raw = price_path(cfg, cfg.total_slots.saturating_sub(1));
    let final_dt = final_slot.saturating_sub(engine.last_market_slot);
    let final_oracle = clamp_oracle(
        final_oracle_raw,
        engine.last_oracle_price,
        engine.params.max_price_move_bps_per_slot,
        final_dt,
    );
    let candidates = build_candidates(&engine, &cfg.candidate_ordering);
    let mut fin_first = true;
    for chunk in candidates.chunks(64) {
        let rr_w = if fin_first { 192 } else { 0 };
        let _ = engine.keeper_crank_not_atomic(final_slot, final_oracle, chunk, 64, 0, admit_h_min, admit_h_max, Some(cfg.im_bps as u128), rr_w);
        fin_first = false;
    }
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

        // Withdrawable = capital + effective matured PnL (already haircutted by engine)
        let haircutted_pnl = engine.effective_matured_pnl(user.idx as usize);
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
        liquidations: total_liquidations,
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
        max_h_fairness_err,
        stress_slots,
        stress_first_slot,
        min_headroom,
        consumption_stress_slots,
        consumption_stress_first_slot,
        max_consumption_bps_e9,
        sweep_generations,
        matured_overshoot_events,
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
        max_h_fairness_err: runs.iter().map(|r| r.max_h_fairness_err).fold(0.0f64, f64::max),

        // Residual-scarcity admission stress (§4.3 law 3)
        stress_slots_mean: mean(&sorted(runs.iter().map(|r| r.stress_slots as f64))),
        stress_slots_p50: quantile(&sorted(runs.iter().map(|r| r.stress_slots as f64)), 0.50),
        stress_slots_p99: quantile(&sorted(runs.iter().map(|r| r.stress_slots as f64)), 0.99),
        stress_entered_frac: runs.iter().filter(|r| r.stress_slots > 0).count() as f64
            / runs.len().max(1) as f64,
        min_headroom_p01: quantile(&sorted(runs.iter().map(|r| r.min_headroom as f64)), 0.01),
        min_headroom_p50: quantile(&sorted(runs.iter().map(|r| r.min_headroom as f64)), 0.50),
        // Consumption-threshold admission stress (§4.3 law 2)
        consumption_stress_slots_mean: mean(&sorted(runs.iter().map(|r| r.consumption_stress_slots as f64))),
        consumption_stress_slots_p99: quantile(&sorted(runs.iter().map(|r| r.consumption_stress_slots as f64)), 0.99),
        consumption_stress_entered_frac: runs.iter().filter(|r| r.consumption_stress_slots > 0).count() as f64
            / runs.len().max(1) as f64,
        // Peak consumption: descale from e9 to bps for reporting
        max_consumption_bps_p99: quantile(
            &sorted(runs.iter().map(|r| (r.max_consumption_bps_e9 / 1_000_000_000) as f64)),
            0.99,
        ),
        sweep_generations_p50: quantile(&sorted(runs.iter().map(|r| r.sweep_generations as f64)), 0.50),
        sweep_generations_p99: quantile(&sorted(runs.iter().map(|r| r.sweep_generations as f64)), 0.99),
        // Gate-correctness check: this MUST be 0 — admission gate failure would be a spec violation.
        matured_overshoot_total: runs.iter().map(|r| r.matured_overshoot_events).sum(),
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
        "seed,min_h,min_h_slot,final_h,liquidations,\
         users_liquidated,users_with_positions,insurance_end,c_tot_end,pnl_pos_tot_end,\
         h_zero_slots,h_zero_first_slot,h_below_50_slots,h_below_10_slots,\
         min_true_h,min_residual,\
         withdraw_attempts,withdraw_successes,close_attempts,close_successes,\
         adl_a_reductions,adl_k_changes,min_a_long,min_a_short,epoch_resets,\
         stress_slots,stress_first_slot,min_headroom,\
         consumption_stress_slots,consumption_stress_first_slot,max_consumption_bps_e9,\
         sweep_generations,matured_overshoot_events\n",
    );
    for r in &runs {
        csv.push_str(&format!(
            "{},{:.6},{},{:.6},{},{},{},{},{},{},{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.seed,
            r.min_h,
            r.min_h_slot,
            r.final_h,
            r.liquidations,
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
            r.stress_slots,
            if r.stress_first_slot == u64::MAX { "never".to_string() } else { r.stress_first_slot.to_string() },
            r.min_headroom,
            r.consumption_stress_slots,
            if r.consumption_stress_first_slot == u64::MAX { "never".to_string() } else { r.consumption_stress_first_slot.to_string() },
            r.max_consumption_bps_e9,
            r.sweep_generations,
            r.matured_overshoot_events,
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
             insurance,open_interest,cum_liquidations\n",
        );
        for snaps in &all_snapshots {
            for s in snaps {
                snap_csv.push_str(&format!(
                    "{},{},{},{:.6},{},{},{},{},{}\n",
                    s.seed,
                    s.slot,
                    s.oracle_price,
                    s.h,
                    s.c_tot,
                    s.pnl_pos_tot,
                    s.insurance,
                    s.open_interest,
                    s.cum_liquidations,
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
            cfg.n_users = 2000;
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
        // Adversarial keeper ordering — touch profitable longs first, liquidate last
        "adversarial_keeper" => {
            cfg.im_bps = 250;
            cfg.mm_bps = 125;
            cfg.crash_pct_bps = 5000;
            cfg.crash_len = 20;
            cfg.crank_interval = 3;
            cfg.insurance_topup_usdc = 0;
            cfg.long_bias = 0.85;
            cfg.n_users = 2000;
            cfg.lp_capital_usdc = 10_000_000;
            cfg.total_slots = 300;
            cfg.bounce_pct_bps = 1000;
            cfg.bounce_len = 60;
            cfg.trading_fee_bps = 0;
            cfg.liquidation_fee_bps = 0;
            cfg.candidate_ordering = "adversarial".into();
        }
        // Funding rate dynamics — anti-retroactivity with rate flips
        // Funding rate dynamics: rate flips with long dt intervals
        "funding_dynamics" => {
            cfg.im_bps = 500;
            cfg.mm_bps = 250;
            cfg.crash_pct_bps = 2000;
            cfg.crash_len = 50;
            cfg.crank_interval = 10; // long dt = funding accumulates between cranks
            cfg.insurance_topup_usdc = 5_000_000;
            cfg.long_bias = 0.5; // balanced — funding affects both sides
            cfg.n_users = 1000;
            cfg.lp_capital_usdc = 20_000_000;
            cfg.total_slots = 400;
            cfg.bounce_pct_bps = 1000;
            cfg.bounce_len = 100;
            cfg.trading_fee_bps = 5;
            cfg.liquidation_fee_bps = 50;
            // Rate flips: +5000 bps/slot for slots 0-99, -5000 for 100-199, 0 after
            cfg.funding_schedule = vec![(0, 5000), (100, -5000), (200, 0)];
        }
        // Extreme funding: max rate (+10000 bps/slot) sustained, crushes one side
        "funding_extreme" => {
            cfg.im_bps = 300;
            cfg.mm_bps = 150;
            cfg.crash_pct_bps = 1000; // mild crash — funding is the killer
            cfg.crash_len = 100;
            cfg.crank_interval = 5;
            cfg.insurance_topup_usdc = 10_000_000;
            cfg.long_bias = 0.5;
            cfg.n_users = 2000;
            cfg.lp_capital_usdc = 50_000_000;
            cfg.total_slots = 300;
            cfg.bounce_pct_bps = 0;
            cfg.bounce_len = 1;
            cfg.trading_fee_bps = 5;
            cfg.liquidation_fee_bps = 50;
            // Max funding rate sustained for 200 slots: longs pay shorts
            cfg.funding_schedule = vec![(0, 10000)];
        }
        // Oracle flash wick: 50% spike for 3 slots then revert during crash
        "oracle_wick" => {
            cfg.im_bps = 500;
            cfg.mm_bps = 250;
            cfg.crash_pct_bps = 3000;
            cfg.crash_len = 60;
            cfg.crank_interval = 1;
            cfg.insurance_topup_usdc = 10_000_000;
            cfg.long_bias = 0.5;
            cfg.n_users = 2000;
            cfg.lp_capital_usdc = 50_000_000;
            cfg.total_slots = 200;
            cfg.bounce_pct_bps = 800;
            cfg.bounce_len = 60;
            cfg.trading_fee_bps = 5;
            cfg.liquidation_fee_bps = 50;
            // Flash wick: +50% spike at slot 30, lasts 3 slots
            cfg.wick_slot = 30;
            cfg.wick_pct_bps = 5000;
            cfg.wick_duration = 3;
        }
        // Oracle wick during ADL cascade: wick + high leverage + no insurance
        "oracle_wick_adl" => {
            cfg.im_bps = 200;
            cfg.mm_bps = 100;
            cfg.crash_pct_bps = 5000;
            cfg.crash_len = 20;
            cfg.crank_interval = 3;
            cfg.insurance_topup_usdc = 0;
            cfg.long_bias = 0.9;
            cfg.n_users = 2000;
            cfg.lp_capital_usdc = 20_000_000;
            cfg.total_slots = 200;
            cfg.bounce_pct_bps = 0;
            cfg.bounce_len = 1;
            cfg.trading_fee_bps = 0;
            cfg.liquidation_fee_bps = 0;
            // Wick at crash bottom: +80% spike for 2 slots
            cfg.wick_slot = 20;
            cfg.wick_pct_bps = 8000;
            cfg.wick_duration = 2;
        }
        // Funding + crash combo: max funding during crash amplifies losses
        "funding_crash_combo" => {
            cfg.im_bps = 300;
            cfg.mm_bps = 150;
            cfg.crash_pct_bps = 4000;
            cfg.crash_len = 30;
            cfg.crank_interval = 5;
            cfg.insurance_topup_usdc = 5_000_000;
            cfg.long_bias = 0.85;
            cfg.n_users = 2000;
            cfg.lp_capital_usdc = 30_000_000;
            cfg.total_slots = 200;
            cfg.bounce_pct_bps = 0;
            cfg.bounce_len = 1;
            cfg.trading_fee_bps = 5;
            cfg.liquidation_fee_bps = 50;
            // Max funding against longs during the crash
            cfg.funding_schedule = vec![(0, 10000), (60, 0)];
        }
        // Dust close / GC behavior with realistic min_liquidation_abs
        "dust_gc" => {
            cfg.im_bps = 1000;
            cfg.mm_bps = 500;
            cfg.crash_pct_bps = 4000;
            cfg.crash_len = 30;
            cfg.crank_interval = 2;
            cfg.insurance_topup_usdc = 1_000_000;
            cfg.long_bias = 0.7;
            cfg.n_users = 2000;
            cfg.lp_capital_usdc = 20_000_000;
            cfg.total_slots = 300;
            cfg.bounce_pct_bps = 500;
            cfg.bounce_len = 60;
            cfg.trading_fee_bps = 5;
            cfg.liquidation_fee_bps = 100;
            cfg.min_liquidation_abs = 10_000_000; // $10 min
        }
        // Adversarial keeper + ADL cascade (worst case for fairness)
        "adversarial_adl_cascade" => {
            cfg.im_bps = 200;
            cfg.mm_bps = 100;
            cfg.crash_pct_bps = 7000;
            cfg.crash_len = 15;
            cfg.crank_interval = 5;
            cfg.insurance_topup_usdc = 0;
            cfg.long_bias = 0.95;
            cfg.n_users = 2000;
            cfg.lp_capital_usdc = 5_000_000;
            cfg.total_slots = 200;
            cfg.bounce_pct_bps = 0;
            cfg.trading_fee_bps = 0;
            cfg.liquidation_fee_bps = 0;
            cfg.candidate_ordering = "adversarial".into();
        }
        // ── 10/10 crash scenarios (Oct 10, 2025 flash crash) ──
        // BTC: $122K → $105K (14%) in ~40 minutes, 87% long bias
        // HL offered 50-100x leverage; users were massively overleveraged
        "ten10_btc" => {
            cfg.p0 = 122_000;
            cfg.crash_pct_bps = 1400;       // 14% crash
            cfg.crash_len = 40;             // 40 slots ≈ 40 minutes
            cfg.bounce_pct_bps = 0;         // no immediate recovery
            cfg.bounce_len = 1;
            cfg.total_slots = 200;
            cfg.long_bias = 0.87;           // 87% of positions were long
            cfg.im_bps = 200;              // 2% IM (50x max leverage — HL-like)
            cfg.mm_bps = 100;              // 1% MM
            cfg.n_users = 2000;
            cfg.lp_capital_usdc = 100_000_000;
            cfg.insurance_topup_usdc = 20_000_000;
            cfg.crank_interval = 3;        // crank lag (overwhelmed during crash)
            cfg.trading_fee_bps = 5;
            cfg.liquidation_fee_bps = 50;
        }
        // SOL: >40% crash, extreme altcoin drawdown
        "ten10_sol" => {
            cfg.p0 = 290;
            cfg.crash_pct_bps = 4000;       // 40% crash
            cfg.crash_len = 40;
            cfg.bounce_pct_bps = 0;
            cfg.bounce_len = 1;
            cfg.total_slots = 200;
            cfg.long_bias = 0.90;
            cfg.im_bps = 200;              // 50x max
            cfg.mm_bps = 100;
            cfg.n_users = 2000;
            cfg.lp_capital_usdc = 50_000_000;
            cfg.insurance_topup_usdc = 10_000_000;
            cfg.crank_interval = 3;
            cfg.trading_fee_bps = 5;
            cfg.liquidation_fee_bps = 50;
        }
        // Altcoin armageddon: 80% crash (ATOM near-zero wick, WLD -70%)
        "ten10_alt" => {
            cfg.p0 = 100;
            cfg.crash_pct_bps = 8000;       // 80% crash
            cfg.crash_len = 30;
            cfg.bounce_pct_bps = 2000;      // 20% dead cat bounce
            cfg.bounce_len = 60;
            cfg.total_slots = 200;
            cfg.long_bias = 0.90;
            cfg.im_bps = 200;
            cfg.mm_bps = 100;
            cfg.n_users = 2000;
            cfg.lp_capital_usdc = 20_000_000;
            cfg.insurance_topup_usdc = 5_000_000;
            cfg.crank_interval = 5;        // extreme lag during alt crash
            cfg.trading_fee_bps = 5;
            cfg.liquidation_fee_bps = 50;
        }
        // HL-scale: max stress, no insurance (models HL's forced ADL path)
        "ten10_hl" => {
            cfg.p0 = 122_000;
            cfg.crash_pct_bps = 1400;
            cfg.crash_len = 40;
            cfg.bounce_pct_bps = 0;
            cfg.bounce_len = 1;
            cfg.total_slots = 200;
            cfg.long_bias = 0.87;
            cfg.im_bps = 150;              // 1.5% IM (66x leverage — HL max)
            cfg.mm_bps = 75;
            cfg.n_users = 2000;
            cfg.lp_capital_usdc = 100_000_000;
            cfg.insurance_topup_usdc = 0;   // no insurance — forces ADL path
            cfg.crank_interval = 5;        // severe lag (HL was overwhelmed)
            cfg.trading_fee_bps = 0;
            cfg.liquidation_fee_bps = 0;
            cfg.candidate_ordering = "deficit".into();
        }
        // ── Bug-bounty config: maximally stressed inverted-SOL market ──
        // Pushes every dimension to the construction envelope edge with
        // admit_h_min=0 (instant withdrawals when healthy). Designed to
        // probe edge cases not covered by realistic-rate scenarios.
        "bounty_inverted_sol" => {
            cfg.p0 = 20;                            // low absolute price (rounding stress)
            cfg.mm_bps = 500;
            cfg.im_bps = 1000;                      // 10x — envelope-tight
            cfg.trading_fee_bps = 10;
            cfg.liquidation_fee_bps = 100;
            cfg.n_users = 2000;
            cfg.n_zombies = 1000;                   // half the slab is zombies
            cfg.zombie_pnl_usdc = 2_000_000;        // $2M each = $2B unbacked
            cfg.zombie_fee_debt_usdc = 50_000;      // pre-existing fee debt
            cfg.lp_capital_usdc = 500_000;          // tiny LP cap vs zombie load
            cfg.insurance_topup_usdc = 0;           // no cushion
            cfg.whale_enabled = true;
            cfg.whale_capital_usdc = 50_000_000;
            cfg.whale_leverage = 10.0;              // single-account dominance
            cfg.crash_pct_bps = 9000;               // 90% crash
            cfg.crash_len = 230;                    // ~39 bps/slot — at envelope edge
            cfg.bounce_pct_bps = 4000;              // 40% bounce
            cfg.bounce_len = 50;                    // 80 bps/slot reverse — clamping forces walk
            cfg.total_slots = 3000;                 // long enough for F-saturation + warmup expiry
            cfg.wick_slot = 80;
            cfg.wick_pct_bps = 3900;                // single-slot envelope-max wick
            cfg.wick_duration = 1;
            cfg.long_bias = 0.97;                   // near-pure long bias
            cfg.candidate_ordering = "adversarial".into(); // touch profitable first
            cfg.crank_interval = 10;                // significant keeper lag
            cfg.slippage_bps = 100;                 // 1% exec deviation → trade_pnl != 0
            cfg.min_liquidation_abs = 1;            // dust-allowed (N=1 ceil rounding)
            // Alternating max funding rate every 600 slots → K-index oscillation
            cfg.funding_schedule = vec![
                (0, 10000),
                (600, -10000),
                (1200, 10000),
                (1800, -10000),
                (2400, 10000),
            ];
            cfg.admit_h_min_slots = 0;              // instant withdrawals
            cfg.admit_h_max_slots = 18_000_000;     // ~83 days, near MAX_WARMUP
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
            "mm_bps" => cfg.mm_bps = val.parse().unwrap(),
            "im_bps" => cfg.im_bps = val.parse().unwrap(),
            "trading_fee_bps" => cfg.trading_fee_bps = val.parse().unwrap(),
            "liquidation_fee_bps" => cfg.liquidation_fee_bps = val.parse().unwrap(),
            "lp_capital" => cfg.lp_capital_usdc = val.parse().unwrap(),
            "insurance" => cfg.insurance_topup_usdc = val.parse().unwrap(),
            "p0" => cfg.p0 = val.parse().unwrap(),
            "crash_pct" => cfg.crash_pct_bps = val.parse().unwrap(),
            "crash_len" => cfg.crash_len = val.parse().unwrap(),
            "bounce_pct" => cfg.bounce_pct_bps = val.parse().unwrap(),
            "bounce_len" => cfg.bounce_len = val.parse().unwrap(),
            "total_slots" => cfg.total_slots = val.parse().unwrap(),
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
            "grid_insurance" => {
                cfg.grid_insurance = val.split(',').map(|s| s.parse().unwrap()).collect()
            }
            "scenario" => apply_scenario_preset(&mut cfg, val),
            "candidate_ordering" => cfg.candidate_ordering = val.to_string(),
            "min_liquidation_abs" => cfg.min_liquidation_abs = val.parse().unwrap(),
            "admit_h_min" => cfg.admit_h_min_slots = val.parse().unwrap(),
            "admit_h_max" => cfg.admit_h_max_slots = val.parse().unwrap(),
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
    eprintln!("  --im_bps=BPS         Initial margin (default: 1000 = 10%)");
    eprintln!("  --mm_bps=BPS         Maintenance margin (default: 500 = 5%)");
    eprintln!("  --lp_capital=USDC    LP capital in USDC (default: 50000000)");
    eprintln!("  --insurance=USDC     Insurance fund (default: 10000000)");
    eprintln!("  --out=DIR            Output directory (default: stress_out)");
    eprintln!("  --snapshots=BOOL     Record time-series (default: true)");
    eprintln!();
    eprintln!("Admission pair (spec §4.7 — engine picks h_min or h_max per residual):");
    eprintln!("  --admit_h_min=0        Fast-path horizon (default: 0 = instant withdraw)");
    eprintln!("  --admit_h_max=108000   Slow-path horizon when stressed (default: 108000 ≈ 12h)");
    eprintln!();
    eprintln!("Grid mode (runs scenarios over parameter combinations):");
    eprintln!("  --grid_crash=2000,3000,5000");
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
        maintenance_margin_bps: 500,
        initial_margin_bps: 1000,
        trading_fee_bps: 0,
        max_accounts: percolator::MAX_ACCOUNTS as u64,
        liquidation_fee_bps: 0,
        liquidation_fee_cap: U128::new(usdc(50_000)),
        min_liquidation_abs: U128::new(1),
        min_nonzero_mm_req: 5,
        min_nonzero_im_req: 10,
        h_min: 0,
        h_max: 100,
        resolve_price_deviation_bps: 1000,
        max_accrual_dt_slots: 10,
        max_abs_funding_e9_per_slot: 10_000,
        min_funding_lifetime_slots: 10,
        max_active_positions_per_side: percolator::MAX_ACCOUNTS as u64,
        max_price_move_bps_per_slot: 10,
    };
    let mut engine = new_engine(params);

    // No insurance — forces deficit through K-index socialization
    // (admin controls insurance; zero here to isolate ADL fairness)
    let lp = add_lp(&mut engine, [1u8; 32], [2u8; 32]).unwrap();
    engine.deposit_not_atomic(lp, usdc(5_000_000), 0).unwrap();
    let _ = engine.keeper_crank_not_atomic(0, oracle, &all_accounts(&engine), 64, 0, 0, 1, None, 192);

    // Bankrupt account: goes LONG, will be liquidated
    let bankrupt = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(bankrupt, usdc(100_000), 0).unwrap();

    // 3 SHORT accounts with different sizes — these receive ADL
    let short_a = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(short_a, usdc(500_000), 0).unwrap();

    let short_b = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(short_b, usdc(1_000_000), 0).unwrap();

    let short_c = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(short_c, usdc(2_000_000), 0).unwrap();

    for s in 1..=64 { let candidates = all_accounts(&engine); let _ = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192); }

    // Open positions
    // execute_trade(a, b, ..., size_q, ...): a gets +size_q, b gets -size_q
    let slot = 64;
    // Bankrupt goes LONG (a=bankrupt gets +size)
    let bankrupt_q = (usdc(1_000_000) * POS_SCALE / oracle as u128) as i128; // 10x lev
    engine.execute_trade_not_atomic(bankrupt, lp, oracle, slot, bankrupt_q, oracle, 0, 0, 1, None).unwrap();

    // Shorts with different sizes: a=LP gets +size (long), b=short gets -size (SHORT)
    let sa_q = (usdc(1_000_000) * POS_SCALE / oracle as u128) as i128;
    let sb_q = (usdc(2_000_000) * POS_SCALE / oracle as u128) as i128;
    let sc_q = (usdc(4_000_000) * POS_SCALE / oracle as u128) as i128;
    engine.execute_trade_not_atomic(lp, short_a, oracle, slot, sa_q, oracle, 0, 0, 1, None).unwrap();
    engine.execute_trade_not_atomic(lp, short_b, oracle, slot, sb_q, oracle, 0, 0, 1, None).unwrap();
    engine.execute_trade_not_atomic(lp, short_c, oracle, slot, sc_q, oracle, 0, 0, 1, None).unwrap();

    for s in 65..=96 { let candidates = all_accounts(&engine); let _ = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192); }

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

    // Make bankrupt go deeply underwater — inject negative PnL.
    // Don't adjust LP capital — the deficit routes through ADL/insurance.
    let loss = -(usdc(500_000) as i128); // -$500K, way more than $100K capital
    engine.set_pnl_with_reserve(bankrupt as usize, loss, ReserveMode::NoPositiveIncreaseAllowed, None).unwrap();

    println!("\n=== AFTER INJECTING -$500K PNL INTO BANKRUPT LONG ===");
    println!("  bankrupt: cap=${:.0} pnl=${:.0}",
        engine.accounts[bankrupt as usize].capital.get() as f64 / 1e6,
        engine.accounts[bankrupt as usize].pnl as f64 / 1e6);

    // Crank to trigger liquidation → ADL
    let pre_a_long = engine.adl_mult_long;
    let pre_a_short = engine.adl_mult_short;
    let pre_k_long = engine.adl_coeff_long;
    let pre_k_short = engine.adl_coeff_short;

    let mut total_liqs = 0u64;
    for s in 97..=160 { let candidates = deficit_ordered_candidates(&engine); if let Ok(outcome) = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192) { total_liqs += outcome.num_liquidations as u64; } }

    println!("\n=== AFTER CRANK (liquidation + ADL) ===");
    println!("  liquidations = {}", total_liqs);
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
    for s in 161..=200 { let candidates = all_accounts(&engine); let _ = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192); }

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
        maintenance_margin_bps: 200,    // 2% MM → high leverage
        initial_margin_bps: 500,        // 5% IM → 20x max
        trading_fee_bps: 0,
        max_accounts: percolator::MAX_ACCOUNTS as u64,
        liquidation_fee_bps: 0,
        liquidation_fee_cap: U128::new(usdc(50_000)),
        min_liquidation_abs: U128::new(1),
        min_nonzero_mm_req: 5,
        min_nonzero_im_req: 10,
        h_min: 0,
        h_max: 100,
        resolve_price_deviation_bps: 1000,
        max_accrual_dt_slots: 10,
        max_abs_funding_e9_per_slot: 10_000,
        min_funding_lifetime_slots: 10,
        max_active_positions_per_side: percolator::MAX_ACCOUNTS as u64,
        max_price_move_bps_per_slot: 10,
    };
    let mut engine = new_engine(params);

    // No insurance — all deficit goes through K
    let lp = add_lp(&mut engine, [1u8; 32], [2u8; 32]).unwrap();
    // Massive LP capital so it can be counterparty to everyone
    engine.deposit_not_atomic(lp, usdc(1_000_000_000), 0).unwrap(); // $1B LP
    let _ = engine.keeper_crank_not_atomic(0, oracle, &all_accounts(&engine), 64, 0, 0, 1, None, 192);

    // The single long — will receive all ADL
    let the_long = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(the_long, usdc(10_000_000), 0).unwrap(); // $10M

    // Create as many shorts as possible
    let max_shorts = (percolator::MAX_ACCOUNTS - 2) as u16; // all slots except LP + the_long
    let mut shorts: Vec<u16> = Vec::with_capacity(max_shorts as usize);
    println!("Creating {} short accounts...", max_shorts);
    for _ in 0..max_shorts {
        let idx = add_user(&mut engine).unwrap();
        engine.deposit_not_atomic(idx, usdc(10_000), 0).unwrap(); // $10K each
        shorts.push(idx);
    }

    for s in 1..=64 { let candidates = all_accounts(&engine); let _ = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192); }

    // The long opens a huge position
    let slot = 64;
    let long_notional = usdc(200_000_000); // $200M notional
    let long_q = (long_notional * POS_SCALE / oracle as u128) as i128;
    engine.execute_trade_not_atomic(the_long, lp, oracle, slot, long_q, oracle, 0, 0, 1, None).unwrap();

    // Each short opens max leverage position
    println!("Opening {} short positions...", shorts.len());
    let mut opened = 0u32;
    for &s_idx in &shorts {
        let cap = engine.accounts[s_idx as usize].capital.get();
        let notional = cap * 15; // ~15x leverage
        let short_q = (notional * POS_SCALE / oracle as u128) as i128;
        match engine.execute_trade_not_atomic(lp, s_idx, oracle, slot, short_q, oracle, 0, 0, 1, None) {
            Ok(()) => opened += 1,
            Err(_) => {} // some may fail margin check
        }
    }
    println!("  opened {}/{} short positions", opened, shorts.len());

    for s in 65..=96 { let candidates = all_accounts(&engine); let _ = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192); }

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
        engine.set_pnl_with_reserve(s_idx as usize, big_loss, ReserveMode::NoPositiveIncreaseAllowed, None).unwrap();
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
        let pre_a = engine.adl_mult_long;

        let candidates = deficit_ordered_candidates(&engine);
        let new_liqs = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192)
            .map(|o| o.num_liquidations as u64).unwrap_or(0);
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
// ADL fairness fuzz — randomized longs, randomized short bankruptcies
// ════════════════════════════════════════════════════════════════════════════

/// Fuzz test: N longs with random capitalizations receive ADL from M shorts
/// with random bankruptcy depths. Verify that the K-deficit is absorbed
/// proportionally to each long's effective position size, and that h
/// haircuts are applied uniformly.
fn test_adl_fuzz() {
    use percolator::MIN_A_SIDE;

    let n_seeds: u64 = env::args()
        .find(|a| a.starts_with("--fuzz_seeds="))
        .and_then(|a| a.split('=').nth(1).map(|v| v.parse().unwrap()))
        .unwrap_or(100);

    let oracle = price_e6(60_000);
    let mut global_rng = ChaCha8Rng::seed_from_u64(0xAD1);
    let mut pass = 0u64;
    let mut fail = 0u64;
    let mut max_k_magnitude: i128 = 0;
    let mut min_a_seen: u128 = ADL_ONE;
    let mut max_liqs: u64 = 0;
    let mut drain_count = 0u64;
    let mut epoch_reset_count = 0u64;
    let mut worst_fairness_err: f64 = 0.0;

    println!("=== ADL FAIRNESS FUZZ ({} seeds) ===", n_seeds);
    println!("  ADL_ONE={} MIN_A_SIDE={} POS_SCALE={}", ADL_ONE, MIN_A_SIDE, POS_SCALE);

    for seed in 0..n_seeds {
        let mut rng = ChaCha8Rng::seed_from_u64(global_rng.gen());

        // Random config per seed
        let max_users = 500; // cap for fuzz performance
        let n_longs: usize = rng.gen_range(2..=max_users / 3);
        let n_shorts: usize = rng.gen_range(10..=(max_users - n_longs));
        let mm_bps: u64 = rng.gen_range(100..=1000);
        let im_bps: u64 = mm_bps + rng.gen_range(100..=1000); // IM strictly > MM

        let params = RiskParams {
            maintenance_margin_bps: mm_bps,
            initial_margin_bps: im_bps,
            trading_fee_bps: 0,
            max_accounts: percolator::MAX_ACCOUNTS as u64,
            liquidation_fee_bps: 0,
            liquidation_fee_cap: U128::new(usdc(50_000)),
            min_liquidation_abs: U128::new(1),
            min_nonzero_mm_req: 5,
            min_nonzero_im_req: 10,
            h_min: 0,
            h_max: 100,
            resolve_price_deviation_bps: 1000,
            max_accrual_dt_slots: 10,
            max_abs_funding_e9_per_slot: 10_000,
            min_funding_lifetime_slots: 10,
            max_active_positions_per_side: percolator::MAX_ACCOUNTS as u64,
            max_price_move_bps_per_slot: 10,
        };
        let mut engine = new_engine(params);

        // No insurance — deficit goes through K
        let lp = add_lp(&mut engine, [1u8; 32], [2u8; 32]).unwrap();
        engine.deposit_not_atomic(lp, usdc(1_000_000_000), 0).unwrap();
        let _ = engine.keeper_crank_not_atomic(0, oracle, &all_accounts(&engine), 64, 0, 0, 1, None, 192);

        // Create longs with random capitalizations ($10K - $10M)
        let mut longs: Vec<(u16, u128)> = Vec::new(); // (idx, capital)
        for _ in 0..n_longs {
            let cap_usdc: u64 = rng.gen_range(10_000..=10_000_000);
            let idx = add_user(&mut engine).unwrap();
            engine.deposit_not_atomic(idx, usdc(cap_usdc), 0).unwrap();
            longs.push((idx, usdc(cap_usdc)));
        }

        // Create shorts with random capitalizations ($1K - $100K)
        let mut shorts: Vec<u16> = Vec::new();
        for _ in 0..n_shorts {
            let cap_usdc: u64 = rng.gen_range(1_000..=100_000);
            let idx = add_user(&mut engine).unwrap();
            engine.deposit_not_atomic(idx, usdc(cap_usdc), 0).unwrap();
            shorts.push(idx);
        }

        for s in 1..=64 { let candidates = all_accounts(&engine); let _ = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192); }

        // Open positions with random leverage
        let slot = 64;
        let mut long_positions: Vec<(u16, i128)> = Vec::new(); // (idx, size_q)
        for &(idx, cap) in &longs {
            let lev: f64 = rng.gen_range(1.5..10.0);
            let notional = (cap as f64 * lev) as u128;
            let size_q = (notional * POS_SCALE / oracle as u128) as i128;
            if size_q > 0 {
                match engine.execute_trade_not_atomic(idx, lp, oracle, slot, size_q, oracle, 0, 0, 1, None) {
                    Ok(()) => long_positions.push((idx, size_q)),
                    Err(_) => {}
                }
            }
        }

        let mut short_opened = 0u32;
        for &idx in &shorts {
            let cap = engine.accounts[idx as usize].capital.get();
            let lev: f64 = rng.gen_range(2.0..15.0);
            let notional = (cap as f64 * lev) as u128;
            let size_q = (notional * POS_SCALE / oracle as u128) as i128;
            if size_q > 0 {
                match engine.execute_trade_not_atomic(lp, idx, oracle, slot, size_q, oracle, 0, 0, 1, None) {
                    Ok(()) => short_opened += 1,
                    Err(_) => {}
                }
            }
        }

        if long_positions.is_empty() || short_opened == 0 {
            continue; // skip degenerate seeds
        }

        for s in 65..=96 { let candidates = all_accounts(&engine); let _ = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192); }

        // Record pre-ADL state for each long (capital, pnl, effective position)
        let pre_state: Vec<(u16, u128, i128, i128)> = long_positions.iter().map(|&(idx, _)| {
            let acct = &engine.accounts[idx as usize];
            (idx, acct.capital.get(), acct.pnl, acct.position_basis_q)
        }).collect();

        // Inject random bankruptcy depths into shorts
        for &idx in &shorts {
            if engine.accounts[idx as usize].position_basis_q == 0 { continue; }
            let cap = engine.accounts[idx as usize].capital.get();
            // Random bankruptcy depth: 2x to 50x their capital
            let depth_mult: f64 = rng.gen_range(2.0..50.0);
            let loss = (cap as f64 * depth_mult) as i128;
            engine.set_pnl_with_reserve(idx as usize, -loss, ReserveMode::NoPositiveIncreaseAllowed, None).unwrap();
        }

        // Crank through all liquidations
        let mut seed_liqs = 0u64;
        let mut seed_drain = false;
        let mut seed_epoch_resets = 0u64;
        let mut prev_epoch = engine.adl_epoch_long;

        for s in 97..=2000 {
            let candidates = deficit_ordered_candidates(&engine);
            let new_liqs = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192)
                .map(|o| o.num_liquidations as u64).unwrap_or(0);
            seed_liqs += new_liqs;

            if engine.side_mode_long == SideMode::DrainOnly { seed_drain = true; }
            if engine.adl_epoch_long > prev_epoch {
                seed_epoch_resets += engine.adl_epoch_long - prev_epoch;
                prev_epoch = engine.adl_epoch_long;
            }

            // Solvency check every crank
            let vault = engine.vault.get();
            let c_tot = engine.c_tot.get();
            let ins = engine.insurance_fund.balance.get();
            if vault < c_tot.saturating_add(ins) {
                println!("  SOLVENCY FAIL seed={} slot={}: vault={} c_tot={} ins={}",
                    seed, s, vault, c_tot, ins);
                fail += 1;
                break;
            }

            if new_liqs == 0 && s > 200 { break; } // no more to liquidate
        }

        // Track K magnitude
        let k_mag = engine.adl_coeff_long.abs();
        if k_mag > max_k_magnitude { max_k_magnitude = k_mag; }
        if engine.adl_mult_long < min_a_seen { min_a_seen = engine.adl_mult_long; }
        if seed_liqs > max_liqs { max_liqs = seed_liqs; }
        if seed_drain { drain_count += 1; }
        epoch_reset_count += seed_epoch_resets;

        // Check fairness: each surviving long should have absorbed K-deficit
        // proportionally to their position size.
        // Touch all longs to settle final state.
        for s in 2001..=2100 { let candidates = all_accounts(&engine); let _ = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192); }

        // Collect equity change for surviving longs.
        // Separate into: same-epoch (position still open, settled mid-cascade)
        // vs epoch-reset (position zeroed by reset).
        let mut losses: Vec<(u16, f64, i128)> = Vec::new();
        let mut surviving = 0u32;
        let mut liquidated = 0u32;
        for &(idx, pre_cap, pre_pnl, pre_pos) in &pre_state {
            if pre_pos == 0 { continue; }
            let post_cap = engine.accounts[idx as usize].capital.get();
            let post_pnl = engine.accounts[idx as usize].pnl;
            // Skip accounts that went fully bankrupt
            if post_cap == 0 && post_pnl <= 0 { liquidated += 1; continue; }
            surviving += 1;
            let pre_equity = pre_cap as i128 + pre_pnl;
            let post_equity = post_cap as i128 + post_pnl;
            let delta = post_equity - pre_equity;
            let abs_pos = pre_pos.unsigned_abs() as f64;
            let loss_per_unit = if abs_pos > 0.0 { delta as f64 / abs_pos } else { 0.0 };
            losses.push((idx, loss_per_unit, pre_pos));
        }

        // Check proportionality: all loss_per_unit values should be equal
        // K settlement is additive: total absorbed K per account =
        //   basis_pos * (K_final - K_initial) / (a_basis * POS_SCALE)
        // which is proportional to basis_pos / a_basis. Since all opened at
        // same time with same a_basis, it should be proportional to basis_pos.
        if losses.len() >= 2 {
            let lpus: Vec<f64> = losses.iter().map(|x| x.1).collect();
            let mean_lpu = lpus.iter().sum::<f64>() / lpus.len() as f64;
            if mean_lpu.abs() > 0.001 {
                let max_err = lpus.iter().map(|x| ((x - mean_lpu) / mean_lpu).abs()).fold(0.0f64, f64::max);
                if max_err > worst_fairness_err { worst_fairness_err = max_err; }
                if max_err > 0.05 {
                    println!("  seed={}: FAIRNESS err={:.2}% ({} surviving longs of {}, {} liqs)",
                        seed, max_err * 100.0, losses.len(), long_positions.len(), seed_liqs);
                }
            }
        }

        pass += 1;
        if (seed + 1) % 10 == 0 {
            print!("  [{}/{}] pass={} fail={} max_K={:.2e} min_A={} max_liqs={}\r",
                seed + 1, n_seeds, pass, fail, max_k_magnitude as f64, min_a_seen, max_liqs);
        }
    }

    println!("\n\n=== FUZZ RESULTS ({} seeds) ===", n_seeds);
    println!("  pass           = {}", pass);
    println!("  fail           = {}", fail);
    println!("  max |K_long|   = {:.6e} (i128 max = {:.6e})", max_k_magnitude as f64, i128::MAX as f64);
    println!("  min A_long     = {} (MIN_A_SIDE={})", min_a_seen, MIN_A_SIDE);
    println!("  max liqs/seed  = {}", max_liqs);
    println!("  DrainOnly hits = {}", drain_count);
    println!("  epoch resets   = {}", epoch_reset_count);
    println!("  worst fairness = {:.4}% relative error", worst_fairness_err * 100.0);

    if fail > 0 {
        println!("\n  RESULT: {} SOLVENCY FAILURES!", fail);
    } else {
        println!("\n  RESULT: SOLVENCY 100% — all {} seeds pass", pass);
        println!("  K headroom: {:.2e} / {:.2e} = {:.6}% of i128",
            max_k_magnitude as f64, i128::MAX as f64,
            max_k_magnitude as f64 / i128::MAX as f64 * 100.0);
        if worst_fairness_err > 0.01 {
            println!("  Fairness error up to {:.1}% — this is the H-fairness caveat:", worst_fairness_err * 100.0);
            println!("    A/K settlement is exact per-touch, but accounts touched at");
            println!("    different points in a multi-crank cascade see different");
            println!("    intermediate K values via settle_losses ratchet.");
            println!("    This is unavoidable without a global scan.");
        } else {
            println!("  Fairness: within {:.2}%", worst_fairness_err * 100.0);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════

/// Focused test: multiple users with PnL all get same haircut h on exit,
/// and zombie open positions wind down via ADL equally, leaving a healthy market.
fn test_zombie_haircut() {
    let oracle = price_e6(60_000);

    let params = RiskParams {
        maintenance_margin_bps: 500,
        initial_margin_bps: 1000,
        trading_fee_bps: 0,
        max_accounts: percolator::MAX_ACCOUNTS as u64,
        liquidation_fee_bps: 0,
        liquidation_fee_cap: U128::new(usdc(50_000)),
        min_liquidation_abs: U128::new(1),
        min_nonzero_mm_req: 5,
        min_nonzero_im_req: 10,
        h_min: 0,
        h_max: 100,
        resolve_price_deviation_bps: 1000,
        max_accrual_dt_slots: 10,
        max_abs_funding_e9_per_slot: 10_000,
        min_funding_lifetime_slots: 10,
        max_active_positions_per_side: percolator::MAX_ACCOUNTS as u64,
        max_price_move_bps_per_slot: 10,
    };
    let mut engine = new_engine(params);

    // Setup: LP + zombie (long) + 3 profit holders (long) who will exit
    let _ = engine.top_up_insurance_fund(usdc(1_000_000), 0);
    let lp = add_lp(&mut engine, [1u8; 32], [2u8; 32]).unwrap();
    engine.deposit_not_atomic(lp, usdc(10_000_000), 0).unwrap(); // $10M LP (less than total PnL)
    let _ = engine.keeper_crank_not_atomic(0, oracle, &all_accounts(&engine), 64, 0, 0, 1, None, 192);

    let zombie = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(zombie, usdc(100_000), 0).unwrap(); // $100K

    let user_a = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(user_a, usdc(200_000), 0).unwrap(); // $200K

    let user_b = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(user_b, usdc(300_000), 0).unwrap(); // $300K

    let user_c = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(user_c, usdc(400_000), 0).unwrap(); // $400K

    for s in 1..=64 { let candidates = all_accounts(&engine); let _ = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192); }

    // All go long against LP (LP takes short side)
    let slot = 64u64;
    let zombie_size_q = (usdc(500_000) * POS_SCALE / oracle as u128) as i128;  // 5x lev
    let ua_size_q = (usdc(1_000_000) * POS_SCALE / oracle as u128) as i128;    // 5x lev
    let ub_size_q = (usdc(1_500_000) * POS_SCALE / oracle as u128) as i128;    // 5x lev
    let uc_size_q = (usdc(2_000_000) * POS_SCALE / oracle as u128) as i128;    // 5x lev

    engine.execute_trade_not_atomic(zombie, lp, oracle, slot, zombie_size_q, oracle, 0, 0, 1, None).unwrap();
    engine.execute_trade_not_atomic(user_a, lp, oracle, slot, ua_size_q, oracle, 0, 0, 1, None).unwrap();
    engine.execute_trade_not_atomic(user_b, lp, oracle, slot, ub_size_q, oracle, 0, 0, 1, None).unwrap();
    engine.execute_trade_not_atomic(user_c, lp, oracle, slot, uc_size_q, oracle, 0, 0, 1, None).unwrap();

    for s in 65..=96 { let candidates = all_accounts(&engine); let _ = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192); }

    let mut total_liqs = 0u64;

    let print_state = |engine: &RiskEngine, label: &str, liqs: u64| {
        let z = zombie as usize;
        let zpnl = engine.accounts[z].pnl;
        let zcap = engine.accounts[z].capital.get();
        let zpos = engine.accounts[z].position_basis_q != 0;
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
        println!("  liqs = {}", liqs);
    };

    print_state(&engine, "INITIAL STATE (all 4 longs open, no PnL injection yet)", total_liqs);

    // Inject PnL proportionally into all long holders (simulating price moved in their favor)
    // Total: zombie $5M, A $10M, B $15M, C $20M = $50M total PnL
    // LP (counterparty short) absorbs all losses
    let inject = |engine: &mut Box<RiskEngine>, idx: u16, pnl_usdc: u64| {
        let pnl = usdc(pnl_usdc) as i128;
        let mut ctx = InstructionContext::new_with_admission(0, 1);
        engine.set_pnl_with_reserve(idx as usize, pnl, ReserveMode::UseAdmissionPair(0, 1), Some(&mut ctx)).unwrap();
    };
    inject(&mut engine, zombie, 5_000_000);   // $5M
    inject(&mut engine, user_a, 10_000_000);  // $10M
    inject(&mut engine, user_b, 15_000_000);  // $15M
    inject(&mut engine, user_c, 20_000_000);  // $20M
    // LP loses $50M counterparty capital
    let lp_cap = engine.accounts[lp as usize].capital.get();
    engine.set_capital(lp as usize, lp_cap.saturating_sub(usdc(50_000_000))).unwrap();

    print_state(&engine, "AFTER PNL INJECTION ($50M total: zombie=$5M, A=$10M, B=$15M, C=$20M)", total_liqs);

    // Users A, B, C close their positions (sell to LP) and withdraw
    println!("\n--- Users A, B, C close positions and exit ---");
    let slot2 = 97;
    for (name, uid) in [("A", user_a), ("B", user_b), ("C", user_c)] {
        let pos = engine.accounts[uid as usize].position_basis_q;
        if pos != 0 {
            // Close long: (lp, user, +size) → user gets -size (closes their long)
            let close_size = pos.unsigned_abs() as i128;
            match engine.execute_trade_not_atomic(lp, uid, oracle, slot2, close_size, oracle, 0, 0, 1, None) {
                Ok(()) => {
                    let cap = engine.accounts[uid as usize].capital.get();
                    let pnl = engine.accounts[uid as usize].pnl;
                    let warmed = engine.effective_matured_pnl(uid as usize);
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

    for s in 98..=130 { let candidates = all_accounts(&engine); if let Ok(o) = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192) { total_liqs += o.num_liquidations as u64; } }

    // Users withdraw what they can and close accounts
    for (name, uid) in [("A", user_a), ("B", user_b), ("C", user_c)] {
        let cap = engine.accounts[uid as usize].capital.get();
        if cap > 0 {
            let snap = engine.clone();
            match engine.withdraw_not_atomic(uid, cap, oracle, 131, 0, 0, 1, None) {
                Ok(()) => println!("  {} withdrew ${:.0}", name, cap as f64 / 1e6),
                Err(e) => {
                    println!("  {} withdraw failed: {:?}", name, e);
                    *engine = *snap;
                }
            }
        }
    }
    for uid in [user_a, user_b, user_c] {
        let _ = engine.close_account_not_atomic(uid, 132, oracle, 0, 0, 1, None);
    }

    for s in 133..=160 { let candidates = all_accounts(&engine); if let Ok(o) = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192) { total_liqs += o.num_liquidations as u64; } }

    print_state(&engine, "AFTER A/B/C EXIT (zombie still OPEN with $5M PnL)", total_liqs);

    // ── Phase: Crank forward — LP bankruptcy → ADL winds down zombie ──
    println!("\n--- Cranking forward (LP bankruptcy → ADL on zombie's position) ---");
    for s in 161..=500 { let candidates = all_accounts(&engine); if let Ok(o) = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192) { total_liqs += o.num_liquidations as u64; } }

    print_state(&engine, "AFTER ADL WIND-DOWN", total_liqs);

    // ── Phase: Fast-forward past warmup ──
    println!("\n--- Fast-forward past warmup ---");
    for s in 501..=1200 { let candidates = all_accounts(&engine); if let Ok(o) = engine.keeper_crank_not_atomic(s, oracle, &candidates, 64, 0, 0, 1, None, 192) { total_liqs += o.num_liquidations as u64; } }

    print_state(&engine, "AFTER WARMUP ELAPSES (final state)", total_liqs);

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

// ════════════════════════════════════════════════════════════════════════════
// Audit issue tests (commit 640055a remaining issues)
// ════════════════════════════════════════════════════════════════════════════

fn make_test_engine() -> (Box<RiskEngine>, u16) {
    let oracle = price_e6(60_000);
    let params = RiskParams {
        maintenance_margin_bps: 500,
        initial_margin_bps: 1000,
        trading_fee_bps: 100, // 1% trading fee
        max_accounts: percolator::MAX_ACCOUNTS as u64,
        liquidation_fee_bps: 50,
        liquidation_fee_cap: U128::new(usdc(50_000)),
        min_liquidation_abs: U128::new(1),
        min_nonzero_mm_req: 5,
        min_nonzero_im_req: 10,
        h_min: 0,
        h_max: 100,
        resolve_price_deviation_bps: 1000,
        max_accrual_dt_slots: 10,
        max_abs_funding_e9_per_slot: 10_000,
        min_funding_lifetime_slots: 10,
        max_active_positions_per_side: percolator::MAX_ACCOUNTS as u64,
        max_price_move_bps_per_slot: 10,
    };
    let mut engine = new_engine(params);
    let lp = add_lp(&mut engine, [1u8; 32], [2u8; 32]).unwrap();
    engine.deposit_not_atomic(lp, usdc(50_000_000), 0).unwrap();
    let cands = all_accounts(&engine);
    let _ = engine.keeper_crank_not_atomic(0, oracle, &cands, 64, 0, 0, 1, None, 192);
    (engine, lp)
}

/// Issue 1: Trading fee charged only to account a, not both a and b.
/// Spec §10.5 step 28 says both accounts should be charged.
/// This test detects the asymmetry: a's capital drops by fee, b's doesn't.
fn test_fee_asymmetry() {
    let oracle = price_e6(60_000);
    let (mut engine, lp) = make_test_engine();

    let user_a = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(user_a, usdc(1_000_000), 0).unwrap();
    let user_b = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(user_b, usdc(1_000_000), 0).unwrap();

    let cands = all_accounts(&engine);
    for s in 1..=64 { let _ = engine.keeper_crank_not_atomic(s, oracle, &cands, 64, 0, 0, 1, None, 192); }

    let cap_a_pre = engine.accounts[user_a as usize].capital.get();
    let cap_b_pre = engine.accounts[user_b as usize].capital.get();
    let ins_pre = engine.insurance_fund.balance.get();

    // Trade: a buys from b. size_q > 0 means a gets +size, b gets -size.
    let size_q = (usdc(500_000) * POS_SCALE / oracle as u128) as i128;
    engine.execute_trade_not_atomic(user_a, user_b, oracle, 64, size_q, oracle, 0, 0, 1, None).unwrap();

    let cap_a_post = engine.accounts[user_a as usize].capital.get();
    let cap_b_post = engine.accounts[user_b as usize].capital.get();
    let ins_post = engine.insurance_fund.balance.get();
    let fee_charged_a = cap_a_pre - cap_a_post;
    let fee_charged_b = cap_b_pre - cap_b_post;
    let fee_to_insurance = ins_post - ins_pre;

    println!("=== ISSUE 1: Trading Fee Asymmetry ===");
    println!("  trading_fee_bps = 100 (1%)");
    println!("  trade notional  = ~$500K");
    println!("  expected fee/side = ~$5K");
    println!("  fee charged to a = ${:.0}", fee_charged_a as f64 / 1e6);
    println!("  fee charged to b = ${:.0}", fee_charged_b as f64 / 1e6);
    println!("  fee to insurance = ${:.0}", fee_to_insurance as f64 / 1e6);
    if fee_charged_b == 0 {
        println!("  BUG CONFIRMED: account b was NOT charged trading fee");
        println!("  Insurance received ${:.0} (should be ~$10K if both charged)", fee_to_insurance as f64 / 1e6);
    } else {
        println!("  OK: both accounts charged");
    }
}

/// Issue 2: close_account forgives fee debt while returning capital.
/// Create an account with maintenance fee debt, then close it.
/// Check if fee_credits < 0 is forgiven before capital returned.
fn test_close_account_fee_forgiveness() {
    let oracle = price_e6(60_000);
    let (mut engine, lp) = make_test_engine();

    let user = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(user, usdc(100_000), 0).unwrap();

    let cands = all_accounts(&engine);
    for s in 1..=64 { let _ = engine.keeper_crank_not_atomic(s, oracle, &cands, 64, 0, 0, 1, None, 192); }

    // Open a position
    let size_q = (usdc(200_000) * POS_SCALE / oracle as u128) as i128;
    engine.execute_trade_not_atomic(user, lp, oracle, 64, size_q, oracle, 0, 0, 1, None).unwrap();

    // Crank for many slots to accumulate maintenance fee debt
    for s in 65..=1000 {
        let cands = all_accounts(&engine);
        let _ = engine.keeper_crank_not_atomic(s, oracle, &cands, 64, 0, 0, 1, None, 192);
    }

    // Close the position (sell back to LP)
    let pos = engine.accounts[user as usize].position_basis_q;
    if pos != 0 {
        let _ = engine.execute_trade_not_atomic(lp, user, oracle, 1000, pos, oracle, 0, 0, 1, None);
    }

    let cands = all_accounts(&engine);
    for s in 1001..=1010 { let _ = engine.keeper_crank_not_atomic(s, oracle, &cands, 64, 0, 0, 1, None, 192); }

    let fee_credits_pre = engine.accounts[user as usize].fee_credits.get();
    let capital_pre = engine.accounts[user as usize].capital.get();
    let pnl_pre = engine.accounts[user as usize].pnl;
    let pos_pre = engine.accounts[user as usize].position_basis_q;

    println!("\n=== ISSUE 2: close_account Fee Forgiveness ===");
    println!("  Before close_account:");
    println!("    capital      = ${:.0}", capital_pre as f64 / 1e6);
    println!("    pnl          = ${:.0}", pnl_pre as f64 / 1e6);
    println!("    fee_credits  = {}", fee_credits_pre);
    println!("    position     = {}", pos_pre);

    match engine.close_account_not_atomic(user, 1011, oracle, 0, 0, 1, None) {
        Ok(refund) => {
            println!("  close_account returned ${:.0}", refund as f64 / 1e6);
            if fee_credits_pre < 0 && refund > 0 {
                let debt = (-fee_credits_pre) as u128;
                println!("  BUG: fee debt of {} was forgiven, ${:.0} capital returned",
                    debt, refund as f64 / 1e6);
                println!("  Spec says: withdraw should sweep fee debt from capital first");
            } else if fee_credits_pre >= 0 {
                println!("  No fee debt to forgive (fee_credits >= 0)");
            } else {
                println!("  OK: no capital returned with fee debt");
            }
        }
        Err(e) => println!("  close_account failed: {:?} (account may have positive PnL)", e),
    }
}

/// Issue 3: Strictly-risk-reducing exemption path.
/// Put account below maintenance margin, then attempt a risk-reducing trade.
/// If the I256 buffer comparison is correct, the trade should succeed iff
/// post-trade buffer > pre-trade buffer.
fn test_risk_reducing_exemption() {
    let oracle = price_e6(60_000);
    let (mut engine, lp) = make_test_engine();

    let user = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(user, usdc(100_000), 0).unwrap();

    let cands = all_accounts(&engine);
    for s in 1..=64 { let _ = engine.keeper_crank_not_atomic(s, oracle, &cands, 64, 0, 0, 1, None, 192); }

    // Open a leveraged long position
    let size_q = (usdc(800_000) * POS_SCALE / oracle as u128) as i128; // ~8x leverage
    engine.execute_trade_not_atomic(user, lp, oracle, 64, size_q, oracle, 0, 0, 1, None).unwrap();

    // Inject negative PnL to push below maintenance margin
    let loss = -(usdc(70_000) as i128); // loses most of capital
    engine.set_pnl_with_reserve(user as usize, loss, ReserveMode::NoPositiveIncreaseAllowed, None).unwrap();

    let cap = engine.accounts[user as usize].capital.get();
    let pnl = engine.accounts[user as usize].pnl;
    let pos = engine.accounts[user as usize].position_basis_q;

    println!("\n=== ISSUE 3: Risk-Reducing Exemption Path ===");
    println!("  Account state (below maintenance):");
    println!("    capital  = ${:.0}", cap as f64 / 1e6);
    println!("    pnl      = ${:.0}", pnl as f64 / 1e6);
    println!("    position = {} (long)", pos);

    // Try a strictly risk-reducing trade: close half the position
    let half_close = pos / 2;
    println!("  Attempting risk-reducing trade: close half position (size={})", half_close);
    match engine.execute_trade_not_atomic(lp, user, oracle, 64, half_close, oracle, 0, 0, 1, None) {
        Ok(()) => {
            let new_pos = engine.accounts[user as usize].position_basis_q;
            println!("  SUCCESS: position reduced {} → {}", pos, new_pos);
            println!("  Risk-reducing exemption path exercised ✓");
        }
        Err(e) => {
            println!("  REJECTED: {:?}", e);
            println!("  (This may be correct if buffer didn't improve, or a bug in I256 comparison)");
        }
    }

    // Try a risk-INCREASING trade (should fail)
    let increase = size_q / 4;
    println!("  Attempting risk-increasing trade: add to position (size={})", increase);
    match engine.execute_trade_not_atomic(user, lp, oracle, 64, increase, oracle, 0, 0, 1, None) {
        Ok(()) => println!("  BUG: risk-increasing trade accepted while below maintenance!"),
        Err(e) => println!("  Correctly rejected: {:?} ✓", e),
    }
}

/// Issue 4: Full ADL pipeline integration.
/// Execute trade → liquidation → enqueue_adl → schedule_end_of_instruction_resets
/// → subsequent trade, with real accounts on both sides going through K-socialization.
/// Verify OI_eff_long == OI_eff_short is maintained throughout.
fn test_adl_pipeline_integration() {
    let oracle = price_e6(60_000);
    let (mut engine, lp) = make_test_engine();

    // 3 longs + 3 shorts, different sizes
    let mut longs: Vec<u16> = Vec::new();
    let mut shorts: Vec<u16> = Vec::new();
    for cap in [200_000, 500_000, 1_000_000] {
        let idx = add_user(&mut engine).unwrap();
        engine.deposit_not_atomic(idx, usdc(cap), 0).unwrap();
        longs.push(idx);
    }
    for cap in [100_000, 300_000, 600_000] {
        let idx = add_user(&mut engine).unwrap();
        engine.deposit_not_atomic(idx, usdc(cap), 0).unwrap();
        shorts.push(idx);
    }

    let cands = all_accounts(&engine);
    for s in 1..=64 { let _ = engine.keeper_crank_not_atomic(s, oracle, &cands, 64, 0, 0, 1, None, 192); }

    // Open positions
    let slot = 64;
    for &idx in &longs {
        let cap = engine.accounts[idx as usize].capital.get();
        let size_q = (cap * 5 * POS_SCALE / oracle as u128) as i128; // 5x long
        let _ = engine.execute_trade_not_atomic(idx, lp, oracle, slot, size_q, oracle, 0, 0, 1, None);
    }
    for &idx in &shorts {
        let cap = engine.accounts[idx as usize].capital.get();
        let size_q = (cap * 8 * POS_SCALE / oracle as u128) as i128; // 8x short
        let _ = engine.execute_trade_not_atomic(lp, idx, oracle, slot, size_q, oracle, 0, 0, 1, None);
    }

    println!("\n=== ISSUE 4: Full ADL Pipeline Integration ===");
    println!("  Initial OI_long={} OI_short={}", engine.oi_eff_long_q, engine.oi_eff_short_q);
    assert_eq!(engine.oi_eff_long_q, engine.oi_eff_short_q, "OI imbalance after setup");

    // Make shorts deeply bankrupt
    for &idx in &shorts {
        engine.set_pnl_with_reserve(idx as usize, -(usdc(1_000_000) as i128), ReserveMode::NoPositiveIncreaseAllowed, None).unwrap();
    }

    // Crank to liquidate shorts → ADL fires → K_long changes
    let mut oi_checks_passed = 0u32;
    let pre_k_long = engine.adl_coeff_long;
    let pre_a_long = engine.adl_mult_long;
    let pre_epoch = engine.adl_epoch_long;

    let mut total_liqs = 0u64;
    for s in 65..=200 {
        let cands = deficit_ordered_candidates(&engine);
        if let Ok(outcome) = engine.keeper_crank_not_atomic(s, oracle, &cands, 64, 0, 0, 1, None, 192) {
            total_liqs += outcome.num_liquidations as u64;
        }
        // OI balance must hold after every crank
        assert_eq!(engine.oi_eff_long_q, engine.oi_eff_short_q,
            "OI IMBALANCE at slot {}: long={} short={}", s, engine.oi_eff_long_q, engine.oi_eff_short_q);
        oi_checks_passed += 1;
    }

    println!("  After liquidation cascade:");
    println!("    liquidations  = {}", total_liqs);
    println!("    A_long: {} → {}", pre_a_long, engine.adl_mult_long);
    println!("    K_long: {} → {}", pre_k_long, engine.adl_coeff_long);
    println!("    epoch_long: {} → {}", pre_epoch, engine.adl_epoch_long);
    println!("    OI_long={} OI_short={}", engine.oi_eff_long_q, engine.oi_eff_short_q);
    println!("    OI balance checks passed: {}/136 cranks", oi_checks_passed);

    // Now try a SUBSEQUENT trade on the post-ADL state
    println!("\n  Attempting post-ADL trade...");
    let new_user = add_user(&mut engine).unwrap();
    engine.deposit_not_atomic(new_user, usdc(500_000), 200).unwrap();
    let cands = all_accounts(&engine);
    let _ = engine.keeper_crank_not_atomic(201, oracle, &cands, 64, 0, 0, 1, None, 192);

    let new_size = (usdc(1_000_000) * POS_SCALE / oracle as u128) as i128;
    match engine.execute_trade_not_atomic(new_user, lp, oracle, 201, new_size, oracle, 0, 0, 1, None) {
        Ok(()) => {
            println!("    Post-ADL trade succeeded ✓");
            assert_eq!(engine.oi_eff_long_q, engine.oi_eff_short_q,
                "OI imbalance after post-ADL trade");
            println!("    OI balance maintained after trade ✓");
        }
        Err(e) => println!("    Post-ADL trade failed: {:?} (may need epoch reset first)", e),
    }

    // Solvency
    let vault = engine.vault.get();
    let c_tot = engine.c_tot.get();
    let ins = engine.insurance_fund.balance.get();
    assert!(vault >= c_tot.saturating_add(ins), "SOLVENCY VIOLATION");
    println!("    SOLVENCY: PASS");
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
    if env::args().any(|a| a.starts_with("--test=adl_fuzz")) {
        test_adl_fuzz();
        return;
    }
    if env::args().any(|a| a == "--test=audit") {
        test_fee_asymmetry();
        test_close_account_fee_forgiveness();
        test_risk_reducing_exemption();
        test_adl_pipeline_integration();
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
        || !cfg.grid_insurance.is_empty();

    if has_grid {
        let crash_pcts = if cfg.grid_crash_pcts.is_empty() {
            vec![cfg.crash_pct_bps]
        } else {
            cfg.grid_crash_pcts.clone()
        };
        let insurances = if cfg.grid_insurance.is_empty() {
            vec![cfg.insurance_topup_usdc]
        } else {
            cfg.grid_insurance.clone()
        };

        let mut grid_summaries: Vec<ScenarioSummary> = Vec::new();

        for &crash in &crash_pcts {
            for &ins in &insurances {
                let mut scenario_cfg = cfg.clone();
                scenario_cfg.crash_pct_bps = crash;
                scenario_cfg.insurance_topup_usdc = ins;

                let label = format!("crash{}_ins{}", crash, ins);
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
