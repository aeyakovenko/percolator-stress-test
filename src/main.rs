//! Monte Carlo stress simulator for the Percolator risk engine.
//!
//! Runs crash scenarios through the real engine implementation and
//! aggregates outcome distributions across many RNG seeds.

use std::{
    alloc::{self, Layout},
    env, fs,
    path::PathBuf,
    time::Instant,
};

use percolator::{NoOpMatcher, RiskEngine, RiskParams, I128, U128};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, LogNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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
    /// Number of slots where h <= 0.0 (complete insolvency)
    h_zero_slots: u64,
    /// First slot where h <= 0.0 (or u64::MAX if never)
    h_zero_first_slot: u64,
    /// Number of slots where h < 0.5
    h_below_50_slots: u64,
    /// Number of slots where h < 0.1
    h_below_10_slots: u64,
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

    /// Fraction of runs where h hit exactly 0.0 at any point
    insolvency_frac: f64,
    /// Among insolvent runs: median slots spent at h=0
    h_zero_slots_p50: f64,
    /// Among insolvent runs: median first slot where h=0
    h_zero_first_slot_p50: f64,
    /// Fraction of runs where h dipped below 0.5
    h_below_50_frac: f64,
    /// Fraction of runs where h dipped below 0.1
    h_below_10_frac: f64,
    /// Median slot where min_h occurred
    min_h_slot_p50: f64,
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
    if hd == 0 {
        1.0
    } else {
        hn as f64 / hd as f64
    }
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
    let matcher = NoOpMatcher;
    let p0 = price_e6(cfg.p0);

    // ── Build engine ────────────────────────────────────────────────────
    let params = RiskParams {
        warmup_period_slots: cfg.warmup_slots,
        maintenance_margin_bps: cfg.mm_bps,
        initial_margin_bps: cfg.im_bps,
        trading_fee_bps: cfg.trading_fee_bps,
        max_accounts: 4096,
        new_account_fee: U128::new(0),
        risk_reduction_threshold: U128::new(0),
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
    engine.deposit(lp_idx, usdc(cfg.lp_capital_usdc), 0).unwrap();

    // Initial crank at slot 0
    let _ = engine.keeper_crank(lp_idx, 0, p0, 0, false);

    // ── Capital distributions (lognormal mixtures) ──────────────────────
    let retail = LogNormal::new(2_000f64.ln(), 1.0).unwrap();
    let pro = LogNormal::new(50_000f64.ln(), 0.8).unwrap();
    let whale = LogNormal::new(1_000_000f64.ln(), 0.7).unwrap();

    // ── Add whale account if enabled ────────────────────────────────────
    let mut users: Vec<UserInfo> = Vec::with_capacity(cfg.n_users + 1);
    if cfg.whale_enabled {
        let whale_idx = engine.add_user(0).unwrap();
        engine.deposit(whale_idx, usdc(cfg.whale_capital_usdc), 0).unwrap();
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
        if engine.deposit(user_idx, usdc(cap_usdc), 0).is_err() {
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
        let _ = engine.keeper_crank(lp_idx, s, p0, 0, false);
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

        // position = notional * 1e6 / price
        let pos_abs = notional_atomic
            .saturating_mul(1_000_000)
            .checked_div(p0 as u128)
            .unwrap_or(1)
            .max(1);

        let size = if long {
            pos_abs as i128
        } else {
            -(pos_abs as i128)
        };

        // Retry with halved size on failure
        let mut s = size;
        for _ in 0..5 {
            if engine
                .execute_trade(&matcher, lp_idx, user.idx, trade_slot, p0, s)
                .is_ok()
            {
                user.had_position = true;
                break;
            }
            s /= 2;
            if s == 0 {
                break;
            }
        }
    }

    // ── Post-trade sweep ────────────────────────────────────────────────
    for s in (SETUP_SLOTS + 1)..=(SETUP_SLOTS + 32) {
        let _ = engine.keeper_crank(lp_idx, s, p0, 0, false);
    }

    // ── Inject zombies: positive PnL + fee debt ─────────────────────────
    let zombie_count = cfg.n_zombies.min(users.len());
    for _ in 0..zombie_count {
        let k = rng.gen_range(0..users.len());
        let idx = users[k].idx as usize;

        // Add positive realized PnL to zombie. Model zero-sum: the LP
        // was the counterparty and lost capital. Reduce LP capital so that
        // C_tot drops and Residual (V - C_tot - I) rises to back the PnL.
        // Any zombie PnL exceeding LP's remaining capital is unbacked gap
        // loss, which naturally collapses h.
        let add_pnl = usdc(cfg.zombie_pnl_usdc) as i128;
        let old_pnl = engine.accounts[idx].pnl.get();
        engine.set_pnl(idx, old_pnl.saturating_add(add_pnl));

        // LP counterparty loss (zero-sum backing)
        let lp_cap = engine.accounts[lp_idx as usize].capital.get();
        let loss = (add_pnl as u128).min(lp_cap);
        engine.set_capital(lp_idx as usize, lp_cap.saturating_sub(loss));

        // Create fee debt (push fee_credits negative)
        let debt = usdc(cfg.zombie_fee_debt_usdc) as i128;
        let old_credits = engine.accounts[idx].fee_credits.get();
        engine.accounts[idx].fee_credits = I128::new(old_credits.saturating_sub(debt));

        // Set warmup slope so crank can convert over time
        let _ = engine.update_warmup_slope(users[k].idx);
    }

    // ── Crash simulation ────────────────────────────────────────────────
    let crash_start = SETUP_SLOTS + 33;
    let mut min_h: f64 = f64::MAX;
    let mut min_h_slot: u64 = 0;
    let mut h_zero_slots: u64 = 0;
    let mut h_zero_first_slot: u64 = u64::MAX;
    let mut h_below_50_slots: u64 = 0;
    let mut h_below_10_slots: u64 = 0;
    let mut snapshots: Vec<SlotSnapshot> = Vec::new();

    let crank_every = cfg.crank_interval.max(1);

    for slot_offset in 0..cfg.total_slots {
        let slot = crash_start + slot_offset;
        let oracle = price_path(cfg, slot_offset);

        // Only crank every N slots to simulate keeper lag
        if slot_offset % crank_every == 0 {
            let _ = engine.keeper_crank(
                lp_idx,
                slot,
                oracle,
                cfg.funding_rate_bps_per_slot,
                false,
            );
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

        if cfg.snapshots && slot_offset % SNAPSHOT_INTERVAL == 0 {
            snapshots.push(SlotSnapshot {
                seed,
                slot: slot_offset,
                oracle_price: oracle,
                h,
                c_tot: engine.c_tot.get(),
                pnl_pos_tot: engine.pnl_pos_tot.get(),
                insurance: engine.insurance_fund.balance.get(),
                open_interest: engine.total_open_interest.get(),
                cum_liquidations: engine.lifetime_liquidations,
                cum_force_closes: engine.lifetime_force_realize_closes,
            });
        }
    }

    // ── End-of-run metrics ──────────────────────────────────────────────
    let final_h = haircut_f64(&engine);
    let final_oracle = price_path(cfg, cfg.total_slots.saturating_sub(1));

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

        // MTM equity (paper PnL — includes unrealized mark)
        let final_eq = engine.account_equity_mtm_at_oracle(acct, final_oracle) as f64;
        let mtm_ratio = if init > 0.0 { final_eq / init } else { 0.0 };
        capital_ratios.push(mtm_ratio);

        // Protected principal only (already safe, no warmup gate)
        let capital = acct.capital.get() as f64;
        let prin_ratio = if init > 0.0 { capital / init } else { 0.0 };
        principal_ratios.push(prin_ratio);

        // Withdrawable = capital + haircutted warmed-up PnL
        let warmed_pnl = engine.withdrawable_pnl(acct);
        let haircutted_pnl = if h_den > 0 {
            warmed_pnl.saturating_mul(h_num) / h_den
        } else {
            0
        };
        let withdrawable = acct.capital.get().saturating_add(haircutted_pnl) as f64;
        let wd_ratio = if init > 0.0 { withdrawable / init } else { 0.0 };
        withdrawable_ratios.push(wd_ratio);

        // Liquidated = had position, now closed, equity < 10% of initial
        if acct.position_size.get() == 0 && mtm_ratio < 0.1 {
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
        pnl_pos_tot_end: engine.pnl_pos_tot.get(),
        vault_end: engine.vault.get(),
        liquidations: engine.lifetime_liquidations,
        force_closes: engine.lifetime_force_realize_closes,
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

    // Insolvency tracking
    let insolvent_runs: Vec<&RunSummary> = runs.iter().filter(|r| r.h_zero_slots > 0).collect();
    let insolvency_frac = insolvent_runs.len() as f64 / runs.len().max(1) as f64;
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

        insolvency_frac,
        h_zero_slots_p50: quantile(&h_zero_slots_sorted, 0.50),
        h_zero_first_slot_p50: quantile(&h_zero_first_sorted, 0.50),
        h_below_50_frac,
        h_below_10_frac,
        min_h_slot_p50: quantile(&min_h_slots, 0.50),
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
         h_zero_slots,h_zero_first_slot,h_below_50_slots,h_below_10_slots\n",
    );
    for r in &runs {
        csv.push_str(&format!(
            "{},{:.6},{},{:.6},{},{},{},{},{},{},{},{},{},{},{}\n",
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
// Main
// ════════════════════════════════════════════════════════════════════════════

fn main() {
    if env::args().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }

    let cfg = parse_args();
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
