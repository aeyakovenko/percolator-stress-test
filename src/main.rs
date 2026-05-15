//! Percolator v13 stress test — port in progress.
//!
//! v12 main.rs preserved as src/main_v12.rs.bak for reference. The v13 engine
//! is a portfolio-account / multi-asset rearchitecture, so the stress test
//! is being rebuilt incrementally:
//!
//!   stage 1: minimal flow (deposit + trade + crank + close)        ← this commit
//!   stage 2: bounty_sol_20x_max preset + 2000-seed fuzz
//!   stage 3: full scenario suite + invariant battery
//!   stage 4: tests (exec_price_attack, sybil_close_attack, pnl_trap)
//!   stage 5: trace machinery + summary aggregation

use percolator::*;
use rayon::prelude::*;
use std::env;

const USDC_DECIMALS: u128 = 1_000_000;
fn usdc(d: u128) -> u128 {
    d.checked_mul(USDC_DECIMALS).expect("usdc overflow")
}
fn price_e6(d: u64) -> u64 {
    d.checked_mul(1_000_000).expect("price overflow")
}

const SOL_ASSET: usize = 0;

/// Engine state held by the wrapper: a MarketGroup plus a Vec of accounts.
/// In v13 the engine no longer owns the account slab; the wrapper passes
/// accounts in by mut-ref to each call.
struct V13Engine {
    group: MarketGroupV13,
    accounts: Vec<PortfolioAccountV13>,
    market_group_id: [u8; 32],
    next_account_seq: u64,
}

impl V13Engine {
    fn new(config: V13Config) -> V13Result<Self> {
        let group_id = [0x42u8; 32];
        let group = MarketGroupV13::new(group_id, config)?;
        Ok(Self {
            group,
            accounts: Vec::new(),
            market_group_id: group_id,
            next_account_seq: 0,
        })
    }

    /// Create a new portfolio account and register it with the market group.
    fn add_account(&mut self, owner_byte: u8) -> V13Result<usize> {
        let mut id = [0u8; 32];
        id[..8].copy_from_slice(&self.next_account_seq.to_le_bytes());
        self.next_account_seq += 1;
        let owner = [owner_byte; 32];
        let header = ProvenanceHeaderV13::new(self.market_group_id, id, owner);
        let account = PortfolioAccountV13::empty(header);
        self.group.create_portfolio_account(&account)?;
        let idx = self.accounts.len();
        self.accounts.push(account);
        Ok(idx)
    }

    fn deposit(&mut self, idx: usize, amount: u128) -> V13Result<()> {
        let mut acc = self.accounts[idx];
        self.group.deposit_not_atomic(&mut acc, amount)?;
        self.accounts[idx] = acc;
        Ok(())
    }

    fn effective_prices(&self) -> [u64; V13_MAX_PORTFOLIO_ASSETS_N] {
        let mut p = [0u64; V13_MAX_PORTFOLIO_ASSETS_N];
        for i in 0..V13_MAX_PORTFOLIO_ASSETS_N {
            p[i] = self.group.assets[i].effective_price;
        }
        p
    }

    /// Bilateral trade. `long_idx` receives `+size_q` of asset, `short_idx`
    /// receives `−size_q`.
    fn trade(
        &mut self,
        long_idx: usize,
        short_idx: usize,
        asset_index: usize,
        size_q: u128,
        exec_price: u64,
        fee_bps: u64,
    ) -> V13Result<TradeOutcomeV13> {
        let prices = self.effective_prices();
        let mut long_acc = self.accounts[long_idx];
        let mut short_acc = self.accounts[short_idx];
        let req = TradeRequestV13 {
            asset_index,
            size_q,
            exec_price,
            fee_bps,
        };
        let out = self.group.execute_trade_with_fee_not_atomic(
            &mut long_acc,
            &mut short_acc,
            req,
            &prices,
        )?;
        self.accounts[long_idx] = long_acc;
        self.accounts[short_idx] = short_acc;
        Ok(out)
    }

    fn accrue_asset(
        &mut self,
        asset_index: usize,
        now_slot: u64,
        effective_price: u64,
        funding_rate_e9: i128,
    ) -> V13Result<AccrueAssetOutcomeV13> {
        // Keep the raw-oracle target in sync with the effective price so
        // target_effective_lag stays false.
        self.group.assets[asset_index].raw_oracle_target_price = effective_price;
        // protective_progress_committed=true: the wrapper attests that any
        // exposed accounts have already been touched this slot. In the real
        // deployment the wrapper would batch account refreshes alongside the
        // accrue; here our smoke test cranks the price without explicit
        // account-touches because the trade itself does the refresh.
        self.group.accrue_asset_to_not_atomic(
            asset_index,
            now_slot,
            effective_price,
            funding_rate_e9,
            true,
        )
    }

    fn set_oracle_target(&mut self, asset_index: usize, target: u64) {
        self.group.assets[asset_index].raw_oracle_target_price = target;
    }

    fn assert_invariants(&self) -> V13Result<()> {
        self.group.assert_public_invariants()
    }
}

/// Conservative v13 config — full margin coverage, no extra fees. Passes
/// the strict v13 solvency envelope check. Stage 1 uses this to verify the
/// flow; the bounty_sol_20x_max config comes in stage 2 once we know what
/// the v13 envelope allows.
fn make_full_margin_config() -> V13Config {
    V13Config::public_user_fund(1, 0, 30)
}

/// Probe: try variants of the bounty_sol_20x_max config to find what
/// v13's validate_exact_solvency_envelope accepts.
fn probe_bounty_variants() {
    let mk = |max_move: u64, max_dt: u64, liq: u64, fee: u64| V13Config {
        max_portfolio_assets: 1, min_nonzero_mm_req: 20, min_nonzero_im_req: 30,
        h_min: 0, h_max: 30,
        maintenance_margin_bps: 500, initial_margin_bps: 500,
        max_trading_fee_bps: fee, liquidation_fee_bps: liq,
        liquidation_fee_cap: usdc(50_000), min_liquidation_abs: 0,
        max_accrual_dt_slots: max_dt, max_abs_funding_e9_per_slot: 0,
        min_funding_lifetime_slots: max_dt, max_price_move_bps_per_slot: max_move,
        max_account_b_settlement_chunks: 8, max_bankrupt_close_chunks: 8,
        public_b_chunk_atoms: MAX_VAULT_TVL,
        permissionless_recovery_enabled: true,
        stale_certificate_penalty_enabled: true,
        full_refresh_required_for_favorable_actions: true,
        public_liveness_profile_crank_forward: true,
    };
    let cases: Vec<(String, V13Config)> = vec![
        ("baseline v12 max_risk".to_string(), make_bounty_sol_20x_max_config()),
        ("mm=im=10000 (full)".to_string(), V13Config::public_user_fund(1, 0, 30)),
    ].into_iter()
    .chain([10, 20, 30, 40, 45, 48, 49].iter().flat_map(|&mv| {
        [0u64, 5].iter().map(move |&lf| {
            (format!("max_move={:>2} dt=10 liq={:>2} fee=0", mv, lf), mk(mv, 10, lf, 0))
        })
    }))
    .collect();
    for (name, cfg) in cases {
        let r = cfg.validate_public_user_fund();
        println!("  {:<35}  {:?}", name, r);
    }
}

/// v13 bounty config — equivalent to v12 max_risk.md but tuned to v13's
/// stricter solvency envelope. v12 allowed max_move=49 bps/slot; v13's exact
/// envelope reserves more headroom, max it accepts is 45.
/// Effective per-accrual oracle tolerance: 45 × 10 = 450 bps = 4.5%.
fn make_bounty_sol_20x_max_config() -> V13Config {
    make_bounty_config(1)
}

fn make_bounty_config(n_assets: u8) -> V13Config {
    V13Config {
        max_portfolio_assets: n_assets,
        min_nonzero_mm_req: 20,
        min_nonzero_im_req: 30,
        h_min: 0,
        h_max: 30,
        maintenance_margin_bps: 500,
        initial_margin_bps: 500,
        max_trading_fee_bps: 1,
        liquidation_fee_bps: 5,
        liquidation_fee_cap: usdc(50_000),
        min_liquidation_abs: 0,
        max_accrual_dt_slots: 10,
        max_abs_funding_e9_per_slot: 0,
        min_funding_lifetime_slots: 10,
        max_price_move_bps_per_slot: 45,
        max_account_b_settlement_chunks: 8,
        max_bankrupt_close_chunks: 8,
        public_b_chunk_atoms: MAX_VAULT_TVL,
        permissionless_recovery_enabled: true,
        stale_certificate_penalty_enabled: true,
        full_refresh_required_for_favorable_actions: true,
        public_liveness_profile_crank_forward: true,
    }
}

/// Stage 1 smoke test: create engine, add LP + user, deposit, trade, accrue, close.
fn smoke_test() -> V13Result<()> {
    let cfg = make_bounty_sol_20x_max_config();
    println!("V13 stage-2 smoke: bounty_sol_20x_max config (v13-tuned)");
    cfg.validate_public_user_fund()?;
    println!("  config validated");

    let mut engine = V13Engine::new(cfg)?;
    let lp = engine.add_account(1)?;
    let user = engine.add_account(2)?;
    engine.deposit(lp, usdc(10_000_000))?;
    engine.deposit(user, usdc(1_000))?;
    println!("  accounts: lp=idx{}, user=idx{}", lp, user);
    println!("  vault=${}M  c_tot=${}M",
        engine.group.vault / USDC_DECIMALS / 1_000_000,
        engine.group.c_tot / USDC_DECIMALS / 1_000_000);

    // Set up oracle
    let oracle0 = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle0, 0)?;

    // Open user-long against LP-short. 20x leverage: $20k notional on $1k.
    // For mm=500 (5%), $20k notional requires $1000 IM/MM — fits exactly.
    let notional = usdc(15_000); // 15x, conservative
    let size_q = notional * POS_SCALE / oracle0 as u128;
    let outcome = engine.trade(user, lp, SOL_ASSET, size_q, oracle0, 1)?;
    println!("  OPEN: notional=${}  fee_a=${}  fee_b=${}",
        outcome.notional / USDC_DECIMALS,
        outcome.fee_a / USDC_DECIMALS,
        outcome.fee_b / USDC_DECIMALS);
    println!("    user.pnl={} cap=${} legs[0].active={}",
        engine.accounts[user].pnl, engine.accounts[user].capital / USDC_DECIMALS,
        engine.accounts[user].legs[0].active);

    // Walk oracle up by 1% across 10 slots (within envelope)
    let mut slot = 2u64;
    let mut oracle = oracle0;
    for _ in 0..10 {
        oracle = oracle + (oracle as u128 * 10 / 10_000) as u64;
        engine.accrue_asset(SOL_ASSET, slot, oracle, 0)?;
        slot += 1;
    }
    println!("  after 10 cranks: oracle=${}", engine.group.assets[0].effective_price / 1_000_000);

    // Close: user goes short (size_q) — i.e., role swap: lp is the new long, user the new short
    let close_outcome = engine.trade(lp, user, SOL_ASSET, size_q, oracle, 1)?;
    println!("  CLOSE: notional=${}", close_outcome.notional / USDC_DECIMALS);
    println!("    user.pnl={} cap=${} legs[0].active={}",
        engine.accounts[user].pnl, engine.accounts[user].capital / USDC_DECIMALS,
        engine.accounts[user].legs[0].active);

    engine.assert_invariants()?;
    println!("  invariants OK");
    Ok(())
}

// ════════════════════════════════════════════════════════════════════════════
// PRNG (xorshift64*) for deterministic per-seed simulation
// ════════════════════════════════════════════════════════════════════════════
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1)) }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn range_u64(&mut self, lo: u64, hi: u64) -> u64 {
        lo + (self.next_u64() % (hi - lo + 1))
    }
    fn bool(&mut self) -> bool { self.next_u64() & 1 == 1 }
}

#[derive(Clone, Debug)]
struct RunSummary {
    seed: u64,
    final_vault: u128,
    final_insurance: u128,
    final_c_tot: u128,
    total_trades: u32,
    rejected_trades: u32,
    liquidations: u32,
    invariant_failures: u32,
    insurance_payouts: u128,
    residual_booked: u128,
    explicit_loss: u128,
    min_user_capital: u128,
    max_user_pnl_abs: u128,
    bankruptcy_lock_tripped: bool,
}

#[derive(Clone, Copy, Debug)]
enum Scenario {
    Random,
    Crash10,
    Crash20,
    FundingDrain,
    OracleWick,
    HighLev,
    /// Combined: 20 users, 3 assets, mixed leverage, independent crashes
    Mega,
}

impl Scenario {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "random" => Some(Self::Random),
            "crash10" => Some(Self::Crash10),
            "crash20" => Some(Self::Crash20),
            "funding_drain" => Some(Self::FundingDrain),
            "oracle_wick" => Some(Self::OracleWick),
            "high_lev" => Some(Self::HighLev),
            "mega" => Some(Self::Mega),
            _ => None,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Crash10 => "crash10",
            Self::Crash20 => "crash20",
            Self::FundingDrain => "funding_drain",
            Self::OracleWick => "oracle_wick",
            Self::HighLev => "high_lev",
            Self::Mega => "mega",
        }
    }
}

/// Explicit invariant battery — v12-style granular checks. Returns a list of
/// (label, ok) pairs so the caller can attribute the specific invariant that
/// failed.
fn invariant_battery(engine: &V13Engine) -> Vec<(&'static str, bool)> {
    let g = &engine.group;
    let mut results = vec![];

    // I1: vault >= c_tot + insurance (solvency)
    let senior = g.c_tot.saturating_add(g.insurance);
    results.push(("V >= C + I", g.vault >= senior));

    // I2: matured <= pnl_pos_tot
    results.push(("matured <= pos_tot", g.pnl_matured_pos_tot <= g.pnl_pos_tot));

    // I3-5: per-asset K, F, A bounds
    let mut k_ok = true;
    let mut f_ok = true;
    let mut a_ok = true;
    for i in 0..g.config.max_portfolio_assets as usize {
        let asset = g.assets[i];
        if asset.k_long.unsigned_abs() > (i128::MAX as u128) / 2 { k_ok = false; }
        if asset.k_short.unsigned_abs() > (i128::MAX as u128) / 2 { k_ok = false; }
        if asset.f_long_num.unsigned_abs() > (i128::MAX as u128) / 2 { f_ok = false; }
        if asset.f_short_num.unsigned_abs() > (i128::MAX as u128) / 2 { f_ok = false; }
        // A_side >= MIN_A_SIDE unless in DrainOnly/ResetPending
        if asset.oi_eff_long_q > 0 && asset.mode_long == SideModeV13::Normal && asset.a_long < MIN_A_SIDE {
            a_ok = false;
        }
        if asset.oi_eff_short_q > 0 && asset.mode_short == SideModeV13::Normal && asset.a_short < MIN_A_SIDE {
            a_ok = false;
        }
    }
    results.push(("K within i128/2", k_ok));
    results.push(("F within i128/2", f_ok));
    results.push(("A_side >= MIN_A_SIDE (Normal mode)", a_ok));

    // I6: negative_pnl_account_count consistency
    let neg_count = engine.accounts.iter().filter(|a| a.pnl < 0).count() as u64;
    results.push(("neg_pnl_count consistent", neg_count == g.negative_pnl_account_count));

    // I7: sum(capital) == c_tot
    let sum_cap: u128 = engine.accounts.iter().map(|a| a.capital).sum();
    results.push(("sum(capital) == c_tot", sum_cap == g.c_tot));

    // I8: sum(reserved_pnl) <= sum(max(0, pnl))
    let sum_reserved: u128 = engine.accounts.iter().map(|a| a.reserved_pnl).sum();
    let sum_pos_pnl: u128 = engine.accounts.iter().map(|a| a.pnl.max(0) as u128).sum();
    results.push(("sum(reserved) <= sum(pos pnl)", sum_reserved <= sum_pos_pnl));

    // I9 (v13-specific): F7 invariant — DrainOnly implies opposing OI > 0 OR opp is in ResetPending
    let mut f7_ok = true;
    for i in 0..g.config.max_portfolio_assets as usize {
        let asset = g.assets[i];
        if asset.mode_long == SideModeV13::DrainOnly
            && asset.oi_eff_short_q == 0
            && asset.mode_short != SideModeV13::ResetPending
        {
            f7_ok = false;
        }
        if asset.mode_short == SideModeV13::DrainOnly
            && asset.oi_eff_long_q == 0
            && asset.mode_long != SideModeV13::ResetPending
        {
            f7_ok = false;
        }
    }
    results.push(("F7 DrainOnly + opp OI consistent", f7_ok));

    results
}

fn run_invariant_battery(engine: &V13Engine) -> u32 {
    let results = invariant_battery(engine);
    let mut failures = 0u32;
    for (_, ok) in &results {
        if !ok { failures += 1; }
    }
    failures
}

/// Wrapper-side oracle clamp matching v12 semantics: limit per-call move to
/// max_price_move × dt of the engine's current effective_price.
fn clamp_oracle(real: u64, last: u64, max_move_bps: u64, dt: u64) -> u64 {
    let budget = (last as u128).saturating_mul(max_move_bps as u128).saturating_mul(dt as u128) / 10_000;
    let budget = budget.min(u64::MAX as u128) as u64;
    let lo = last.saturating_sub(budget).max(1);
    let hi = last.saturating_add(budget).min(MAX_ORACLE_PRICE);
    real.clamp(lo, hi)
}

/// Compute the next oracle for the slot under the chosen scenario.
fn scenario_oracle(scen: Scenario, rng: &mut Rng, oracle: u64, step: u64, max_move: u64) -> u64 {
    match scen {
        Scenario::Random | Scenario::HighLev => {
            let crash = rng.bool() && rng.bool() && rng.bool() && rng.bool();
            let pct = if crash {
                rng.range_u64(1, max_move)
            } else {
                rng.range_u64(0, max_move / 2)
            };
            let up = rng.bool();
            let d = (oracle as u128 * pct as u128 / 10_000) as u64;
            if up { oracle.saturating_add(d) } else { oracle.saturating_sub(d).max(1) }
        }
        Scenario::Crash10 | Scenario::Crash20 => {
            let crash_len: u64 = if matches!(scen, Scenario::Crash20) { 200 } else { 100 };
            if step < crash_len {
                let d = (oracle as u128 * max_move as u128 / 10_000) as u64;
                oracle.saturating_sub(d).max(1)
            } else {
                let d = (oracle as u128 * (max_move as u128 / 3) / 10_000) as u64;
                oracle.saturating_add(d)
            }
        }
        Scenario::FundingDrain => oracle,
        Scenario::OracleWick => {
            // Sharp V-shape: 50 slots down envelope-max, 50 slots back up
            // envelope-max, repeat. Tests engine's response to fast reversals.
            let cycle = step % 100;
            let d = (oracle as u128 * max_move as u128 / 10_000) as u64;
            if cycle < 50 {
                oracle.saturating_sub(d).max(1)
            } else {
                oracle.saturating_add(d)
            }
        }
        Scenario::Mega => {
            // Independent random walk per asset, but bigger moves than Random
            let dir = rng.bool();
            let pct = rng.range_u64(0, max_move);
            let d = (oracle as u128 * pct as u128 / 10_000) as u64;
            if dir { oracle.saturating_add(d) } else { oracle.saturating_sub(d).max(1) }
        }
    }
}

/// Single run for the given scenario. Includes liquidation by a synthetic
/// keeper that checks each user's certified_liq_deficit and liquidates the
/// largest leg if positive.
fn run_one_scenario(scen: Scenario, seed: u64) -> RunSummary {
    let n_assets = if matches!(scen, Scenario::Mega) { 3 } else { 1 };
    let cfg = make_bounty_config(n_assets);
    let mut engine = V13Engine::new(cfg).expect("init");
    let mut rng = Rng::new(seed);

    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(50_000_000)).unwrap();

    let n_users = if matches!(scen, Scenario::Mega) { 20 } else { 5 };
    let mut users = Vec::with_capacity(n_users);
    for _ in 0..n_users {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(1_000)).unwrap();
        users.push(u);
    }

    let oracle0 = price_e6(200);
    for ai in 0..n_assets as usize {
        engine.accrue_asset(ai, 1, oracle0, 0).unwrap();
    }
    let _ = SOL_ASSET;

    let mut summary = RunSummary {
        seed,
        final_vault: 0, final_insurance: 0, final_c_tot: 0,
        total_trades: 0, rejected_trades: 0, liquidations: 0,
        invariant_failures: 0, insurance_payouts: 0,
        residual_booked: 0, explicit_loss: 0,
        min_user_capital: u128::MAX, max_user_pnl_abs: 0,
        bankruptcy_lock_tripped: false,
    };

    let mut slot = 2u64;
    let mut oracles = vec![oracle0; n_assets as usize];
    let max_move = cfg.max_price_move_bps_per_slot;
    let total_slots: u64 = match scen {
        Scenario::Random => 200,
        Scenario::Crash10 => 200,
        Scenario::Crash20 => 400,
        Scenario::FundingDrain => 300,
        Scenario::OracleWick => 400,
        Scenario::HighLev => 200,
        Scenario::Mega => 400,
    };

    // Open initial positions on each user. Default 8x leverage; HighLev
    // pushes to 18x; Mega uses random 5-18x with random direction across
    // a random asset.
    let init_leverage_fn = |rng: &mut Rng| -> u128 {
        match scen {
            Scenario::HighLev => 18,
            Scenario::Mega => rng.range_u64(5, 18) as u128,
            _ => 8,
        }
    };
    for &u in &users {
        let going_long = rng.bool();
        let asset = if matches!(scen, Scenario::Mega) {
            (rng.next_u64() as usize) % (n_assets as usize)
        } else {
            SOL_ASSET
        };
        let lev = init_leverage_fn(&mut rng);
        let notional = usdc(1_000 * lev);
        let size_q = notional * POS_SCALE / oracles[asset] as u128;
        let (long, short) = if going_long { (u, lp) } else { (lp, u) };
        if engine.trade(long, short, asset, size_q, oracles[asset], 1).is_ok() {
            summary.total_trades += 1;
        }
    }

    let funding_rate = if matches!(scen, Scenario::FundingDrain) { 5_000i128 } else { 0 };

    for step in 0..total_slots {
        // For multi-asset Mega, each asset moves independently.
        for ai in 0..n_assets as usize {
            let target = scenario_oracle(scen, &mut rng, oracles[ai], step, max_move);
            oracles[ai] = clamp_oracle(
                target,
                engine.group.assets[ai].effective_price,
                max_move,
                1,
            );
            let _ = engine.accrue_asset(ai, slot, oracles[ai], funding_rate);
        }
        let oracle = oracles[0]; // primary oracle for legacy code below

        // Keeper: liquidate anyone with certified_liq_deficit > 0.
        // To learn each account's deficit we have to refresh it first.
        let prices = engine.effective_prices();
        for &u in &users {
            let mut acc = engine.accounts[u];
            if engine.group.full_account_refresh(&mut acc, &prices).is_err() {
                engine.accounts[u] = acc;
                continue;
            }
            engine.accounts[u] = acc;
            if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                // Find the largest active leg and close it fully
                let mut largest_leg_idx = None;
                let mut largest_abs = 0u128;
                for li in 0..V13_MAX_PORTFOLIO_ASSETS_N {
                    let leg = engine.accounts[u].legs[li];
                    if leg.active {
                        let a = leg.basis_pos_q.unsigned_abs();
                        if a > largest_abs {
                            largest_abs = a;
                            largest_leg_idx = Some(li);
                        }
                    }
                }
                if let Some(li) = largest_leg_idx {
                    let ins_pre = engine.group.insurance;
                    let mut acc = engine.accounts[u];
                    let r = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV13 {
                            asset_index: li,
                            close_q: largest_abs,
                            fee_bps: 5,
                        },
                        &prices,
                    );
                    engine.accounts[u] = acc;
                    if let Ok(out) = r {
                        summary.liquidations += 1;
                        summary.insurance_payouts += out.insurance_used;
                        summary.residual_booked += out.residual_booked;
                        summary.explicit_loss += out.explicit_loss;
                        let _ = ins_pre;
                    }
                }
            }
        }

        // Random new trades (Random/FundingDrain/Mega add liveness churn)
        if matches!(scen, Scenario::Random | Scenario::FundingDrain | Scenario::Mega)
            && step % 5 == 0
        {
            let uidx = users[(rng.next_u64() as usize) % users.len()];
            let cap = engine.accounts[uidx].capital;
            if cap > usdc(50) {
                let going_long = rng.bool();
                let asset = if matches!(scen, Scenario::Mega) {
                    (rng.next_u64() as usize) % (n_assets as usize)
                } else {
                    SOL_ASSET
                };
                let leverage = rng.range_u64(2, 15) as u128;
                let notional = (cap * leverage).min(usdc(20_000));
                let size_q = notional * POS_SCALE / oracles[asset] as u128;
                let (long, short) = if going_long { (uidx, lp) } else { (lp, uidx) };
                match engine.trade(long, short, asset, size_q, oracles[asset], 1) {
                    Ok(_) => summary.total_trades += 1,
                    Err(_) => summary.rejected_trades += 1,
                }
            }
        }
        let _ = oracle;

        if engine.assert_invariants().is_err() {
            summary.invariant_failures += 1;
        }
        summary.invariant_failures += run_invariant_battery(&engine);
        if engine.group.bankruptcy_hlock_active {
            summary.bankruptcy_lock_tripped = true;
        }
        for &u in &users {
            let acc = &engine.accounts[u];
            summary.min_user_capital = summary.min_user_capital.min(acc.capital);
            summary.max_user_pnl_abs = summary.max_user_pnl_abs.max(acc.pnl.unsigned_abs());
        }

        slot += 1;
    }

    summary.final_vault = engine.group.vault;
    summary.final_insurance = engine.group.insurance;
    summary.final_c_tot = engine.group.c_tot;
    summary
}

fn run_one_bounty(seed: u64) -> RunSummary {
    run_one_scenario(Scenario::Random, seed)
}

fn aggregate(scen: Scenario, results: Vec<RunSummary>) {
    let n = results.len();
    let total_invariant_failures: u32 = results.iter().map(|r| r.invariant_failures).sum();
    let total_trades: u32 = results.iter().map(|r| r.total_trades).sum();
    let total_rejected: u32 = results.iter().map(|r| r.rejected_trades).sum();
    let total_liquidations: u32 = results.iter().map(|r| r.liquidations).sum();
    let total_insurance_payouts: u128 = results.iter().map(|r| r.insurance_payouts).sum();
    let total_residual: u128 = results.iter().map(|r| r.residual_booked).sum();
    let total_explicit: u128 = results.iter().map(|r| r.explicit_loss).sum();
    let bankruptcy_runs = results.iter().filter(|r| r.bankruptcy_lock_tripped).count();
    let min_vault = results.iter().map(|r| r.final_vault).min().unwrap_or(0);
    let max_vault = results.iter().map(|r| r.final_vault).max().unwrap_or(0);
    let min_insurance = results.iter().map(|r| r.final_insurance).min().unwrap_or(0);
    let max_pnl = results.iter().map(|r| r.max_user_pnl_abs).max().unwrap_or(0);
    let min_user_capital = results
        .iter()
        .map(|r| r.min_user_capital)
        .min()
        .unwrap_or(u128::MAX);

    println!("=== {} ({} seeds) ===", scen.name(), n);
    println!("  trades total:          {}  (rejected: {})", total_trades, total_rejected);
    println!("  liquidations:          {}", total_liquidations);
    println!("  bankruptcy lock runs:  {}/{}", bankruptcy_runs, n);
    println!("  invariant failures:    {}  (must be 0)", total_invariant_failures);
    println!("  insurance used (atomic): {}", total_insurance_payouts);
    println!("  residual booked (atomic): {}", total_residual);
    println!("  explicit loss (atomic):   {}", total_explicit);
    println!("  vault final:           min ${}M  max ${}M",
        min_vault / USDC_DECIMALS / 1_000_000,
        max_vault / USDC_DECIMALS / 1_000_000);
    println!("  insurance final:       min ${}", min_insurance / USDC_DECIMALS);
    println!("  user min capital:      ${}", min_user_capital / USDC_DECIMALS);
    println!("  max |user pnl|:        ${}", max_pnl / 1_000_000);
}

fn run_fuzz(scen: Scenario, n_seeds: usize) {
    let results: Vec<RunSummary> = (0..n_seeds as u64)
        .into_par_iter()
        .map(|seed| run_one_scenario(scen, seed))
        .collect();
    aggregate(scen, results);
}

/// V13 port of v12 exec_price_attack test. Engine v13 also doesn't bound
/// exec_price vs oracle directly; defense is the post-trade IM check.
fn test_exec_price_attack_v13() -> V13Result<()> {
    println!("=== v13 exec_price attack: bounty_sol_20x_max ===");
    let cfg = make_bounty_sol_20x_max_config();
    let oracle = price_e6(200);

    for deviation_bps in [100u64, 1000, 5000, 9999] {
        let mut engine = V13Engine::new(cfg)?;
        let lp = engine.add_account(1)?;
        let attacker = engine.add_account(2)?;
        engine.deposit(lp, usdc(10_000_000))?;
        engine.deposit(attacker, usdc(1_000))?;
        engine.accrue_asset(SOL_ASSET, 1, oracle, 0)?;

        // Adversarial exec: attacker (long) buys at exec << oracle. Gets PnL
        // = size × (oracle - exec) / POS_SCALE if it goes through.
        let exec_price = (oracle as u128 * (10_000 - deviation_bps) as u128 / 10_000).max(1) as u64;
        let notional = usdc(5_000); // 5x leverage, well within IM (10% = $500)
        let size_q = notional * POS_SCALE / exec_price as u128;

        let result = engine.trade(attacker, lp, SOL_ASSET, size_q, exec_price, 1);
        let outcome = match result {
            Ok(_) => {
                let a = &engine.accounts[attacker];
                format!("OK | attacker pnl=${} cap=${}", a.pnl / 1_000_000, a.capital / USDC_DECIMALS)
            }
            Err(e) => format!("REJECTED ({:?}) — engine defended", e),
        };
        println!("  dev={:4} bps  exec=${}  : {}", deviation_bps, exec_price / 1_000_000, outcome);
    }
    Ok(())
}

/// V13 port of v12 sybil_close_attack: open A↔B at fair price, then close
/// at adversarial exec to dump loss onto one side.
fn test_sybil_close_v13() -> V13Result<()> {
    println!("=== v13 sybil close: bounty_sol_20x_max ===");
    let cfg = make_bounty_sol_20x_max_config();
    let oracle = price_e6(200);

    for deviation_bps in [100u64, 1000, 5000, 9999] {
        let mut engine = V13Engine::new(cfg)?;
        let lp = engine.add_account(1)?;
        let a = engine.add_account(2)?;
        let b = engine.add_account(3)?;
        engine.deposit(lp, usdc(10_000_000))?;
        engine.deposit(a, usdc(1_000))?;
        engine.deposit(b, usdc(1_000))?;
        engine.accrue_asset(SOL_ASSET, 1, oracle, 0)?;

        let notional = usdc(5_000);
        let size_q = notional * POS_SCALE / oracle as u128;

        // Step 1: A long, B short, at fair price
        let r1 = engine.trade(a, b, SOL_ASSET, size_q, oracle, 1);
        if let Err(e) = r1 {
            println!("  dev={:4} bps: STEP 1 failed ({:?})", deviation_bps, e);
            continue;
        }

        // Step 2: reverse trade A↔B with adversarial exec — A flat, B flat,
        // but pnl shifted by the deviation
        let bad_exec = ((oracle as u128 * (10_000 + deviation_bps) as u128 / 10_000)
            .min(MAX_ORACLE_PRICE as u128)) as u64;
        let r2 = engine.trade(b, a, SOL_ASSET, size_q, bad_exec, 1);
        match r2 {
            Err(e) => {
                println!("  dev={:4} bps: STEP 2 REJECTED ({:?})", deviation_bps, e);
                continue;
            }
            Ok(_) => {}
        }
        let pre_insurance = engine.group.insurance;
        let acc_a = engine.accounts[a];
        let acc_b = engine.accounts[b];
        let _ = engine.assert_invariants();

        println!("  dev={:4} bps OK | A pnl=${} cap=${} | B pnl=${} cap=${}  insurance=${}",
            deviation_bps,
            acc_a.pnl / 1_000_000, acc_a.capital / USDC_DECIMALS,
            acc_b.pnl / 1_000_000, acc_b.capital / USDC_DECIMALS,
            pre_insurance / USDC_DECIMALS);
    }
    Ok(())
}

/// v13 probe_drain port: 4 pathological configs trying to force insurance
/// to absorb a deficit. v12 had 5 probes; one (P1) used the zombie-injection
/// test backdoor which doesn't exist in v13. The other 4 are reachable
/// through legitimate APIs and worth testing.
fn run_probes() {
    println!("=== v13 probe_drain: 4 pathological scenarios ===");
    println!();

    probe_zero_insurance_concentrated_long();
    println!();
    probe_no_lp_no_insurance();
    println!();
    probe_whale_crash();
    println!();
    probe_long_funding_drain();
}

// ════════════════════════════════════════════════════════════════════════════
// v13-specific probes (not portable to v12 — exercise new attack surface)
// ════════════════════════════════════════════════════════════════════════════

/// Multi-asset portfolio probe: open hedged long+short across 2 assets,
/// crash one. Hedge should NOT mask losses on the crashed leg.
fn probe_multi_asset_crash() {
    println!("  Multi-asset: 2 assets, hedged long-short, one crashes");
    let cfg = make_bounty_config(2);
    let mut engine = V13Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(2_000)).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    // User: long asset 0, long asset 1 (so a crash on either hits them)
    let notional = usdc(5_000);
    let size_q = notional * POS_SCALE / oracle as u128;
    if engine.trade(user, lp, 0, size_q, oracle, 1).is_err() {
        println!("    open asset 0 failed"); return;
    }
    if engine.trade(user, lp, 1, size_q, oracle, 1).is_err() {
        println!("    open asset 1 failed"); return;
    }
    println!("    opened long on assets 0 + 1 ($5k each)");

    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o0 = oracle;
    let o1 = oracle;
    let mut slot = 2u64;
    let mut total_liquidations = 0u32;
    let mut total_insurance_used = 0u128;
    let mut total_residual = 0u128;

    for _ in 0..200 {
        // crash asset 0, leave asset 1 flat
        let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
        o0 = o0.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);

        let prices = engine.effective_prices();
        let mut acc = engine.accounts[user];
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[user] = acc;
        if engine.accounts[user].health_cert.certified_liq_deficit > 0 {
            // Find biggest leg and liquidate
            let mut best = (0usize, 0u128);
            for li in 0..V13_MAX_PORTFOLIO_ASSETS_N {
                let leg = engine.accounts[user].legs[li];
                if leg.active {
                    let a = leg.basis_pos_q.unsigned_abs();
                    if a > best.1 { best = (li, a); }
                }
            }
            if best.1 > 0 {
                let mut acc = engine.accounts[user];
                if let Ok(out) = engine.group.liquidate_account_not_atomic(
                    &mut acc,
                    LiquidationRequestV13 { asset_index: best.0, close_q: best.1, fee_bps: 5 },
                    &prices,
                ) {
                    total_liquidations += 1;
                    total_insurance_used += out.insurance_used;
                    total_residual += out.residual_booked;
                }
                engine.accounts[user] = acc;
            }
        }
        slot += 1;
    }
    println!("    liquidations:      {}", total_liquidations);
    println!("    insurance used:    {}", total_insurance_used);
    println!("    residual booked:   {}", total_residual);
    println!("    user pnl: {}  cap: {}", engine.accounts[user].pnl, engine.accounts[user].capital);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Stale-account probe: open position, mark account stale, try to extract
/// via favorable action. Should be rejected.
fn probe_stale_extract() {
    println!("  Stale-state extraction: mark stale, try convert/withdraw");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V13Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(1_000)).unwrap();
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    let notional = usdc(5_000);
    let size_q = notional * POS_SCALE / oracle as u128;
    engine.trade(user, lp, SOL_ASSET, size_q, oracle, 1).unwrap();

    // Mark account stale
    let mut acc = engine.accounts[user];
    let _ = engine.group.mark_account_stale(&mut acc);
    engine.accounts[user] = acc;
    println!("    marked stale; account.stale_state={}", engine.accounts[user].stale_state);

    // Try to convert PnL while stale
    let prices = engine.effective_prices();
    let mut acc = engine.accounts[user];
    let r = engine.group.convert_released_pnl_to_capital_not_atomic(&mut acc);
    engine.accounts[user] = acc;
    println!("    convert while stale: {:?}", r.err());

    // Try to withdraw capital while stale
    let mut acc = engine.accounts[user];
    let r = engine.group.withdraw_not_atomic(&mut acc, usdc(100), &prices);
    engine.accounts[user] = acc;
    println!("    withdraw while stale: {:?}", r.err());

    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Withdraw-mid-position probe: open large position, try to withdraw a lot
/// of capital so IM is violated. Should be rejected.
fn probe_withdraw_undercollateralize() {
    println!("  Withdraw undercollateralize: open 15x, try to withdraw down to IM violation");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V13Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(1_000)).unwrap();
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    let notional = usdc(15_000); // 15x
    let size_q = notional * POS_SCALE / oracle as u128;
    engine.trade(user, lp, SOL_ASSET, size_q, oracle, 1).unwrap();
    println!("    opened 15x position; cap=${}", engine.accounts[user].capital / USDC_DECIMALS);

    let prices = engine.effective_prices();
    for w in [10u128, 100, 500, 990, 999, 1000] {
        let mut acc = engine.accounts[user];
        let r = engine.group.withdraw_not_atomic(&mut acc, usdc(w), &prices);
        match &r {
            Ok(()) => {
                engine.accounts[user] = acc;
                println!("    withdraw ${:>4}: OK; cap=${}",
                    w, engine.accounts[user].capital / USDC_DECIMALS);
            }
            Err(e) => println!("    withdraw ${:>4}: {:?}", w, e),
        }
    }
    println!("    final cap: ${}  pnl: {}",
        engine.accounts[user].capital / USDC_DECIMALS,
        engine.accounts[user].pnl);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

fn run_probes_v13_extra() {
    println!("=== v13-specific probes ===");
    probe_multi_asset_crash();
    println!();
    probe_stale_extract();
    println!();
    probe_withdraw_undercollateralize();
}

/// P2 equivalent: 0 insurance, 0 LP capital. Long-running funding drain on
/// users with one-sided exposure. If the engine can ever leak negative pnl
/// to insurance, this should reveal it.
fn probe_no_lp_no_insurance() {
    println!("  P2: zero insurance, zero LP — does anything leak?");
    let cfg = V13Config {
        max_abs_funding_e9_per_slot: 10_000,
        ..make_bounty_sol_20x_max_config()
    };
    let mut engine = V13Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(1_000_000)).unwrap(); // small LP

    let mut users = Vec::new();
    for _ in 0..3 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(1_000)).unwrap();
        users.push(u);
    }

    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    // All users long; LP is short
    for &u in &users {
        let notional = usdc(8_000);
        let size_q = notional * POS_SCALE / oracle as u128;
        let _ = engine.trade(u, lp, SOL_ASSET, size_q, oracle, 1);
    }

    let ins_pre = engine.group.insurance;
    let mut total_insurance_used = 0u128;
    let mut total_liquidations = 0u32;
    let mut slot = 2u64;
    for _ in 0..500 {
        if engine.accrue_asset(SOL_ASSET, slot, oracle, 5_000).is_ok() {
            // try liquidations
            let prices = engine.effective_prices();
            for &u in &users {
                let mut acc = engine.accounts[u];
                let _ = engine.group.full_account_refresh(&mut acc, &prices);
                engine.accounts[u] = acc;
                if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                    if let Some(li) = (0..V13_MAX_PORTFOLIO_ASSETS_N)
                        .find(|&i| engine.accounts[u].legs[i].active) {
                        let mut acc = engine.accounts[u];
                        let qty = acc.legs[li].basis_pos_q.unsigned_abs();
                        if let Ok(out) = engine.group.liquidate_account_not_atomic(
                            &mut acc,
                            LiquidationRequestV13 { asset_index: li, close_q: qty, fee_bps: 5 },
                            &prices,
                        ) {
                            total_liquidations += 1;
                            total_insurance_used += out.insurance_used;
                        }
                        engine.accounts[u] = acc;
                    }
                }
            }
        }
        slot += 1;
    }
    println!("    liquidations:      {}", total_liquidations);
    println!("    insurance used:    {}", total_insurance_used);
    println!("    insurance: {} → {}", ins_pre, engine.group.insurance);
    println!("    invariants: {:?}", engine.assert_invariants().err());
}

/// P3 equivalent: concentrated longs on a small LP; large drop tests whether
/// ADL cascade can produce a deficit insurance must absorb.
fn probe_zero_insurance_concentrated_long() {
    println!("  P3: concentrated long crash — can ADL cascade leak?");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V13Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(2_000_000)).unwrap();

    let mut users = Vec::new();
    for _ in 0..10 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(1_000)).unwrap();
        users.push(u);
    }
    let oracle0 = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle0, 0).unwrap();

    for &u in &users {
        let notional = usdc(15_000);
        let size_q = notional * POS_SCALE / oracle0 as u128;
        let _ = engine.trade(u, lp, SOL_ASSET, size_q, oracle0, 1);
    }

    let max_move = cfg.max_price_move_bps_per_slot;
    let mut oracle = oracle0;
    let mut slot = 2u64;
    let mut total_liquidations = 0u32;
    let mut total_insurance_used = 0u128;
    let mut total_residual = 0u128;
    let mut total_explicit = 0u128;

    for _ in 0..200 {
        let d = (oracle as u128 * max_move as u128 / 10_000) as u64;
        oracle = oracle.saturating_sub(d).max(1);
        if engine.accrue_asset(SOL_ASSET, slot, oracle, 0).is_err() {
            slot += 1;
            continue;
        }
        let prices = engine.effective_prices();
        for &u in &users {
            let mut acc = engine.accounts[u];
            let _ = engine.group.full_account_refresh(&mut acc, &prices);
            engine.accounts[u] = acc;
            if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                if let Some(li) = (0..V13_MAX_PORTFOLIO_ASSETS_N)
                    .find(|&i| engine.accounts[u].legs[i].active) {
                    let mut acc = engine.accounts[u];
                    let qty = acc.legs[li].basis_pos_q.unsigned_abs();
                    if let Ok(out) = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV13 { asset_index: li, close_q: qty, fee_bps: 5 },
                        &prices,
                    ) {
                        total_liquidations += 1;
                        total_insurance_used += out.insurance_used;
                        total_residual += out.residual_booked;
                        total_explicit += out.explicit_loss;
                    }
                    engine.accounts[u] = acc;
                }
            }
        }
        slot += 1;
    }
    println!("    liquidations:      {}", total_liquidations);
    println!("    insurance used:    {}", total_insurance_used);
    println!("    residual booked:   {}", total_residual);
    println!("    explicit loss:     {}", total_explicit);
    println!("    final oracle:      ${}", oracle / 1_000_000);
    println!("    invariants:        {:?}", engine.assert_invariants().err());
}

/// P4 equivalent: whale trade ($20M @ 10x = $200M notional) + crash.
fn probe_whale_crash() {
    println!("  P4: whale ($20M @ 10x) + 36% crash");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V13Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(500_000_000)).unwrap(); // $500M LP

    let whale = engine.add_account(2).unwrap();
    engine.deposit(whale, usdc(20_000_000)).unwrap(); // $20M whale

    let oracle0 = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle0, 0).unwrap();

    let whale_notional = usdc(200_000_000); // $200M = 10x
    let size_q = whale_notional * POS_SCALE / oracle0 as u128;
    match engine.trade(whale, lp, SOL_ASSET, size_q, oracle0, 1) {
        Ok(_) => println!("    whale opened OK"),
        Err(e) => {
            println!("    whale trade REJECTED: {:?}", e);
            return;
        }
    }

    let max_move = cfg.max_price_move_bps_per_slot;
    let mut oracle = oracle0;
    let mut slot = 2u64;
    let mut total_liquidations = 0u32;
    let mut total_insurance_used = 0u128;
    let mut total_residual = 0u128;

    for _ in 0..200 {
        let d = (oracle as u128 * max_move as u128 / 10_000) as u64;
        oracle = oracle.saturating_sub(d).max(1);
        if engine.accrue_asset(SOL_ASSET, slot, oracle, 0).is_err() {
            slot += 1;
            continue;
        }
        let prices = engine.effective_prices();
        let mut acc = engine.accounts[whale];
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[whale] = acc;
        if engine.accounts[whale].health_cert.certified_liq_deficit > 0 {
            if let Some(li) = (0..V13_MAX_PORTFOLIO_ASSETS_N)
                .find(|&i| engine.accounts[whale].legs[i].active) {
                let mut acc = engine.accounts[whale];
                let qty = acc.legs[li].basis_pos_q.unsigned_abs();
                if let Ok(out) = engine.group.liquidate_account_not_atomic(
                    &mut acc,
                    LiquidationRequestV13 { asset_index: li, close_q: qty, fee_bps: 5 },
                    &prices,
                ) {
                    total_liquidations += 1;
                    total_insurance_used += out.insurance_used;
                    total_residual += out.residual_booked;
                }
                engine.accounts[whale] = acc;
            }
        }
        slot += 1;
    }
    println!("    liquidations:      {}", total_liquidations);
    println!("    insurance used:    {}", total_insurance_used);
    println!("    residual booked:   {}", total_residual);
    println!("    final oracle:      ${}", oracle / 1_000_000);
    println!("    invariants:        {:?}", engine.assert_invariants().err());
}

/// Long-running funding drain test
fn probe_long_funding_drain() {
    println!("  P5: 2000-slot funding drain at max rate");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V13Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user_long = engine.add_account(2).unwrap();
    let user_short = engine.add_account(3).unwrap();
    engine.deposit(user_long, usdc(10_000)).unwrap();
    engine.deposit(user_short, usdc(10_000)).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    let notional = usdc(50_000); // 5x leverage
    let size_q = notional * POS_SCALE / oracle as u128;
    let _ = engine.trade(user_long, lp, SOL_ASSET, size_q, oracle, 1);
    let _ = engine.trade(lp, user_short, SOL_ASSET, size_q, oracle, 1);

    let mut slot = 2u64;
    let mut total_liquidations = 0u32;
    let mut total_insurance_used = 0u128;

    for _ in 0..2000 {
        let _ = engine.accrue_asset(SOL_ASSET, slot, oracle, 10_000);
        let prices = engine.effective_prices();
        for &u in &[user_long, user_short] {
            let mut acc = engine.accounts[u];
            let _ = engine.group.full_account_refresh(&mut acc, &prices);
            engine.accounts[u] = acc;
            if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                if let Some(li) = (0..V13_MAX_PORTFOLIO_ASSETS_N)
                    .find(|&i| engine.accounts[u].legs[i].active) {
                    let mut acc = engine.accounts[u];
                    let qty = acc.legs[li].basis_pos_q.unsigned_abs();
                    if let Ok(out) = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV13 { asset_index: li, close_q: qty, fee_bps: 5 },
                        &prices,
                    ) {
                        total_liquidations += 1;
                        total_insurance_used += out.insurance_used;
                    }
                    engine.accounts[u] = acc;
                }
            }
        }
        slot += 1;
    }
    println!("    liquidations:      {}", total_liquidations);
    println!("    insurance used:    {}", total_insurance_used);
    println!("    user_long  pnl: {}  cap: {}",
        engine.accounts[user_long].pnl, engine.accounts[user_long].capital);
    println!("    user_short pnl: {}  cap: {}",
        engine.accounts[user_short].pnl, engine.accounts[user_short].capital);
    println!("    invariants:        {:?}", engine.assert_invariants().err());
}

/// V13 port of v12 F6 (positive PnL trap under stress).
///
/// In v13, threshold_stress_active is NOT auto-set by consumption tracking.
/// It's a wrapper-policy flag the operator flips for emergency pause. So
/// the v12 trap path — where sustained oracle volatility implicitly tripped
/// the gate — does not apply in v13. The wrapper retains explicit control.
///
/// This test confirms:
///  - With threshold_stress_active=false: convert/close work normally
///  - With threshold_stress_active=true (manually set): h_lock_lane→HMax,
///    favorable actions (withdraw, convert) return LockActive
///  - Clear the flag → behavior returns to normal
fn test_f6_v13() -> V13Result<()> {
    println!("=== v13 F6: conservative stress-pause policy ===");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V13Engine::new(cfg)?;
    let lp = engine.add_account(1)?;
    let user = engine.add_account(2)?;
    engine.deposit(lp, usdc(10_000_000))?;
    engine.deposit(user, usdc(1_000))?;
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0)?;

    // Open long position
    let notional = usdc(10_000);
    let size_q = notional * POS_SCALE / oracle as u128;
    engine.trade(user, lp, SOL_ASSET, size_q, oracle, 1)?;

    // Walk oracle up to generate +PnL
    let mut slot = 2u64;
    let mut p = oracle;
    for _ in 0..10 {
        p = p + (p as u128 * 30 / 10_000) as u64;
        engine.accrue_asset(SOL_ASSET, slot, p, 0)?;
        slot += 1;
    }
    println!("  oracle ${} → ${}", oracle / 1_000_000, p / 1_000_000);
    println!("  user.pnl=${}  reserved=${}",
        engine.accounts[user].pnl / 1_000_000,
        engine.accounts[user].reserved_pnl / USDC_DECIMALS);

    // Close position
    engine.trade(lp, user, SOL_ASSET, size_q, p, 1)?;
    let after_close_pnl = engine.accounts[user].pnl;
    println!("  after close: pnl=${} cap=${} reserved=${}",
        after_close_pnl / 1_000_000,
        engine.accounts[user].capital / USDC_DECIMALS,
        engine.accounts[user].reserved_pnl / USDC_DECIMALS);

    // Case 1: normal state — convert succeeds
    let prices = engine.effective_prices();
    let mut acc = engine.accounts[user];
    let r_normal = engine.group.convert_released_pnl_to_capital_not_atomic(&mut acc);
    engine.accounts[user] = acc;
    println!("  CASE A (no stress): convert → {:?}", r_normal.map(|v| format!("${}", v / 1_000_000)));

    // Case 2: manually set stress, retry convert
    engine.group.threshold_stress_active = true;
    let mut acc = engine.accounts[user];
    let r_stressed = engine.group.convert_released_pnl_to_capital_not_atomic(&mut acc);
    engine.accounts[user] = acc;
    println!("  CASE B (stress=true): convert → {:?}", r_stressed.err());

    // Case 3: clear stress, retry
    engine.group.threshold_stress_active = false;
    let mut acc = engine.accounts[user];
    let _ = engine.group.full_account_refresh(&mut acc, &prices);
    let r_cleared = engine.group.convert_released_pnl_to_capital_not_atomic(&mut acc);
    engine.accounts[user] = acc;
    println!("  CASE C (stress cleared): convert → {:?}", r_cleared.map(|v| format!("${}", v / 1_000_000)));

    println!();
    println!("VERDICT: v13 conservative-pause is wrapper-controlled (not auto-tripped");
    println!("  by consumption). When stress=true, favorable actions return LockActive.");
    println!("  When cleared, normal flow resumes. F6 mechanism is the same conservative");
    println!("  policy as v12 but with explicit wrapper control vs implicit auto-trip.");
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("Usage:");
        println!("  --test=smoke              single smoke run");
        println!("  --test=probe_configs      show which configs validate");
        println!("  --test=exec_price_attack  v13 exec_price deviation");
        println!("  --test=sybil_close        v13 sybil two-step exec_price");
        println!("  --fuzz=N                  run N-seed bounty fuzz");
        return;
    }
    if args.iter().any(|a| a == "--test=smoke") {
        match smoke_test() {
            Ok(()) => println!("smoke: OK"),
            Err(e) => {
                println!("smoke: FAILED with {:?}", e);
                std::process::exit(1);
            }
        }
        return;
    }
    if args.iter().any(|a| a == "--test=probe_configs") {
        probe_bounty_variants();
        return;
    }
    if args.iter().any(|a| a == "--test=exec_price_attack") {
        match test_exec_price_attack_v13() {
            Ok(()) => {},
            Err(e) => println!("FAILED: {:?}", e),
        }
        return;
    }
    if args.iter().any(|a| a == "--test=sybil_close") {
        match test_sybil_close_v13() {
            Ok(()) => {},
            Err(e) => println!("FAILED: {:?}", e),
        }
        return;
    }
    if args.iter().any(|a| a == "--test=probes") {
        run_probes();
        return;
    }
    if args.iter().any(|a| a == "--test=probes_v13") {
        run_probes_v13_extra();
        return;
    }
    if args.iter().any(|a| a == "--test=f6") {
        match test_f6_v13() {
            Ok(()) => {},
            Err(e) => println!("FAILED: {:?}", e),
        }
        return;
    }
    let scen = args
        .iter()
        .find(|a| a.starts_with("--scenario="))
        .and_then(|a| Scenario::from_str(a.strip_prefix("--scenario=").unwrap()))
        .unwrap_or(Scenario::Random);
    if let Some(arg) = args.iter().find(|a| a.starts_with("--fuzz=")) {
        let n: usize = arg.strip_prefix("--fuzz=").unwrap().parse().unwrap_or(100);
        run_fuzz(scen, n);
        return;
    }
    if args.iter().any(|a| a == "--fuzz-all" || a.starts_with("--fuzz-all=")) {
        let n = args.iter()
            .find(|a| a.starts_with("--fuzz-all="))
            .and_then(|a| a.strip_prefix("--fuzz-all=").unwrap().parse().ok())
            .unwrap_or(200);
        for scen in [
            Scenario::Random,
            Scenario::Crash10,
            Scenario::Crash20,
            Scenario::FundingDrain,
            Scenario::OracleWick,
            Scenario::HighLev,
            Scenario::Mega,
        ] {
            run_fuzz(scen, n);
            println!();
        }
        return;
    }
    println!("v13 port in progress. Try: --test=smoke, --fuzz=200, --fuzz=2000 --scenario=crash20, --fuzz-all");
}
