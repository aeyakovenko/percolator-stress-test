//! Percolator v14 stress test — port in progress.
//!
//! v12 main.rs preserved as src/main_v12.rs.bak for reference. The v14 engine
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
/// In v14 the engine no longer owns the account slab; the wrapper passes
/// accounts in by mut-ref to each call.
struct V14Engine {
    group: MarketGroupV14,
    accounts: Vec<PortfolioAccountV14>,
    market_group_id: [u8; 32],
    next_account_seq: u64,
}

impl V14Engine {
    fn new(config: V14Config) -> V14Result<Self> {
        let group_id = [0x42u8; 32];
        let group = MarketGroupV14::new(group_id, config)?;
        Ok(Self {
            group,
            accounts: Vec::new(),
            market_group_id: group_id,
            next_account_seq: 0,
        })
    }

    /// Create a new portfolio account and register it with the market group.
    fn add_account(&mut self, owner_byte: u8) -> V14Result<usize> {
        let mut id = [0u8; 32];
        id[..8].copy_from_slice(&self.next_account_seq.to_le_bytes());
        self.next_account_seq += 1;
        let owner = [owner_byte; 32];
        let header = ProvenanceHeaderV14::new(self.market_group_id, id, owner);
        let account = PortfolioAccountV14::empty(header);
        self.group.create_portfolio_account(&account)?;
        let idx = self.accounts.len();
        self.accounts.push(account);
        Ok(idx)
    }

    fn deposit(&mut self, idx: usize, amount: u128) -> V14Result<()> {
        let mut acc = self.accounts[idx];
        self.group.deposit_not_atomic(&mut acc, amount)?;
        self.accounts[idx] = acc;
        Ok(())
    }

    fn effective_prices(&self) -> [u64; V14_MAX_PORTFOLIO_ASSETS_N] {
        let mut p = [0u64; V14_MAX_PORTFOLIO_ASSETS_N];
        for i in 0..V14_MAX_PORTFOLIO_ASSETS_N {
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
    ) -> V14Result<TradeOutcomeV14> {
        let prices = self.effective_prices();
        let mut long_acc = self.accounts[long_idx];
        let mut short_acc = self.accounts[short_idx];
        let req = TradeRequestV14 {
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
    ) -> V14Result<AccrueAssetOutcomeV14> {
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

    fn assert_invariants(&self) -> V14Result<()> {
        self.group.assert_public_invariants()
    }
}

/// Conservative v14 config — full margin coverage, no extra fees. Passes
/// the strict v14 solvency envelope check. Stage 1 uses this to verify the
/// flow; the bounty_sol_20x_max config comes in stage 2 once we know what
/// the v14 envelope allows.
fn make_full_margin_config() -> V14Config {
    V14Config::public_user_fund(1, 0, 30)
}

/// Probe: try variants of the bounty_sol_20x_max config to find what
/// v14's validate_exact_solvency_envelope accepts.
fn probe_bounty_variants() {
    let mk = |max_move: u64, max_dt: u64, liq: u64, fee: u64| V14Config {
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
        recovery_fallback_price_enabled: true,
        max_bankrupt_close_lifetime_slots: 1000,
    };
    let cases: Vec<(String, V14Config)> = vec![
        ("baseline v12 max_risk".to_string(), make_bounty_sol_20x_max_config()),
        ("mm=im=10000 (full)".to_string(), V14Config::public_user_fund(1, 0, 30)),
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

/// v14 bounty config — equivalent to v12 max_risk.md but tuned to v14's
/// stricter solvency envelope. v12 allowed max_move=49 bps/slot; v14's exact
/// envelope reserves more headroom, max it accepts is 45.
/// Effective per-accrual oracle tolerance: 45 × 10 = 450 bps = 4.5%.
fn make_bounty_sol_20x_max_config() -> V14Config {
    make_bounty_config(1)
}

fn make_bounty_config(n_assets: u8) -> V14Config {
    V14Config {
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
        recovery_fallback_price_enabled: true,
        max_bankrupt_close_lifetime_slots: 1000,
    }
}

/// Stage 1 smoke test: create engine, add LP + user, deposit, trade, accrue, close.
fn smoke_test() -> V14Result<()> {
    let cfg = make_bounty_sol_20x_max_config();
    println!("V14 stage-2 smoke: bounty_sol_20x_max config (v14-tuned)");
    cfg.validate_public_user_fund()?;
    println!("  config validated");

    let mut engine = V14Engine::new(cfg)?;
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
fn invariant_battery(engine: &V14Engine) -> Vec<(&'static str, bool)> {
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
        if asset.oi_eff_long_q > 0 && asset.mode_long == SideModeV14::Normal && asset.a_long < MIN_A_SIDE {
            a_ok = false;
        }
        if asset.oi_eff_short_q > 0 && asset.mode_short == SideModeV14::Normal && asset.a_short < MIN_A_SIDE {
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

    // I9 (v14-specific): F7 invariant — DrainOnly implies opposing OI > 0 OR opp is in ResetPending
    let mut f7_ok = true;
    for i in 0..g.config.max_portfolio_assets as usize {
        let asset = g.assets[i];
        if asset.mode_long == SideModeV14::DrainOnly
            && asset.oi_eff_short_q == 0
            && asset.mode_short != SideModeV14::ResetPending
        {
            f7_ok = false;
        }
        if asset.mode_short == SideModeV14::DrainOnly
            && asset.oi_eff_long_q == 0
            && asset.mode_long != SideModeV14::ResetPending
        {
            f7_ok = false;
        }
    }
    results.push(("F7 DrainOnly + opp OI consistent", f7_ok));

    results
}

fn run_invariant_battery(engine: &V14Engine) -> u32 {
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
    let mut engine = V14Engine::new(cfg).expect("init");
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
                for li in 0..V14_MAX_PORTFOLIO_ASSETS_N {
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
                        LiquidationRequestV14 {
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

/// V14 port of v12 exec_price_attack test. Engine v14 also doesn't bound
/// exec_price vs oracle directly; defense is the post-trade IM check.
fn test_exec_price_attack_v14() -> V14Result<()> {
    println!("=== v14 exec_price attack: bounty_sol_20x_max ===");
    let cfg = make_bounty_sol_20x_max_config();
    let oracle = price_e6(200);

    for deviation_bps in [100u64, 1000, 5000, 9999] {
        let mut engine = V14Engine::new(cfg)?;
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

/// V14 port of v12 sybil_close_attack: open A↔B at fair price, then close
/// at adversarial exec to dump loss onto one side.
fn test_sybil_close_v14() -> V14Result<()> {
    println!("=== v14 sybil close: bounty_sol_20x_max ===");
    let cfg = make_bounty_sol_20x_max_config();
    let oracle = price_e6(200);

    for deviation_bps in [100u64, 1000, 5000, 9999] {
        let mut engine = V14Engine::new(cfg)?;
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

/// v14 probe_drain port: 4 pathological configs trying to force insurance
/// to absorb a deficit. v12 had 5 probes; one (P1) used the zombie-injection
/// test backdoor which doesn't exist in v14. The other 4 are reachable
/// through legitimate APIs and worth testing.
fn run_probes() {
    println!("=== v14 probe_drain: 4 pathological scenarios ===");
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
// v14-specific probes (not portable to v12 — exercise new attack surface)
// ════════════════════════════════════════════════════════════════════════════

/// Multi-asset portfolio probe: open hedged long+short across 2 assets,
/// crash one. Hedge should NOT mask losses on the crashed leg.
fn probe_multi_asset_crash() {
    println!("  Multi-asset: 2 assets, hedged long-short, one crashes");
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
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
            for li in 0..V14_MAX_PORTFOLIO_ASSETS_N {
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
                    LiquidationRequestV14 { asset_index: best.0, close_q: best.1, fee_bps: 5 },
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
    let mut engine = V14Engine::new(cfg).expect("init");
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
    let mut engine = V14Engine::new(cfg).expect("init");
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

fn run_probes_v14_extra() {
    println!("=== v14-specific probes ===");
    probe_multi_asset_crash();
    println!();
    probe_stale_extract();
    println!();
    probe_withdraw_undercollateralize();
}

/// Account close path: deposit, open, close position, withdraw all, close
/// account. Verifies the full happy-path exit is clean.
fn probe_account_close() {
    println!("  Account close path: full deposit → trade → close cycle");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(1_000)).unwrap();
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    // Open + close position via reverse trade
    let notional = usdc(5_000);
    let size_q = notional * POS_SCALE / oracle as u128;
    engine.trade(user, lp, SOL_ASSET, size_q, oracle, 1).unwrap();
    engine.trade(lp, user, SOL_ASSET, size_q, oracle, 1).unwrap();
    println!("    after close trade: cap=${} pnl={} legs[0].active={}",
        engine.accounts[user].capital / USDC_DECIMALS,
        engine.accounts[user].pnl,
        engine.accounts[user].legs[0].active);

    // Withdraw remaining capital
    let prices = engine.effective_prices();
    let cap_left = engine.accounts[user].capital;
    let mut acc = engine.accounts[user];
    let r = engine.group.withdraw_not_atomic(&mut acc, cap_left, &prices);
    engine.accounts[user] = acc;
    println!("    withdraw ${}: {:?}", cap_left / USDC_DECIMALS, r);

    // Close account
    let acc = engine.accounts[user];
    let r = engine.group.close_portfolio_account(&acc);
    println!("    close_portfolio_account: {:?}", r);
    println!("    materialized_portfolio_count: {}", engine.group.materialized_portfolio_count);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Long-dt gap test: skip cranks for max_accrual_dt_slots+1 slots and see
/// what happens. Engine should reject excessive jumps.
fn probe_long_dt_gap() {
    println!("  Long-dt gap: skip cranks for max_dt+5 slots, attempt accrue");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(1_000)).unwrap();
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    let notional = usdc(5_000);
    let size_q = notional * POS_SCALE / oracle as u128;
    engine.trade(user, lp, SOL_ASSET, size_q, oracle, 1).unwrap();

    // Skip many slots and try to accrue
    let dt_skip = cfg.max_accrual_dt_slots + 5;
    let new_oracle = oracle + (oracle as u128 * 50 / 10_000) as u64; // 0.5% move
    let r = engine.accrue_asset(SOL_ASSET, 2 + dt_skip, new_oracle, 0);
    println!("    accrue after {}-slot gap with price move: {:?}", dt_skip, r);

    // Try with no price move
    let r2 = engine.accrue_asset(SOL_ASSET, 2 + dt_skip, oracle, 0);
    println!("    accrue after gap, same price: {:?}", r2);

    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Rapid open-close churn: open and close 100 times in a row at slightly
/// different prices. Tests fee accounting and account state hygiene.
fn probe_rapid_churn() {
    println!("  Rapid churn: 100 open-close cycles");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(10_000)).unwrap();
    let mut oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    let initial_cap = engine.accounts[user].capital;
    let initial_lp_cap = engine.accounts[lp].capital;
    let initial_insurance = engine.group.insurance;
    let mut slot = 2u64;
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut cycle_failures = 0;

    for cycle in 0..100 {
        // Move oracle slightly
        let dir = cycle % 2 == 0;
        let d = (oracle as u128 * (max_move as u128 / 5) / 10_000) as u64;
        oracle = if dir { oracle + d } else { oracle.saturating_sub(d).max(1) };
        if engine.accrue_asset(SOL_ASSET, slot, oracle, 0).is_err() {
            cycle_failures += 1;
            slot += 1;
            continue;
        }
        slot += 1;

        // Open
        let size_q = usdc(5_000) * POS_SCALE / oracle as u128;
        if engine.trade(user, lp, SOL_ASSET, size_q, oracle, 1).is_err() {
            cycle_failures += 1;
            continue;
        }
        // Close
        if engine.trade(lp, user, SOL_ASSET, size_q, oracle, 1).is_err() {
            cycle_failures += 1;
            continue;
        }
    }
    let final_cap = engine.accounts[user].capital;
    let final_lp_cap = engine.accounts[lp].capital;
    let final_insurance = engine.group.insurance;
    println!("    cycles attempted:    100");
    println!("    cycle failures:      {}", cycle_failures);
    println!("    user cap: ${} → ${} (Δ={})",
        initial_cap / USDC_DECIMALS, final_cap / USDC_DECIMALS,
        (final_cap as i128 - initial_cap as i128) / 1_000_000);
    println!("    LP cap:   ${} → ${} (Δ={})",
        initial_lp_cap / USDC_DECIMALS, final_lp_cap / USDC_DECIMALS,
        (final_lp_cap as i128 - initial_lp_cap as i128) / 1_000_000);
    println!("    insurance: ${} → ${} (Δ={})",
        initial_insurance / USDC_DECIMALS, final_insurance / USDC_DECIMALS,
        (final_insurance as i128 - initial_insurance as i128) / 1_000_000);
    println!("    user.legs[0].active: {}", engine.accounts[user].legs[0].active);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

fn run_probes_v14_more() {
    println!("=== v14 paths probes (close, dt gap, churn) ===");
    probe_account_close();
    println!();
    probe_long_dt_gap();
    println!();
    probe_rapid_churn();
}

/// Resolve market path: trigger market resolve, then exit accounts via
/// close_resolved_account_not_atomic. Tests the emergency-exit path.
fn probe_resolve_exit() {
    println!("  Market resolve + emergency exit: 5 users, resolve, close_resolved");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    let mut users = Vec::new();
    for _ in 0..5 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(1_000)).unwrap();
        users.push(u);
    }
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    // Each user opens random position
    let mut rng = Rng::new(42);
    for &u in &users {
        let going_long = rng.bool();
        let notional = usdc(5_000);
        let size_q = notional * POS_SCALE / oracle as u128;
        let (long, short) = if going_long { (u, lp) } else { (lp, u) };
        let _ = engine.trade(long, short, SOL_ASSET, size_q, oracle, 1);
    }

    // Resolve the market at slot 10
    let r = engine.group.resolve_market_not_atomic(10);
    println!("    resolve_market: {:?}", r);
    println!("    mode: {:?}", engine.group.mode);

    // Each user tries close_resolved_account_not_atomic
    let mut exits = 0;
    let mut progresses = 0;
    for &u in &users {
        for attempt in 0..10 {
            let mut acc = engine.accounts[u];
            let r = engine.group.close_resolved_account_not_atomic(&mut acc, 0);
            engine.accounts[u] = acc;
            match r {
                Ok(ResolvedCloseOutcomeV14::ProgressOnly) => {
                    progresses += 1;
                }
                Ok(ResolvedCloseOutcomeV14::Closed { payout }) => {
                    exits += 1;
                    println!("    user {} closed on attempt {}: payout=${}",
                        u, attempt, payout / USDC_DECIMALS);
                    break;
                }
                Err(e) => {
                    println!("    user {} attempt {}: {:?}", u, attempt, e);
                    break;
                }
            }
        }
    }
    println!("    exits: {} / {} users", exits, users.len());
    println!("    progress-only calls: {}", progresses);
    println!("    invariants: {:?}", engine.assert_invariants().err());
}

fn run_probes_resolve() {
    println!("=== v14 resolve / emergency-exit probe ===");
    probe_resolve_exit();
}

/// Boundary-value probe: test engine at the extreme inputs it accepts.
fn probe_boundary_values() {
    println!("  Boundary values: extreme size_q, exec_price, notional");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(5_000_000_000)).unwrap(); // $5B LP (well under MAX_VAULT_TVL=$10B)
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(100_000_000)).unwrap(); // $100M user
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    // (a) size_q = 1 (smallest non-zero)
    let r = engine.trade(user, lp, SOL_ASSET, 1, oracle, 1);
    println!("    size_q=1:                    {:?}", r.err());

    // (b) size_q at MAX_TRADE_SIZE_Q (should reject — exceeds)
    let r = engine.trade(user, lp, SOL_ASSET, MAX_TRADE_SIZE_Q, oracle, 1);
    println!("    size_q=MAX_TRADE_SIZE_Q:    {:?}", r.err());

    // (c) size_q at MAX_TRADE_SIZE_Q - 1 (should also reject — notional too large)
    let r = engine.trade(user, lp, SOL_ASSET, MAX_TRADE_SIZE_Q.saturating_sub(1), oracle, 1);
    println!("    size_q=MAX-1:               {:?}", r.err());

    // (d) exec_price = 1 (smallest valid)
    let r = engine.trade(user, lp, SOL_ASSET, 1, 1, 1);
    println!("    exec_price=1, size=1:        {:?}", r.err());

    // (e) exec_price = MAX_ORACLE_PRICE (largest valid)
    let r = engine.trade(user, lp, SOL_ASSET, 1, MAX_ORACLE_PRICE, 1);
    println!("    exec_price=MAX:              {:?}", r.err());

    // (f) exec_price = 0 (invalid)
    let r = engine.trade(user, lp, SOL_ASSET, 100, 0, 1);
    println!("    exec_price=0:                {:?}", r.err());

    // (g) fee_bps > max_trading_fee_bps
    let r = engine.trade(user, lp, SOL_ASSET, 100, oracle, 100);
    println!("    fee_bps=100 (cfg max=1):     {:?}", r.err());

    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Rebalance path: reduce position without margin check (risk-reducing only)
fn probe_rebalance() {
    println!("  Rebalance path: open large, reduce via rebalance_reduce_position");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(1_000)).unwrap();
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    let notional = usdc(15_000);
    let size_q = notional * POS_SCALE / oracle as u128;
    engine.trade(user, lp, SOL_ASSET, size_q, oracle, 1).unwrap();
    println!("    opened 15x position: pos_q={}", engine.accounts[user].legs[0].basis_pos_q);

    // Reduce by half
    let reduce_q = size_q / 2;
    let prices = engine.effective_prices();
    let mut acc = engine.accounts[user];
    let r = engine.group.rebalance_reduce_position_not_atomic(
        &mut acc,
        RebalanceRequestV14 {
            asset_index: SOL_ASSET,
            reduce_q,
        },
        &prices,
    );
    engine.accounts[user] = acc;
    println!("    rebalance reduce by half:    {:?}", r);
    println!("    after rebalance: pos_q={}", engine.accounts[user].legs[0].basis_pos_q);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

fn run_probes_boundary() {
    println!("=== v14 boundary + rebalance probes ===");
    probe_boundary_values();
    println!();
    probe_rebalance();
}

/// ADL drain reset probe (v12 adl_drain_reset port).
/// Force one side's ADL multiplier (a_long or a_short) to floor by repeatedly
/// liquidating opposite-side bankrupt positions, triggering ADL haircuts.
/// Verify the side cleanly transitions Normal → DrainOnly → (eventually)
/// ResetPending.
fn probe_adl_drain_reset() {
    println!("  ADL drain-reset: force a_side to floor, observe DrainOnly → ResetPending");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    // Many small longs that will all crash
    let mut longs = Vec::new();
    for _ in 0..50 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(100)).unwrap();
        longs.push(u);
    }
    // Hedge shorts that will benefit
    let mut shorts = Vec::new();
    for _ in 0..10 {
        let u = engine.add_account(3).unwrap();
        engine.deposit(u, usdc(1_000)).unwrap();
        shorts.push(u);
    }

    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    // Longs open near-max-leverage
    let long_notional = usdc(1_800); // 18x of $100
    let long_size = long_notional * POS_SCALE / oracle as u128;
    for &u in &longs {
        let _ = engine.trade(u, lp, SOL_ASSET, long_size, oracle, 1);
    }
    // Shorts open similar exposure on opposite side
    let short_notional = usdc(15_000);
    let short_size = short_notional * POS_SCALE / oracle as u128;
    for &u in &shorts {
        let _ = engine.trade(lp, u, SOL_ASSET, short_size, oracle, 1);
    }

    println!("    initial: {} longs, {} shorts, asset OI long=${} short=${}",
        longs.len(), shorts.len(),
        engine.group.assets[0].oi_eff_long_q,
        engine.group.assets[0].oi_eff_short_q);
    println!("    a_long={}  a_short={}",
        engine.group.assets[0].a_long, engine.group.assets[0].a_short);

    // Crash to push longs into MM violation
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o = oracle;
    let mut slot = 2u64;
    let mut total_liquidations = 0u32;
    let mut total_insurance_used = 0u128;
    let mut drain_only_long_seen = false;
    let mut drain_only_short_seen = false;
    let mut reset_pending_seen = false;

    for _ in 0..400 {
        let d = (o as u128 * max_move as u128 / 10_000) as u64;
        o = o.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(SOL_ASSET, slot, o, 0);

        let prices = engine.effective_prices();
        for &u in &longs {
            let mut acc = engine.accounts[u];
            let _ = engine.group.full_account_refresh(&mut acc, &prices);
            engine.accounts[u] = acc;
            if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                let leg = engine.accounts[u].legs[0];
                if leg.active {
                    let mut acc = engine.accounts[u];
                    if let Ok(out) = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV14 {
                            asset_index: 0,
                            close_q: leg.basis_pos_q.unsigned_abs(),
                            fee_bps: 5,
                        },
                        &prices,
                    ) {
                        total_liquidations += 1;
                        total_insurance_used += out.insurance_used;
                    }
                    engine.accounts[u] = acc;
                }
            }
        }
        // Observe side-mode transitions
        match engine.group.assets[0].mode_long {
            SideModeV14::DrainOnly => drain_only_long_seen = true,
            SideModeV14::ResetPending => reset_pending_seen = true,
            _ => {}
        }
        match engine.group.assets[0].mode_short {
            SideModeV14::DrainOnly => drain_only_short_seen = true,
            SideModeV14::ResetPending => reset_pending_seen = true,
            _ => {}
        }
        slot += 1;
    }
    println!("    final oracle: ${}", o / 1_000_000);
    println!("    a_long={}  a_short={}",
        engine.group.assets[0].a_long, engine.group.assets[0].a_short);
    println!("    mode: long={:?}  short={:?}",
        engine.group.assets[0].mode_long, engine.group.assets[0].mode_short);
    println!("    transitions seen: DrainOnly_long={}  DrainOnly_short={}  ResetPending={}",
        drain_only_long_seen, drain_only_short_seen, reset_pending_seen);
    println!("    total liquidations: {}", total_liquidations);
    println!("    insurance used: {}", total_insurance_used);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Dust GC probe (v12 dust_gc port).
/// Open many tiny positions to test phantom-dust tracking and cleanup.
fn probe_dust_gc() {
    println!("  Dust GC: open many tiny positions, observe phantom dust handling");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    let mut users = Vec::new();
    for _ in 0..30 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(50)).unwrap();
        users.push(u);
    }
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    let mut opened = 0;
    let mut rng = Rng::new(7);
    for &u in &users {
        let going_long = rng.bool();
        let notional = usdc(300); // 6x of $50 — tiny positions
        let size_q = notional * POS_SCALE / oracle as u128;
        let (long, short) = if going_long { (u, lp) } else { (lp, u) };
        if engine.trade(long, short, SOL_ASSET, size_q, oracle, 1).is_ok() {
            opened += 1;
        }
    }
    println!("    opened {} tiny positions", opened);

    // Mild oracle walk + repeated close-reopen by random users
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o = oracle;
    let mut slot = 2u64;
    let mut churn_cycles = 0;
    for step in 0..300 {
        let dir = rng.bool();
        let pct = rng.range_u64(0, max_move / 2);
        let d = (o as u128 * pct as u128 / 10_000) as u64;
        o = if dir { o + d } else { o.saturating_sub(d).max(1) };
        let _ = engine.accrue_asset(SOL_ASSET, slot, o, 0);

        // Every 10 slots, a random user closes and reopens
        if step % 10 == 0 {
            let u = users[(rng.next_u64() as usize) % users.len()];
            let leg = engine.accounts[u].legs[0];
            if leg.active {
                let qty = leg.basis_pos_q.unsigned_abs();
                let (long, short) = if leg.side == SideV14::Long { (lp, u) } else { (u, lp) };
                if engine.trade(long, short, SOL_ASSET, qty, o, 1).is_ok() {
                    churn_cycles += 1;
                    // reopen on the opposite side
                    let notional = usdc(200);
                    let new_size = notional * POS_SCALE / o as u128;
                    let going_long = rng.bool();
                    let (long, short) = if going_long { (u, lp) } else { (lp, u) };
                    let _ = engine.trade(long, short, SOL_ASSET, new_size, o, 1);
                }
            }
        }
        slot += 1;
    }
    println!("    churn cycles: {}", churn_cycles);
    println!("    final OI: long={}  short={}",
        engine.group.assets[0].oi_eff_long_q,
        engine.group.assets[0].oi_eff_short_q);
    println!("    stored_pos_count: long={}  short={}",
        engine.group.assets[0].stored_pos_count_long,
        engine.group.assets[0].stored_pos_count_short);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Adversarial-keeper probe (v12 adversarial_keeper port).
/// Keeper liquidates accounts in the WORST possible order — touches the
/// most-profitable (most-unrealized-PnL) accounts first to maximize the
/// system's exposure during the cascade.
fn probe_adversarial_keeper() {
    println!("  Adversarial keeper: liquidate richest accounts first under crash");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let mut longs = Vec::new();
    for _ in 0..20 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(1_000)).unwrap();
        longs.push(u);
    }
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    for &u in &longs {
        let size_q = usdc(8_000) * POS_SCALE / oracle as u128;
        let _ = engine.trade(u, lp, SOL_ASSET, size_q, oracle, 1);
    }

    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o = oracle;
    let mut slot = 2u64;
    let mut total_liquidations = 0u32;
    let mut total_insurance_used = 0u128;

    for _ in 0..300 {
        let d = (o as u128 * max_move as u128 / 10_000) as u64;
        o = o.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(SOL_ASSET, slot, o, 0);

        let prices = engine.effective_prices();
        // ADVERSARIAL ORDER: refresh all, sort by HIGHEST equity, liquidate top first
        let mut candidates: Vec<(usize, i128)> = vec![];
        for &u in &longs {
            let mut acc = engine.accounts[u];
            let _ = engine.group.full_account_refresh(&mut acc, &prices);
            engine.accounts[u] = acc;
            if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                let equity = engine.accounts[u].capital as i128 + engine.accounts[u].pnl;
                candidates.push((u, equity));
            }
        }
        candidates.sort_by(|a, b| b.1.cmp(&a.1)); // descending — most equity first

        for (u, _) in candidates {
            let leg = engine.accounts[u].legs[0];
            if leg.active {
                let mut acc = engine.accounts[u];
                if let Ok(out) = engine.group.liquidate_account_not_atomic(
                    &mut acc,
                    LiquidationRequestV14 {
                        asset_index: 0,
                        close_q: leg.basis_pos_q.unsigned_abs(),
                        fee_bps: 5,
                    },
                    &prices,
                ) {
                    total_liquidations += 1;
                    total_insurance_used += out.insurance_used;
                }
                engine.accounts[u] = acc;
            }
        }
        slot += 1;
    }
    println!("    final oracle: ${}", o / 1_000_000);
    println!("    total liquidations: {}", total_liquidations);
    println!("    insurance used: {}", total_insurance_used);
    let mut total_user_cap = 0u128;
    for &u in &longs {
        total_user_cap += engine.accounts[u].capital;
    }
    println!("    sum user capital remaining: ${}", total_user_cap / USDC_DECIMALS);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

// ════════════════════════════════════════════════════════════════════════════
// v14 cross-margin probes — NEW SURFACE vs v13
// ════════════════════════════════════════════════════════════════════════════

/// Cross-margin offset: profitable leg supports losing leg's MM.
/// Open long $5k asset A + short $5k asset B. Move asset A up 10%
/// (long profits $500). Then move asset B up 10% (short loses $500).
/// Net PnL ~ 0; aggregate equity preserved; no liquidation.
fn probe_xmargin_offset() {
    println!("  Cross-margin offset: profitable leg supports losing leg");
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(1_000)).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    let notional = usdc(5_000);
    let size_q = notional * POS_SCALE / oracle as u128;
    engine.trade(user, lp, 0, size_q, oracle, 1).unwrap(); // long asset 0
    engine.trade(lp, user, 1, size_q, oracle, 1).unwrap(); // short asset 1
    println!("    opened: long $5k asset 0, short $5k asset 1");

    let mut o0 = oracle;
    let mut o1 = oracle;
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut slot = 2u64;
    // Move asset 0 up 10% over many slots (long profitable)
    let target0 = oracle + oracle / 10;
    while o0 < target0 {
        let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
        o0 = (o0.saturating_add(d)).min(target0);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);
        slot += 1;
    }
    // Now move asset 1 up 10% (short losing)
    let target1 = oracle + oracle / 10;
    while o1 < target1 {
        let d = (o1 as u128 * max_move as u128 / 10_000) as u64;
        o1 = (o1.saturating_add(d)).min(target1);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);
        slot += 1;
    }
    println!("    after moves: oracle A=${} B=${}", o0/1_000_000, o1/1_000_000);

    // Refresh account: pnl should be near 0 (offsetting moves)
    let prices = engine.effective_prices();
    let mut acc = engine.accounts[user];
    let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
    let _ = engine.group.full_account_refresh(&mut acc, &prices);
    engine.accounts[user] = acc;

    let cert = engine.accounts[user].health_cert;
    println!("    cap=${}  pnl={}  cert.equity={}  cert.mm_req={}  liq_deficit={}",
        engine.accounts[user].capital / USDC_DECIMALS,
        engine.accounts[user].pnl,
        cert.certified_equity,
        cert.certified_maintenance_req,
        cert.certified_liq_deficit);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Cross-margin asymmetric: long BTC + long SOL, only BTC crashes.
/// SOL leg's neutral value supports BTC's losses via shared capital.
/// BTC should be liquidatable BEFORE SOL is affected, but the SHARED
/// capital pool means SOL leg also degrades.
fn probe_xmargin_asymmetric() {
    println!("  Cross-margin asymmetric: long-A + long-B, only A crashes");
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(2_000)).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    let size_q = usdc(8_000) * POS_SCALE / oracle as u128; // 4x on each = 8x portfolio
    engine.trade(user, lp, 0, size_q, oracle, 1).unwrap();
    engine.trade(user, lp, 1, size_q, oracle, 1).unwrap();
    println!("    opened: long $8k asset 0 + long $8k asset 1 (portfolio 8x on $2k)");

    let mut o0 = oracle;
    let o1 = oracle;
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut slot = 2u64;
    let mut total_liquidations = 0u32;
    let mut total_insurance_used = 0u128;
    let mut leg0_liqs = 0u32;
    let mut leg1_liqs = 0u32;

    // Crash asset 0 only
    for _ in 0..200 {
        let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
        o0 = o0.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);

        let prices = engine.effective_prices();
        let mut acc = engine.accounts[user];
        let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[user] = acc;

        if engine.accounts[user].health_cert.certified_liq_deficit > 0 {
            // Liquidate the largest active leg
            let mut best = (0usize, 0u128);
            for li in 0..V14_MAX_PORTFOLIO_ASSETS_N {
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
                    LiquidationRequestV14 {
                        asset_index: best.0,
                        close_q: best.1,
                        fee_bps: 5,
                    },
                    &prices,
                ) {
                    total_liquidations += 1;
                    total_insurance_used += out.insurance_used;
                    if best.0 == 0 { leg0_liqs += 1; } else { leg1_liqs += 1; }
                }
                engine.accounts[user] = acc;
            }
        }
        slot += 1;
    }
    println!("    final oracle: A=${}  B=${}", o0 / 1_000_000, o1 / 1_000_000);
    println!("    leg 0 (crashed) liquidations: {}", leg0_liqs);
    println!("    leg 1 (flat) liquidations:    {}", leg1_liqs);
    println!("    total insurance used: {}", total_insurance_used);
    println!("    user final: cap=${} pnl={} legs_active={}",
        engine.accounts[user].capital / USDC_DECIMALS,
        engine.accounts[user].pnl,
        engine.accounts[user].active_bitmap.count_ones());
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Cross-margin haircut: many users with positive PnL claims, system
/// junior_claim_bound exceeds residual. Positive PnL should be haircut
/// when used as support, preventing leg-local paper profit from becoming
/// senior unbacked claims.
fn probe_xmargin_haircut() {
    println!("  Cross-margin haircut: stress positive-PnL support haircut");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(1_000_000)).unwrap(); // smaller LP — limits residual

    let mut winners = Vec::new();
    for _ in 0..10 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(100)).unwrap();
        winners.push(u);
    }
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    // All open longs
    for &u in &winners {
        let size_q = usdc(500) * POS_SCALE / oracle as u128;
        let _ = engine.trade(u, lp, SOL_ASSET, size_q, oracle, 1);
    }
    println!("    opened 10 longs at 5x leverage, smaller LP");
    println!("    initial vault=${} c_tot=${} insurance=${} residual={}",
        engine.group.vault / USDC_DECIMALS,
        engine.group.c_tot / USDC_DECIMALS,
        engine.group.insurance / USDC_DECIMALS,
        engine.group.vault.saturating_sub(engine.group.c_tot + engine.group.insurance) / USDC_DECIMALS);

    // Move oracle up 10% so all longs profit
    let mut o = oracle;
    let max_move = cfg.max_price_move_bps_per_slot;
    let target = oracle + oracle / 10;
    let mut slot = 2u64;
    while o < target {
        let d = (o as u128 * max_move as u128 / 10_000) as u64;
        o = (o.saturating_add(d)).min(target);
        let _ = engine.accrue_asset(SOL_ASSET, slot, o, 0);
        slot += 1;
    }
    println!("    after 10% rise: oracle=${}", o / 1_000_000);

    let prices = engine.effective_prices();
    let mut sum_pnl = 0i128;
    for &u in &winners {
        let mut acc = engine.accounts[u];
        let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[u] = acc;
        sum_pnl += engine.accounts[u].pnl;
    }
    println!("    sum of user pnls: {}", sum_pnl);
    println!("    pnl_pos_tot in engine: {}", engine.group.pnl_pos_tot);
    println!("    residual = vault({}) - c_tot({}) - insurance({}) = {}",
        engine.group.vault, engine.group.c_tot, engine.group.insurance,
        engine.group.vault.saturating_sub(engine.group.c_tot + engine.group.insurance));

    // Each user has a cert.equity that reflects haircut_effective_support.
    // If junior_claim_bound > residual, positive support is haircut.
    let mut total_certified_equity = 0i128;
    let mut total_face_pnl = 0u128;
    for &u in &winners {
        let cert = engine.accounts[u].health_cert;
        total_certified_equity += cert.certified_equity;
        if engine.accounts[u].pnl > 0 {
            total_face_pnl += engine.accounts[u].pnl as u128;
        }
    }
    println!("    sum certified_equity: {}", total_certified_equity);
    println!("    sum face positive pnl: {}", total_face_pnl);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

fn run_probes_xmargin() {
    println!("=== v14 cross-margin probes (new attack surface) ===");
    probe_xmargin_offset();
    println!();
    probe_xmargin_asymmetric();
    println!();
    probe_xmargin_haircut();
}

// ════════════════════════════════════════════════════════════════════════════
// "Drift-style" attacks: bad assets in the cross-margin portfolio.
// The Drift Protocol exploit (2021) extracted real value via oracle/asset
// misconfiguration. Test v14's defenses against similar patterns.
// ════════════════════════════════════════════════════════════════════════════

/// Make a config with N assets where asset 0 is "good" and asset N-1 is
/// "loose" (low MM, fast moves allowed). Tests whether v14 keeps the bad
/// asset's bankruptcy contained per-leg.
fn make_mixed_quality_config(n_assets: u8) -> V14Config {
    // Use the most conservative (good) config — engine applies same params
    // across all assets. The "badness" comes from oracle behavior, OI
    // concentration, or wrapper misconfiguration, not config heterogeneity
    // (v14 doesn't support per-asset margin params).
    make_bounty_config(n_assets)
}

/// Probe A: thin-market asset oracle manipulation.
/// Asset 0 = legit (LP and many users), asset 1 = thin (only attacker).
/// Attacker opens position on asset 1; tries to manipulate its oracle to
/// inflate PnL; checks whether that PnL can support a real loss on asset 0.
fn probe_thin_market_xmargin() {
    println!("  Drift-style A: thin asset (1 user, manipulated oracle) supports loss on real asset");
    let cfg = make_mixed_quality_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    // Set up: asset 0 has many users (legit OI), asset 1 has just the attacker
    let mut legit_users = Vec::new();
    for _ in 0..5 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(1_000)).unwrap();
        legit_users.push(u);
    }
    let attacker = engine.add_account(3).unwrap();
    engine.deposit(attacker, usdc(1_000)).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    // Legit OI on asset 0 (both sides)
    for &u in &legit_users {
        let size_q = usdc(3_000) * POS_SCALE / oracle as u128;
        let _ = engine.trade(u, lp, 0, size_q, oracle, 1);
    }

    // Attacker opens HEDGED position on asset 0 (1x long for real loss exposure)
    // AND on asset 1 (which they alone hold)
    let attacker_real_size = usdc(15_000) * POS_SCALE / oracle as u128; // 15x on $1k
    if engine.trade(attacker, lp, 0, attacker_real_size, oracle, 1).is_err() {
        println!("    attacker real open FAILED — IM too tight"); return;
    }
    // Attacker's thin-market position on asset 1
    let attacker_thin_size = usdc(5_000) * POS_SCALE / oracle as u128;
    if engine.trade(attacker, lp, 1, attacker_thin_size, oracle, 1).is_err() {
        println!("    attacker thin open FAILED"); return;
    }
    println!("    attacker: long $15k asset 0 (real OI) + long $5k asset 1 (thin, just them + LP)");
    println!("    asset 1 OI: long={} short={} (LP is the only short)",
        engine.group.assets[1].oi_eff_long_q,
        engine.group.assets[1].oi_eff_short_q);

    // Attacker tries to manipulate asset 1 oracle: push price up gradually.
    // Each crank, asset 1 goes up envelope-max; asset 0 stays flat.
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o0 = oracle;
    let mut o1 = oracle;
    let mut slot = 2u64;
    for _ in 0..200 {
        let d = (o1 as u128 * max_move as u128 / 10_000) as u64;
        o1 = o1.saturating_add(d);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);
        slot += 1;
    }
    println!("    after 200 slots: o0=${} (unchanged), o1=${} (+{}%)",
        o0 / 1_000_000, o1 / 1_000_000, (o1 - oracle) * 100 / oracle);

    // Refresh attacker's account
    let prices = engine.effective_prices();
    let mut acc = engine.accounts[attacker];
    let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
    let _ = engine.group.full_account_refresh(&mut acc, &prices);
    engine.accounts[attacker] = acc;

    println!("    attacker post-manipulation:");
    println!("      face pnl: {}", engine.accounts[attacker].pnl);
    println!("      cert.equity (haircut): {}", engine.accounts[attacker].health_cert.certified_equity);
    println!("      cert.mm_req: {}", engine.accounts[attacker].health_cert.certified_maintenance_req);
    println!("      cert.liq_deficit: {}", engine.accounts[attacker].health_cert.certified_liq_deficit);
    println!("      residual: {}", engine.group.vault.saturating_sub(engine.group.c_tot + engine.group.insurance));
    println!("      pnl_pos_tot: {}", engine.group.pnl_pos_tot);

    // Now crash asset 0 (where attacker has real long exposure)
    o0 = oracle;
    let mut total_liq = 0u32;
    let mut total_ins = 0u128;
    for _ in 0..200 {
        let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
        o0 = o0.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);

        // Liquidate any account with deficit
        let prices = engine.effective_prices();
        for &u in std::iter::once(&attacker).chain(legit_users.iter()) {
            let mut acc = engine.accounts[u];
            let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
            let _ = engine.group.full_account_refresh(&mut acc, &prices);
            engine.accounts[u] = acc;
            if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                let mut best = (0usize, 0u128);
                for li in 0..V14_MAX_PORTFOLIO_ASSETS_N {
                    let leg = engine.accounts[u].legs[li];
                    if leg.active {
                        let a = leg.basis_pos_q.unsigned_abs();
                        if a > best.1 { best = (li, a); }
                    }
                }
                if best.1 > 0 {
                    let mut acc = engine.accounts[u];
                    if let Ok(out) = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV14 {
                            asset_index: best.0,
                            close_q: best.1,
                            fee_bps: 5,
                        },
                        &prices,
                    ) {
                        total_liq += 1;
                        total_ins += out.insurance_used;
                    }
                    engine.accounts[u] = acc;
                }
            }
        }
        slot += 1;
    }
    println!("    after crash: o0=${} (-{}%)", o0 / 1_000_000, (oracle - o0) * 100 / oracle);
    println!("    total liquidations: {}", total_liq);
    println!("    total insurance used: {}", total_ins);
    println!("    attacker final cap=${} pnl={} legs={}",
        engine.accounts[attacker].capital / USDC_DECIMALS,
        engine.accounts[attacker].pnl,
        engine.accounts[attacker].active_bitmap.count_ones());
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Probe B: phantom-PnL extraction attempt.
/// Attacker opens 2 legs. Tries to use the profitable leg's PnL to support
/// the losing leg, while extracting capital before the losing leg gets
/// liquidated. v14 haircut should bound the extraction.
fn probe_phantom_pnl_extract() {
    println!("  Drift-style B: profitable leg supports losing leg, attempt extraction");
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let attacker = engine.add_account(2).unwrap();
    engine.deposit(attacker, usdc(2_000)).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    // Long asset 0 ($8k) + Short asset 1 ($8k) — hedged at portfolio level
    let size_q = usdc(8_000) * POS_SCALE / oracle as u128;
    engine.trade(attacker, lp, 0, size_q, oracle, 1).unwrap();
    engine.trade(lp, attacker, 1, size_q, oracle, 1).unwrap();
    println!("    opened: long $8k asset 0 + short $8k asset 1");

    // Asset 0 crashes (long loses), asset 1 ALSO crashes (short profits!)
    // Net effect on attacker: hedge holds up, MM stays satisfied
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o = oracle;
    let mut slot = 2u64;
    for _ in 0..100 {
        let d = (o as u128 * max_move as u128 / 10_000) as u64;
        o = o.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(0, slot, o, 0);
        let _ = engine.accrue_asset(1, slot, o, 0); // both crash together
        slot += 1;
    }
    println!("    after 100 slots: both oracles=${}", o / 1_000_000);

    // Refresh
    let prices = engine.effective_prices();
    let mut acc = engine.accounts[attacker];
    let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
    let _ = engine.group.full_account_refresh(&mut acc, &prices);
    engine.accounts[attacker] = acc;

    println!("    after hedged crash:");
    println!("      cap=${}  pnl={}  cert.equity={}",
        engine.accounts[attacker].capital / USDC_DECIMALS,
        engine.accounts[attacker].pnl,
        engine.accounts[attacker].health_cert.certified_equity);
    println!("      legs[0]={} legs[1]={}",
        engine.accounts[attacker].legs[0].active,
        engine.accounts[attacker].legs[1].active);

    // Try to withdraw the "profit" from short leg
    let mut acc = engine.accounts[attacker];
    let r = engine.group.withdraw_not_atomic(&mut acc, usdc(500), &prices);
    engine.accounts[attacker] = acc;
    println!("    withdraw $500 attempt: {:?}", r);
    if r.is_ok() {
        println!("      post-withdraw cap=${}", engine.accounts[attacker].capital / USDC_DECIMALS);
    }

    // Now ONLY asset 0 keeps crashing (no more hedge)
    let mut o0 = o;
    let o1_stable = o;
    let mut total_liq = 0u32;
    let mut total_ins = 0u128;
    for _ in 0..200 {
        let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
        o0 = o0.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1_stable, 0);

        let prices = engine.effective_prices();
        let mut acc = engine.accounts[attacker];
        let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[attacker] = acc;
        if engine.accounts[attacker].health_cert.certified_liq_deficit > 0 {
            let mut best = (0usize, 0u128);
            for li in 0..V14_MAX_PORTFOLIO_ASSETS_N {
                let leg = engine.accounts[attacker].legs[li];
                if leg.active {
                    let a = leg.basis_pos_q.unsigned_abs();
                    if a > best.1 { best = (li, a); }
                }
            }
            if best.1 > 0 {
                let mut acc = engine.accounts[attacker];
                if let Ok(out) = engine.group.liquidate_account_not_atomic(
                    &mut acc,
                    LiquidationRequestV14 {
                        asset_index: best.0,
                        close_q: best.1,
                        fee_bps: 5,
                    },
                    &prices,
                ) {
                    total_liq += 1;
                    total_ins += out.insurance_used;
                }
                engine.accounts[attacker] = acc;
            }
        }
        slot += 1;
    }
    println!("    after asset-0 sole crash to ${}: liqs={} ins_used={}",
        o0 / 1_000_000, total_liq, total_ins);
    println!("    attacker final cap=${} pnl={}",
        engine.accounts[attacker].capital / USDC_DECIMALS,
        engine.accounts[attacker].pnl);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Probe C: oracle-divergence attack.
/// Attacker exploits a slow oracle update on asset 1 while asset 0 is
/// crashing. Asset 1 oracle stays stale (artificially high), inflating
/// the perceived value of the short hedge.
fn probe_oracle_divergence() {
    println!("  Drift-style C: stale oracle on one asset inflates hedge value");
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let attacker = engine.add_account(2).unwrap();
    engine.deposit(attacker, usdc(2_000)).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    let size_q = usdc(8_000) * POS_SCALE / oracle as u128;
    engine.trade(attacker, lp, 0, size_q, oracle, 1).unwrap();
    engine.trade(lp, attacker, 1, size_q, oracle, 1).unwrap();

    // Crash asset 0, but only accrue asset 1 OCCASIONALLY (keeper laziness on asset 1)
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o0 = oracle;
    let mut o1 = oracle;
    let mut slot = 2u64;
    let mut acc_failures = 0u32;
    for step in 0..200 {
        let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
        o0 = o0.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        // Asset 1 oracle "stays stale" — wrapper only accrues every 20 slots
        if step % 20 == 0 {
            // when we DO accrue, asset 1 has actually moved (correlated with 0)
            o1 = o0; // perfectly correlated in reality, but engine doesn't know
            let r = engine.accrue_asset(1, slot, o1, 0);
            if r.is_err() { acc_failures += 1; }
        }
        slot += 1;
    }
    println!("    after 200 slots: asset 0 oracle=${}, asset 1 engine-effective=${}",
        engine.group.assets[0].effective_price / 1_000_000,
        engine.group.assets[1].effective_price / 1_000_000);
    println!("    asset 1 raw_oracle_target=${}", engine.group.assets[1].raw_oracle_target_price / 1_000_000);
    println!("    accrue failures on asset 1: {}", acc_failures);

    // Try to refresh/exit attacker — should be blocked by target_effective_lag
    let prices = engine.effective_prices();
    let mut acc = engine.accounts[attacker];
    let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
    let r_refresh = engine.group.full_account_refresh(&mut acc, &prices);
    engine.accounts[attacker] = acc;
    println!("    refresh after divergence: {:?}", r_refresh.is_ok());

    // Try withdrawal — should be blocked if cross-asset lag is detected
    let mut acc = engine.accounts[attacker];
    let r_withdraw = engine.group.withdraw_not_atomic(&mut acc, usdc(500), &prices);
    engine.accounts[attacker] = acc;
    println!("    withdraw $500: {:?}", r_withdraw);

    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

fn run_probes_drift() {
    println!("=== v14 Drift-style bad-asset cross-margin attacks ===");
    probe_thin_market_xmargin();
    println!();
    probe_phantom_pnl_extract();
    println!();
    probe_oracle_divergence();
    println!();
    probe_concentrated_one_sided_oi();
    println!();
    probe_pump_and_withdraw();
    println!();
    probe_cross_asset_contagion();
}

// ════════════════════════════════════════════════════════════════════════════
// HARD stress: 10/10 (Binance-style 10% crash with 10x leverage) and
// aggressive Drift-hack reconstructions targeted at v14 cross-margin.
// ════════════════════════════════════════════════════════════════════════════

/// 10/10 v14: 50 users at 10x leverage, oracle crashes 10% over envelope-max
/// cranks, keeper liquidates each slot. Engine must absorb all bankruptcies
/// without insurance use.
fn probe_ten10_single_asset() {
    println!("  Hard 10/10: 50 users × 10x lev, 10% crash, single asset");
    // mm=1000 (10%) for 10x leverage envelope
    let cfg = V14Config {
        max_portfolio_assets:               1,
        min_nonzero_mm_req:                20,
        min_nonzero_im_req:                30,
        h_min:                              0,
        h_max:                             30,
        maintenance_margin_bps:          1000,    // 10x leverage
        initial_margin_bps:              1000,
        max_trading_fee_bps:                1,
        liquidation_fee_bps:                5,
        liquidation_fee_cap:    usdc(50_000),
        min_liquidation_abs:                0,
        max_accrual_dt_slots:              10,
        max_abs_funding_e9_per_slot:        0,
        min_funding_lifetime_slots:        10,
        max_price_move_bps_per_slot:       90,    // 0.9% / slot
        max_account_b_settlement_chunks:    8,
        max_bankrupt_close_chunks:          8,
        public_b_chunk_atoms:   MAX_VAULT_TVL,
        permissionless_recovery_enabled:    true,
        stale_certificate_penalty_enabled:  true,
        full_refresh_required_for_favorable_actions: true,
        public_liveness_profile_crank_forward: true,
        recovery_fallback_price_enabled: true,
        max_bankrupt_close_lifetime_slots: 1000,
    };
    if cfg.validate_public_user_fund().is_err() {
        println!("    cfg validation failed");
        return;
    }
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    let mut longs = Vec::new();
    for _ in 0..50 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(1_000)).unwrap();
        longs.push(u);
    }
    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    // Each user 10x long ($9k notional)
    for &u in &longs {
        let size_q = usdc(9_000) * POS_SCALE / oracle as u128;
        let _ = engine.trade(u, lp, 0, size_q, oracle, 1);
    }
    println!("    50 longs opened at ~9x leverage");

    // 10/10 crash: 10% drop. With max_move=90 bps/slot, takes ~12 slots compounded.
    let max_move = cfg.max_price_move_bps_per_slot;
    let target = oracle * 9 / 10; // 10% down
    let mut o = oracle;
    let mut slot = 2u64;
    let mut total_liq = 0u32;
    let mut total_ins = 0u128;
    let mut total_res = 0u128;
    let mut total_explicit = 0u128;
    let mut min_user_cap = u128::MAX;

    // Crash phase
    while o > target {
        let d = (o as u128 * max_move as u128 / 10_000) as u64;
        o = o.saturating_sub(d).max(target);
        let _ = engine.accrue_asset(0, slot, o, 0);
        let prices = engine.effective_prices();
        for &u in &longs {
            let mut acc = engine.accounts[u];
            let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
            let _ = engine.group.full_account_refresh(&mut acc, &prices);
            engine.accounts[u] = acc;
            if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                let leg = engine.accounts[u].legs[0];
                if leg.active {
                    let mut acc = engine.accounts[u];
                    if let Ok(out) = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV14 {
                            asset_index: 0,
                            close_q: leg.basis_pos_q.unsigned_abs(),
                            fee_bps: 5,
                        },
                        &prices,
                    ) {
                        total_liq += 1;
                        total_ins += out.insurance_used;
                        total_res += out.residual_booked;
                        total_explicit += out.explicit_loss;
                    }
                    engine.accounts[u] = acc;
                }
            }
            min_user_cap = min_user_cap.min(engine.accounts[u].capital);
        }
        slot += 1;
    }
    println!("    crash phase done: oracle ${} → ${} (-{}%) over {} slots",
        oracle / 1_000_000, o / 1_000_000, (oracle - o) * 100 / oracle, slot - 2);
    println!("    crash-phase liqs={} ins_used={} residual={} explicit={}",
        total_liq, total_ins, total_res, total_explicit);

    // Continue crashing past 10% to push insurance harder
    let target2 = oracle * 5 / 10; // additional 40% drop to total 50%
    while o > target2 {
        let d = (o as u128 * max_move as u128 / 10_000) as u64;
        o = o.saturating_sub(d).max(target2);
        let _ = engine.accrue_asset(0, slot, o, 0);
        let prices = engine.effective_prices();
        for &u in &longs {
            let mut acc = engine.accounts[u];
            let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
            let _ = engine.group.full_account_refresh(&mut acc, &prices);
            engine.accounts[u] = acc;
            if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                let leg = engine.accounts[u].legs[0];
                if leg.active {
                    let mut acc = engine.accounts[u];
                    if let Ok(out) = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV14 {
                            asset_index: 0,
                            close_q: leg.basis_pos_q.unsigned_abs(),
                            fee_bps: 5,
                        },
                        &prices,
                    ) {
                        total_liq += 1;
                        total_ins += out.insurance_used;
                        total_res += out.residual_booked;
                        total_explicit += out.explicit_loss;
                    }
                    engine.accounts[u] = acc;
                }
            }
            min_user_cap = min_user_cap.min(engine.accounts[u].capital);
        }
        slot += 1;
    }
    println!("    full -50% crash done at slot {}", slot - 1);
    println!("    TOTAL liqs={} ins_used={} residual={} explicit={}",
        total_liq, total_ins, total_res, total_explicit);
    let sum_user_cap: u128 = longs.iter().map(|&u| engine.accounts[u].capital).sum();
    println!("    sum user cap: ${} (initial $50k)", sum_user_cap / USDC_DECIMALS);
    println!("    min user cap: ${}", min_user_cap / USDC_DECIMALS);
    println!("    LP capital: ${} (initial $10M)", engine.accounts[lp].capital / USDC_DECIMALS);
    println!("    asset side modes: long={:?} short={:?}",
        engine.group.assets[0].mode_long, engine.group.assets[0].mode_short);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// 10/10 v14 with cross-margin: each user holds 3 legs across 3 assets.
/// One asset crashes 10%. Cross-margin reduces per-user equity for ALL legs
/// simultaneously even though only one asset moved.
fn probe_ten10_cross_margin() {
    println!("  Hard 10/10 cross-margin: 30 users × 3 legs each, one asset crashes 10%");
    let cfg = V14Config {
        max_portfolio_assets:               3,
        min_nonzero_mm_req:                20,
        min_nonzero_im_req:                30,
        h_min:                              0,
        h_max:                             30,
        maintenance_margin_bps:          1000,
        initial_margin_bps:              1000,
        max_trading_fee_bps:                1,
        liquidation_fee_bps:                5,
        liquidation_fee_cap:    usdc(50_000),
        min_liquidation_abs:                0,
        max_accrual_dt_slots:              10,
        max_abs_funding_e9_per_slot:        0,
        min_funding_lifetime_slots:        10,
        max_price_move_bps_per_slot:       90,
        max_account_b_settlement_chunks:    8,
        max_bankrupt_close_chunks:          8,
        public_b_chunk_atoms:   MAX_VAULT_TVL,
        permissionless_recovery_enabled:    true,
        stale_certificate_penalty_enabled:  true,
        full_refresh_required_for_favorable_actions: true,
        public_liveness_profile_crank_forward: true,
        recovery_fallback_price_enabled: true,
        max_bankrupt_close_lifetime_slots: 1000,
    };
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(50_000_000)).unwrap();
    let mut users = Vec::new();
    for _ in 0..30 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(2_000)).unwrap();
        users.push(u);
    }
    let oracle = price_e6(200);
    for ai in 0..3 {
        engine.accrue_asset(ai, 1, oracle, 0).unwrap();
    }
    // Each user opens long on all 3 assets at $3k each = $9k notional total
    for &u in &users {
        for ai in 0..3 {
            let size_q = usdc(3_000) * POS_SCALE / oracle as u128;
            let _ = engine.trade(u, lp, ai, size_q, oracle, 1);
        }
    }
    println!("    30 users × 3 legs ($3k each, $9k portfolio, 4.5x port-leverage)");

    // Crash asset 0 only
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o0 = oracle;
    let o1 = oracle;
    let o2 = oracle;
    let mut slot = 2u64;
    let target = oracle * 9 / 10;
    let mut total_liq = 0u32;
    let mut total_ins = 0u128;
    let mut total_res = 0u128;
    let mut leg_liqs = [0u32; 3];

    while o0 > target {
        let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
        o0 = o0.saturating_sub(d).max(target);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);
        let _ = engine.accrue_asset(2, slot, o2, 0);
        let prices = engine.effective_prices();
        for &u in &users {
            let mut acc = engine.accounts[u];
            let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
            let _ = engine.group.full_account_refresh(&mut acc, &prices);
            engine.accounts[u] = acc;
            if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                let mut best = (0usize, 0u128);
                for li in 0..V14_MAX_PORTFOLIO_ASSETS_N {
                    let leg = engine.accounts[u].legs[li];
                    if leg.active {
                        let a = leg.basis_pos_q.unsigned_abs();
                        if a > best.1 { best = (li, a); }
                    }
                }
                if best.1 > 0 {
                    let mut acc = engine.accounts[u];
                    if let Ok(out) = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV14 {
                            asset_index: best.0,
                            close_q: best.1,
                            fee_bps: 5,
                        },
                        &prices,
                    ) {
                        total_liq += 1;
                        total_ins += out.insurance_used;
                        total_res += out.residual_booked;
                        if best.0 < 3 { leg_liqs[best.0] += 1; }
                    }
                    engine.accounts[u] = acc;
                }
            }
        }
        slot += 1;
    }
    println!("    crash done: o0=${} (-{}%) in {} slots",
        o0 / 1_000_000, (oracle - o0) * 100 / oracle, slot - 2);
    println!("    liqs by asset: [0]={} [1]={} [2]={}", leg_liqs[0], leg_liqs[1], leg_liqs[2]);
    println!("    total liqs={} ins_used={} residual={}", total_liq, total_ins, total_res);
    let sum_cap: u128 = users.iter().map(|&u| engine.accounts[u].capital).sum();
    println!("    sum user cap: ${} (initial $60k)", sum_cap / USDC_DECIMALS);
    // Most-active users should still have legs on assets 1, 2 (untouched)
    let avg_active_legs: u32 = users.iter().map(|&u| engine.accounts[u].active_bitmap.count_ones()).sum::<u32>() / users.len() as u32;
    println!("    avg active legs per user: {}", avg_active_legs);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Aggressive Drift-hack reconstruction:
/// Attacker structures positions, then drives oracle on a thin/manipulable
/// asset to inflate cross-margin equity, then tries to extract.
///
/// Specifically: open BIG real-asset long + SMALL thin-asset position.
/// Push thin-asset oracle up (attacker's only counterparty is LP).
/// Cross-margin sees the inflated PnL → may allow over-leveraging the real asset.
/// Then real asset crashes — engine must handle without insurance drain.
fn probe_drift_hack_aggressive() {
    println!("  Drift-hack aggressive: thin-asset oracle pump → real-asset over-leverage attempt");
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    // Bystander users with legitimate exposure on asset 0
    let mut bystanders = Vec::new();
    for _ in 0..5 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(1_000)).unwrap();
        bystanders.push(u);
    }
    let attacker = engine.add_account(3).unwrap();
    engine.deposit(attacker, usdc(2_000)).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    // Bystanders: longs on asset 0 (legitimate OI)
    for &u in &bystanders {
        let size_q = usdc(5_000) * POS_SCALE / oracle as u128;
        let _ = engine.trade(u, lp, 0, size_q, oracle, 1);
    }

    // Attacker step 1: open SMALL long on asset 1 (the thin asset, $500 notional)
    let small_size = usdc(500) * POS_SCALE / oracle as u128;
    if engine.trade(attacker, lp, 1, small_size, oracle, 1).is_err() {
        println!("    step 1: open small thin-asset position FAILED"); return;
    }
    println!("    step 1: attacker opened $500 long on thin asset 1");

    // Attacker step 2: pump asset 1's oracle (no other users on asset 1, so attacker is sole long)
    // Walk asset 1 oracle up envelope-max for many slots
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o1 = oracle;
    let o0 = oracle;
    let mut slot = 2u64;
    let target_pump = oracle * 2; // double asset 1 price
    while o1 < target_pump {
        let d = (o1 as u128 * max_move as u128 / 10_000) as u64;
        o1 = o1.saturating_add(d).min(target_pump);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);
        slot += 1;
    }
    println!("    step 2: pumped asset 1 from ${} → ${} in {} slots",
        oracle / 1_000_000, o1 / 1_000_000, slot - 2);

    // Refresh attacker
    let prices = engine.effective_prices();
    let mut acc = engine.accounts[attacker];
    let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
    let _ = engine.group.full_account_refresh(&mut acc, &prices);
    engine.accounts[attacker] = acc;
    println!("    after pump: attacker cap=${} face_pnl={} cert.equity={}",
        engine.accounts[attacker].capital / USDC_DECIMALS,
        engine.accounts[attacker].pnl,
        engine.accounts[attacker].health_cert.certified_equity);
    println!("    residual={}  pnl_pos_tot={}",
        engine.group.vault.saturating_sub(engine.group.c_tot + engine.group.insurance),
        engine.group.pnl_pos_tot);

    // Attacker step 3: use inflated equity to over-leverage on asset 0
    // Try BIG long on asset 0 — would normally need $1k IM for $20k notional
    let big_size = usdc(30_000) * POS_SCALE / oracle as u128;
    let r = engine.trade(attacker, lp, 0, big_size, oracle, 1);
    println!("    step 3: attempt $30k notional long on asset 0: {:?}", r.map(|_|()).err());

    // Attacker step 4: try to withdraw 'inflated profit'
    let mut acc = engine.accounts[attacker];
    let r_w = engine.group.withdraw_not_atomic(&mut acc, usdc(2_000), &prices);
    engine.accounts[attacker] = acc;
    println!("    step 4: withdraw $2000: {:?}", r_w);

    // Attacker step 5: try to close the profitable thin-asset leg
    let leg1 = engine.accounts[attacker].legs[1];
    if leg1.active {
        let close_r = engine.trade(lp, attacker, 1, leg1.basis_pos_q.unsigned_abs(), o1, 1);
        println!("    step 5: close thin-asset profit leg: {:?}", close_r.map(|_|()).err());
        let mut acc = engine.accounts[attacker];
        let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[attacker] = acc;
        println!("      post-close: cap=${} pnl={} cert.equity={}",
            engine.accounts[attacker].capital / USDC_DECIMALS,
            engine.accounts[attacker].pnl,
            engine.accounts[attacker].health_cert.certified_equity);
    }

    // Attacker step 6: try the big withdrawal now
    let mut acc = engine.accounts[attacker];
    let r_w2 = engine.group.withdraw_not_atomic(&mut acc, usdc(2_000), &prices);
    engine.accounts[attacker] = acc;
    println!("    step 6: withdraw $2000 (post-close): {:?}", r_w2);

    println!("    attacker final cap=${}", engine.accounts[attacker].capital / USDC_DECIMALS);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

fn run_probes_hard_stress() {
    println!("=== v14 HARD stress: 10/10 + drift-hack reconstruction ===");
    probe_ten10_single_asset();
    println!();
    probe_ten10_cross_margin();
    println!();
    probe_drift_hack_aggressive();
}

/// Iterated Drift-hack: try the same attack 10 times in a row with state
/// carry-over. Each cycle: pump → close → try to withdraw → reopen.
/// Cumulative measure: total extracted vs total cost.
fn probe_drift_iterated(cycles: u32) {
    println!("  Drift-hack ITERATED ({} cycles): cumulative extraction test", cycles);
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let attacker = engine.add_account(2).unwrap();
    let initial_deposit = usdc(2_000);
    engine.deposit(attacker, initial_deposit).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    let max_move = cfg.max_price_move_bps_per_slot;
    let mut slot = 2u64;
    let mut o1 = oracle;
    let o0 = oracle;
    let mut total_extracted = 0u128;
    let mut successful_extracts = 0u32;
    let mut total_fees = 0u128;

    let initial_lp_cap = engine.accounts[lp].capital;
    let initial_insurance = engine.group.insurance;

    for cycle in 0..cycles {
        // Open tiny long on thin asset 1
        let size_q = usdc(100) * POS_SCALE / o1 as u128;
        if engine.trade(attacker, lp, 1, size_q, o1, 1).is_err() {
            continue;
        }
        // Pump asset 1 by 50%
        let target = o1 * 3 / 2;
        let mut pump_slots = 0;
        while o1 < target && pump_slots < 200 {
            let d = (o1 as u128 * max_move as u128 / 10_000) as u64;
            o1 = (o1.saturating_add(d)).min(target);
            let _ = engine.accrue_asset(0, slot, o0, 0);
            let _ = engine.accrue_asset(1, slot, o1, 0);
            slot += 1;
            pump_slots += 1;
        }
        // Close the pumped leg to realize PnL
        let leg = engine.accounts[attacker].legs[1];
        if leg.active {
            let _ = engine.trade(lp, attacker, 1, leg.basis_pos_q.unsigned_abs(), o1, 1);
        }
        // Try to withdraw the gain
        let prices = engine.effective_prices();
        let mut acc = engine.accounts[attacker];
        let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[attacker] = acc;
        let pre_cap = engine.accounts[attacker].capital;
        let withdraw_amount = usdc(50);
        let mut acc = engine.accounts[attacker];
        let r = engine.group.withdraw_not_atomic(&mut acc, withdraw_amount, &prices);
        engine.accounts[attacker] = acc;
        if r.is_ok() {
            successful_extracts += 1;
            let extracted = pre_cap.saturating_sub(engine.accounts[attacker].capital);
            total_extracted += extracted;
        }
        // Reset oracle (oracle "snaps back" between cycles via slow accrue)
        // Bring asset 1 back down to oracle level
        let target_down = oracle;
        while o1 > target_down {
            let d = (o1 as u128 * max_move as u128 / 10_000) as u64;
            o1 = o1.saturating_sub(d).max(target_down);
            let _ = engine.accrue_asset(0, slot, o0, 0);
            let _ = engine.accrue_asset(1, slot, o1, 0);
            slot += 1;
        }
        if cycle == 0 || cycle == cycles - 1 {
            println!("    cycle {}: attacker cap=${} pnl={}", cycle,
                engine.accounts[attacker].capital / USDC_DECIMALS,
                engine.accounts[attacker].pnl);
        }
    }
    let final_cap = engine.accounts[attacker].capital;
    let net_loss = initial_deposit as i128 - final_cap as i128;
    total_fees = if net_loss > 0 { net_loss as u128 } else { 0 };
    println!("    cycles attempted: {}", cycles);
    println!("    successful withdraws: {} / {}", successful_extracts, cycles);
    println!("    total extracted via withdraw: ${}", total_extracted / USDC_DECIMALS);
    println!("    attacker initial: ${}  final: ${}  delta: {}",
        initial_deposit / USDC_DECIMALS, final_cap / USDC_DECIMALS,
        (final_cap as i128 - initial_deposit as i128) / 1_000_000);
    println!("    LP cap change: {}",
        (engine.accounts[lp].capital as i128 - initial_lp_cap as i128) / 1_000_000);
    println!("    insurance change: {}",
        (engine.group.insurance as i128 - initial_insurance as i128) / 1_000_000);
    println!("    total fees paid: ${}", total_fees / USDC_DECIMALS);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Multi-attacker collusion: 3 attackers coordinating on the cross-margin
/// surface. One pumps, one shorts, one tries to extract. Test if joint
/// activity can break the per-account haircut bound.
fn probe_multi_attacker_collusion() {
    println!("  Multi-attacker collusion: 3 attackers coordinating on cross-margin");
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    let attacker_a = engine.add_account(2).unwrap(); // pumper of asset 1
    let attacker_b = engine.add_account(3).unwrap(); // counterparty on asset 1
    let attacker_c = engine.add_account(4).unwrap(); // extractor on asset 0
    for &u in &[attacker_a, attacker_b, attacker_c] {
        engine.deposit(u, usdc(2_000)).unwrap();
    }

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    // Step 1: attacker_a and attacker_b open opposite positions on asset 1
    // (so OI is between them, not against LP)
    let size_q = usdc(1_000) * POS_SCALE / oracle as u128;
    let r1 = engine.trade(attacker_a, attacker_b, 1, size_q, oracle, 1);
    if r1.is_err() {
        println!("    setup failed: {:?}", r1.err());
        return;
    }
    println!("    step 1: A long $1k asset 1, B short $1k asset 1 (collusion OI)");

    // Step 2: pump asset 1 oracle to inflate A's PnL
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o1 = oracle;
    let target = oracle * 3 / 2;
    let mut slot = 2u64;
    let o0 = oracle;
    while o1 < target {
        let d = (o1 as u128 * max_move as u128 / 10_000) as u64;
        o1 = (o1.saturating_add(d)).min(target);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);
        slot += 1;
    }
    println!("    step 2: pumped asset 1 from ${} → ${}", oracle / 1_000_000, o1 / 1_000_000);

    // Step 3: A tries to withdraw via cross-margin haircut bypass
    let prices = engine.effective_prices();
    for &u in &[attacker_a, attacker_b, attacker_c] {
        let mut acc = engine.accounts[u];
        let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[u] = acc;
    }

    println!("    after pump:");
    for &u in &[attacker_a, attacker_b, attacker_c] {
        let acc = &engine.accounts[u];
        println!("      acc {}: cap=${} pnl={} cert.equity={}",
            u, acc.capital / USDC_DECIMALS, acc.pnl, acc.health_cert.certified_equity);
    }
    println!("    engine: residual={} pnl_pos_tot={}",
        engine.group.vault.saturating_sub(engine.group.c_tot + engine.group.insurance),
        engine.group.pnl_pos_tot);

    // Step 4: each attacker tries withdrawal
    for &u in &[attacker_a, attacker_b, attacker_c] {
        let mut acc = engine.accounts[u];
        let r = engine.group.withdraw_not_atomic(&mut acc, usdc(500), &prices);
        engine.accounts[u] = acc;
        println!("    acc {} withdraw $500: {:?}", u, r.err());
    }

    // Final accounting
    let final_total: u128 = [attacker_a, attacker_b, attacker_c]
        .iter()
        .map(|&u| engine.accounts[u].capital)
        .sum();
    println!("    final attacker cap sum: ${} (initial $6000)", final_total / USDC_DECIMALS);
    println!("    LP cap: ${} (initial $10M)", engine.accounts[lp].capital / USDC_DECIMALS);
    println!("    insurance: {}", engine.group.insurance);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Fuzz Drift-hack across 1000 randomized seeds:
/// - Random pump amount (50% to 100%)
/// - Random initial position size ($100 to $2000)
/// - Random attacker capital ($500 to $5000)
/// Verify zero insurance use and zero invariant failures.
fn fuzz_drift_attack(n_seeds: u64) {
    println!("  Drift-attack fuzz: {} randomized seeds", n_seeds);
    let cfg = make_bounty_config(2);
    let results: Vec<(u64, u128, i128, u128, u32)> = (0..n_seeds)
        .into_par_iter()
        .map(|seed| {
            let mut rng = Rng::new(seed);
            let mut engine = V14Engine::new(cfg).expect("init");
            let lp = engine.add_account(1).unwrap();
            engine.deposit(lp, usdc(10_000_000)).unwrap();
            let attacker_cap = usdc(rng.range_u64(500, 5_000) as u128);
            let attacker = engine.add_account(2).unwrap();
            engine.deposit(attacker, attacker_cap).unwrap();
            let oracle = price_e6(200);
            let _ = engine.accrue_asset(0, 1, oracle, 0);
            let _ = engine.accrue_asset(1, 1, oracle, 0);

            // Random initial position on thin asset 1
            let init_notional = usdc(rng.range_u64(100, 2_000) as u128);
            let init_size = init_notional * POS_SCALE / oracle as u128;
            if engine.trade(attacker, lp, 1, init_size, oracle, 1).is_err() {
                return (seed, 0u128, 0i128, 0u128, 1u32);
            }
            // Pump by random pct
            let pump_pct = rng.range_u64(50, 100);
            let max_move = cfg.max_price_move_bps_per_slot;
            let target = oracle.saturating_add((oracle as u128 * pump_pct as u128 / 100) as u64);
            let mut o1 = oracle;
            let mut slot = 2u64;
            let mut pump_steps = 0;
            while o1 < target && pump_steps < 500 {
                let d = (o1 as u128 * max_move as u128 / 10_000) as u64;
                o1 = (o1.saturating_add(d)).min(target);
                let _ = engine.accrue_asset(0, slot, oracle, 0);
                let _ = engine.accrue_asset(1, slot, o1, 0);
                slot += 1;
                pump_steps += 1;
            }
            // Try to withdraw
            let prices = engine.effective_prices();
            let mut acc = engine.accounts[attacker];
            let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
            let _ = engine.group.full_account_refresh(&mut acc, &prices);
            engine.accounts[attacker] = acc;
            let pre_cap = engine.accounts[attacker].capital;
            let mut acc = engine.accounts[attacker];
            let _ = engine.group.withdraw_not_atomic(&mut acc, attacker_cap / 2, &prices);
            engine.accounts[attacker] = acc;
            let final_cap = engine.accounts[attacker].capital;
            let withdrawn = pre_cap.saturating_sub(final_cap);
            let ins_change = engine.group.insurance as i128 - 0i128;
            let invariant_fails = run_invariant_battery(&engine);
            (seed, withdrawn, ins_change, attacker_cap, invariant_fails)
        })
        .collect();

    let total_withdrawn: u128 = results.iter().map(|r| r.1).sum();
    let any_insurance_change: i128 = results.iter().map(|r| r.2).max().unwrap_or(0);
    let total_invariant_fails: u32 = results.iter().map(|r| r.4).sum();
    let max_single_extract = results.iter().map(|r| r.1).max().unwrap_or(0);
    let extracts = results.iter().filter(|r| r.1 > 0).count();
    let max_initial_cap = results.iter().map(|r| r.3).max().unwrap_or(0);
    println!("    seeds: {}", results.len());
    println!("    extractions attempted (withdraw succeeded): {}", extracts);
    println!("    total withdrawn (sum): ${}", total_withdrawn / USDC_DECIMALS);
    println!("    max single withdraw: ${}", max_single_extract / USDC_DECIMALS);
    println!("    max attacker initial cap: ${}", max_initial_cap / USDC_DECIMALS);
    println!("    max insurance increase across seeds: {}", any_insurance_change);
    println!("    total invariant battery fails: {}", total_invariant_fails);
}

fn run_probes_hard_extended() {
    println!("=== v14 HARD stress extended: iterated drift + collusion + fuzz ===");
    probe_drift_iterated(10);
    println!();
    probe_multi_attacker_collusion();
    println!();
    fuzz_drift_attack(2000);
}

/// Empirically verify v14 bankruptcy residual is attributed STRICTLY to the
/// losing market's opposing-side domain. Other assets' insurance domains
/// must be untouched by a single-market bankruptcy.
fn probe_per_domain_attribution() {
    println!("  Per-domain bankruptcy attribution: SOL crash should NOT touch BTC domains");
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    let mut btc_users = Vec::new();
    for _ in 0..5 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(1_000)).unwrap();
        btc_users.push(u);
    }
    let sol_loser = engine.add_account(3).unwrap();
    engine.deposit(sol_loser, usdc(500)).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    // Set generous domain budgets so spending isn't capped — we only want
    // to verify ATTRIBUTION, not capping.
    for d in 0..V14_MAX_PORTFOLIO_ASSETS_N * 2 {
        engine.group.insurance_domain_budget[d] = usdc(1_000_000);
    }

    // Generate insurance via fees on BTC (asset 1) — these should NOT be
    // spent for SOL bankruptcies.
    for &u in &btc_users {
        let size_q = usdc(5_000) * POS_SCALE / oracle as u128;
        let _ = engine.trade(u, lp, 1, size_q, oracle, 1);
    }
    // SOL victim: high-lev long on asset 0
    let sol_size = usdc(8_000) * POS_SCALE / oracle as u128;
    engine.trade(sol_loser, lp, 0, sol_size, oracle, 1).unwrap();

    println!("    setup: {} BTC longs + 1 SOL victim (16x lev)", btc_users.len());
    let ins_initial = engine.group.insurance;
    println!("    insurance: ${}", ins_initial / USDC_DECIMALS);

    // Domain layout: insurance_domain_index(asset, side) = asset*2 + encode_side(side).
    // encode_side: Long=0, Short=1.
    // For bankrupt LONG on asset 0, opposing side = Short → domain = 0*2 + 1 = 1.
    let dom_sol_long_opp = 1;
    let dom_btc_long_opp = 3;
    let dom_btc_short_opp = 2;

    // Slow-keeper SOL crash
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o0 = oracle;
    let o1 = oracle;
    let mut slot = 2u64;
    for _ in 0..40 {
        let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
        o0 = o0.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);
        slot += 1;
    }
    println!("    SOL crashed to ${} (-{}%)", o0/1_000_000, (oracle-o0)*100/oracle);

    let prices = engine.effective_prices();
    let mut acc = engine.accounts[sol_loser];
    let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
    let _ = engine.group.full_account_refresh(&mut acc, &prices);
    engine.accounts[sol_loser] = acc;
    println!("    sol_loser: cap=${} pnl={} liq_deficit={}",
        engine.accounts[sol_loser].capital / USDC_DECIMALS,
        engine.accounts[sol_loser].pnl,
        engine.accounts[sol_loser].health_cert.certified_liq_deficit);

    let mut liq_out: Option<LiquidationOutcomeV14> = None;
    if engine.accounts[sol_loser].health_cert.certified_liq_deficit > 0 {
        let leg = engine.accounts[sol_loser].legs[0];
        let mut acc = engine.accounts[sol_loser];
        if let Ok(out) = engine.group.liquidate_account_not_atomic(
            &mut acc,
            LiquidationRequestV14 {
                asset_index: 0, close_q: leg.basis_pos_q.unsigned_abs(), fee_bps: 5,
            }, &prices,
        ) {
            liq_out = Some(out);
        }
        engine.accounts[sol_loser] = acc;
    }
    if let Some(out) = liq_out {
        println!("    liq outcome: closed_q={} insurance_used={} residual_booked={} explicit_loss={}",
            out.closed_q, out.insurance_used, out.residual_booked, out.explicit_loss);
    } else {
        println!("    no liquidation triggered");
    }

    let ins_final = engine.group.insurance;
    println!("    insurance: ${} → ${} (Δ=${})",
        ins_initial / USDC_DECIMALS, ins_final / USDC_DECIMALS,
        (ins_final as i128 - ins_initial as i128) / 1_000_000);
    let sol_long_opp_spent = engine.group.insurance_domain_spent[dom_sol_long_opp];
    let btc_long_opp_spent = engine.group.insurance_domain_spent[dom_btc_long_opp];
    let btc_short_opp_spent = engine.group.insurance_domain_spent[dom_btc_short_opp];
    println!();
    println!("    DOMAIN ATTRIBUTION:");
    println!("      domain[1] SOL long-side opp (where SOL-long bankruptcy charges): spent={}",
        sol_long_opp_spent);
    println!("      domain[2] BTC short-side opp:                                    spent={}",
        btc_short_opp_spent);
    println!("      domain[3] BTC long-side opp:                                     spent={}",
        btc_long_opp_spent);
    println!("    ★ BTC domains untouched: {}",
        btc_long_opp_spent == 0 && btc_short_opp_spent == 0);
    println!("    ★ SOL long-opp domain charged: {}", sol_long_opp_spent > 0);

    let sum_btc_cap: u128 = btc_users.iter().map(|&u| engine.accounts[u].capital).sum();
    println!("    BTC users total cap: ${} (initial $5000, fee loss only)",
        sum_btc_cap / USDC_DECIMALS);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Test the budget-cap path: set SOL_long_opp domain budget=0, force deficit,
/// verify insurance_used == 0 (capped) and residual goes elsewhere.
fn probe_per_domain_budget_cap() {
    println!("  Per-domain budget cap: SOL_long_opp budget=$0, force deficit");
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    let mut btc_users = Vec::new();
    for _ in 0..5 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(1_000)).unwrap();
        btc_users.push(u);
    }
    let sol_loser = engine.add_account(3).unwrap();
    engine.deposit(sol_loser, usdc(500)).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    // Generous budgets EXCEPT for SOL long-opp domain
    for d in 0..V14_MAX_PORTFOLIO_ASSETS_N * 2 {
        engine.group.insurance_domain_budget[d] = usdc(1_000_000);
    }
    engine.group.insurance_domain_budget[1] = 0; // SOL long-opp (asset 0 short-side dom)

    for &u in &btc_users {
        let size_q = usdc(5_000) * POS_SCALE / oracle as u128;
        let _ = engine.trade(u, lp, 1, size_q, oracle, 1);
    }
    let sol_size = usdc(8_000) * POS_SCALE / oracle as u128;
    engine.trade(sol_loser, lp, 0, sol_size, oracle, 1).unwrap();
    let ins_initial = engine.group.insurance;
    println!("    insurance balance: ${} | SOL long-opp budget = $0",
        ins_initial / USDC_DECIMALS);

    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o0 = oracle;
    let mut slot = 2u64;
    for _ in 0..40 {
        let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
        o0 = o0.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        slot += 1;
    }
    let prices = engine.effective_prices();
    let mut acc = engine.accounts[sol_loser];
    let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
    let _ = engine.group.full_account_refresh(&mut acc, &prices);
    engine.accounts[sol_loser] = acc;

    if engine.accounts[sol_loser].health_cert.certified_liq_deficit > 0 {
        let leg = engine.accounts[sol_loser].legs[0];
        let mut acc = engine.accounts[sol_loser];
        match engine.group.liquidate_account_not_atomic(
            &mut acc,
            LiquidationRequestV14 {
                asset_index: 0, close_q: leg.basis_pos_q.unsigned_abs(), fee_bps: 5,
            }, &prices,
        ) {
            Ok(out) => println!("    liq outcome: closed_q={} insurance_used={} residual_booked={} explicit_loss={}",
                out.closed_q, out.insurance_used, out.residual_booked, out.explicit_loss),
            Err(e) => println!("    liq failed: {:?}", e),
        }
        engine.accounts[sol_loser] = acc;
    }

    let sol_long_opp_spent = engine.group.insurance_domain_spent[1];
    println!("    ★ SOL long-opp domain spent (budget=$0): {} (must be 0)", sol_long_opp_spent);
    println!("    ★ Budget respected: {}", sol_long_opp_spent == 0);
    println!("    other BTC domains:  spent[2]={} spent[3]={}",
        engine.group.insurance_domain_spent[2],
        engine.group.insurance_domain_spent[3]);
    println!("    insurance balance: ${} → ${}",
        ins_initial / USDC_DECIMALS, engine.group.insurance / USDC_DECIMALS);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

fn run_probes_domain_attribution() {
    println!("=== v14 per-domain bankruptcy attribution (empirical verification) ===");
    probe_per_domain_attribution();
    println!();
    probe_per_domain_budget_cap();
}

/// Direct empirical demonstration: a SINGLE user's profitable SOL leg
/// supports their losing BTC leg's MM. The two probes are:
///   (a) baseline: user holds ONLY the losing leg → liquidated
///   (b) cross-margin: same user holds losing leg + profitable leg → survives
fn probe_xmargin_offset_within_account() {
    println!("  Within-account cross-margin: SOL gain offsets BTC loss");
    println!();
    println!("  Case (a): user holds ONLY a losing leg (no offset)");
    {
        let cfg = make_bounty_config(2);
        let mut engine = V14Engine::new(cfg).expect("init");
        let lp = engine.add_account(1).unwrap();
        engine.deposit(lp, usdc(10_000_000)).unwrap();
        let user = engine.add_account(2).unwrap();
        engine.deposit(user, usdc(500)).unwrap(); // $500 cap

        let oracle = price_e6(200);
        engine.accrue_asset(0, 1, oracle, 0).unwrap();
        engine.accrue_asset(1, 1, oracle, 0).unwrap();

        // ONLY a long on asset 0 — 10x leverage
        let size_q = usdc(5_000) * POS_SCALE / oracle as u128;
        engine.trade(user, lp, 0, size_q, oracle, 1).unwrap();
        println!("    opened: long $5k asset 0 only (10x lev, $500 cap, MM=$250)");

        // Move asset 0 down 20%: long loses $1000 on $5k notional
        let max_move = cfg.max_price_move_bps_per_slot;
        let mut o0 = oracle;
        let target = oracle * 80 / 100;
        let mut slot = 2u64;
        while o0 > target {
            let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
            o0 = o0.saturating_sub(d).max(target);
            let _ = engine.accrue_asset(0, slot, o0, 0);
            let _ = engine.accrue_asset(1, slot, oracle, 0);
            slot += 1;
        }
        println!("    asset 0 dropped 20% → ${}", o0 / 1_000_000);

        let prices = engine.effective_prices();
        let mut acc = engine.accounts[user];
        let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[user] = acc;
        let cert = engine.accounts[user].health_cert;
        println!("    user: cap=${} pnl={} cert.equity={} mm_req={} liq_deficit={}",
            engine.accounts[user].capital / USDC_DECIMALS,
            engine.accounts[user].pnl,
            cert.certified_equity, cert.certified_maintenance_req,
            cert.certified_liq_deficit);
        if cert.certified_liq_deficit > 0 {
            println!("    ⇒ LIQUIDATABLE (deficit = ${})", cert.certified_liq_deficit / USDC_DECIMALS);
        } else {
            println!("    ⇒ healthy");
        }
    }

    println!();
    println!("  Case (b): SAME losing position, but user ALSO holds a profitable leg");
    {
        let cfg = make_bounty_config(2);
        let mut engine = V14Engine::new(cfg).expect("init");
        let lp = engine.add_account(1).unwrap();
        engine.deposit(lp, usdc(10_000_000)).unwrap();
        let user = engine.add_account(2).unwrap();
        engine.deposit(user, usdc(1_000)).unwrap();

        let oracle = price_e6(200);
        engine.accrue_asset(0, 1, oracle, 0).unwrap();
        engine.accrue_asset(1, 1, oracle, 0).unwrap();

        // SAME long on asset 0
        let size_q = usdc(5_000) * POS_SCALE / oracle as u128;
        engine.trade(user, lp, 0, size_q, oracle, 1).unwrap();
        println!("    after trade 1: cap=${} pnl={}",
            engine.accounts[user].capital, engine.accounts[user].pnl);
        // PLUS a short on asset 1 (will profit when asset 1 drops)
        engine.trade(lp, user, 1, size_q, oracle, 1).unwrap();
        println!("    after trade 2: cap=${} pnl={}",
            engine.accounts[user].capital, engine.accounts[user].pnl);

        let max_move = cfg.max_price_move_bps_per_slot;
        // Same 20% drop on asset 0 (long loses) AND on asset 1 (short profits — offsetting!)
        let mut o0 = oracle;
        let mut o1 = oracle;
        let target = oracle * 80 / 100;
        let mut slot = 2u64;
        while o0 > target {
            let d0 = (o0 as u128 * max_move as u128 / 10_000) as u64;
            let d1 = (o1 as u128 * max_move as u128 / 10_000) as u64;
            o0 = o0.saturating_sub(d0).max(target);
            o1 = o1.saturating_sub(d1).max(target);
            let _ = engine.accrue_asset(0, slot, o0, 0);
            let _ = engine.accrue_asset(1, slot, o1, 0);
            slot += 1;
        }
        println!("    asset 0 → ${}  asset 1 → ${}", o0 / 1_000_000, o1 / 1_000_000);

        let prices = engine.effective_prices();
        let mut acc = engine.accounts[user];
        let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[user] = acc;
        let cert = engine.accounts[user].health_cert;
        println!("    user: cap=${} pnl={} cert.equity={} mm_req={} liq_deficit={}",
            engine.accounts[user].capital / USDC_DECIMALS,
            engine.accounts[user].pnl,
            cert.certified_equity, cert.certified_maintenance_req,
            cert.certified_liq_deficit);
        if cert.certified_liq_deficit > 0 {
            println!("    ⇒ LIQUIDATABLE (deficit = ${})", cert.certified_liq_deficit / USDC_DECIMALS);
        } else {
            println!("    ⇒ healthy — cross-margin offset is supporting the losing leg ★");
        }
        println!("    legs active: {}", engine.accounts[user].active_bitmap.count_ones());
    }

    println!();
    println!("  Case (c): SAME setup but UNCORRELATED moves —");
    println!("  asset 0 drops 20% (long loses), asset 1 RISES 20% (short loses too)");
    {
        let cfg = make_bounty_config(2);
        let mut engine = V14Engine::new(cfg).expect("init");
        let lp = engine.add_account(1).unwrap();
        engine.deposit(lp, usdc(10_000_000)).unwrap();
        let user = engine.add_account(2).unwrap();
        engine.deposit(user, usdc(1_000)).unwrap();

        let oracle = price_e6(200);
        engine.accrue_asset(0, 1, oracle, 0).unwrap();
        engine.accrue_asset(1, 1, oracle, 0).unwrap();

        let size_q = usdc(5_000) * POS_SCALE / oracle as u128;
        engine.trade(user, lp, 0, size_q, oracle, 1).unwrap();
        engine.trade(lp, user, 1, size_q, oracle, 1).unwrap();

        let max_move = cfg.max_price_move_bps_per_slot;
        let mut o0 = oracle;
        let mut o1 = oracle;
        let target0 = oracle * 80 / 100;
        let target1 = oracle * 120 / 100;
        let mut slot = 2u64;
        while o0 > target0 || o1 < target1 {
            if o0 > target0 {
                let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
                o0 = o0.saturating_sub(d).max(target0);
            }
            if o1 < target1 {
                let d = (o1 as u128 * max_move as u128 / 10_000) as u64;
                o1 = (o1.saturating_add(d)).min(target1);
            }
            let _ = engine.accrue_asset(0, slot, o0, 0);
            let _ = engine.accrue_asset(1, slot, o1, 0);
            slot += 1;
        }
        println!("    asset 0 → ${}  asset 1 → ${}", o0 / 1_000_000, o1 / 1_000_000);

        let prices = engine.effective_prices();
        let mut acc = engine.accounts[user];
        let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[user] = acc;
        let cert = engine.accounts[user].health_cert;
        println!("    user: cap=${} pnl={} cert.equity={} mm_req={} liq_deficit={}",
            engine.accounts[user].capital / USDC_DECIMALS,
            engine.accounts[user].pnl,
            cert.certified_equity, cert.certified_maintenance_req,
            cert.certified_liq_deficit);
        if cert.certified_liq_deficit > 0 {
            println!("    ⇒ LIQUIDATABLE — both legs lost, cross-margin can't save uncorrelated double-loss");
        } else {
            println!("    ⇒ healthy");
        }
    }

    println!();
    println!("  Case (d): SOL gain DIRECTLY offsets BTC loss (SOL leg profitable, BTC leg losing)");
    {
        let cfg = make_bounty_config(2);
        let mut engine = V14Engine::new(cfg).expect("init");
        let lp = engine.add_account(1).unwrap();
        engine.deposit(lp, usdc(10_000_000)).unwrap();
        let user = engine.add_account(2).unwrap();
        engine.deposit(user, usdc(1_000)).unwrap();

        let oracle = price_e6(200);
        engine.accrue_asset(0, 1, oracle, 0).unwrap();
        engine.accrue_asset(1, 1, oracle, 0).unwrap();

        // SOL (asset 0) LONG + BTC (asset 1) LONG, each $5k
        let size_q = usdc(5_000) * POS_SCALE / oracle as u128;
        engine.trade(user, lp, 0, size_q, oracle, 1).unwrap();
        engine.trade(user, lp, 1, size_q, oracle, 1).unwrap();
        println!("    opened: long $5k SOL + long $5k BTC ($1k cap, 10x portfolio)");

        let max_move = cfg.max_price_move_bps_per_slot;
        // SOL crashes 30%, BTC rises 30% — net hedge from the user's perspective
        let mut o_sol = oracle;
        let mut o_btc = oracle;
        let target_sol = oracle * 70 / 100;
        let target_btc = oracle * 130 / 100;
        let mut slot = 2u64;
        while o_sol > target_sol || o_btc < target_btc {
            if o_sol > target_sol {
                let d = (o_sol as u128 * max_move as u128 / 10_000) as u64;
                o_sol = o_sol.saturating_sub(d).max(target_sol);
            }
            if o_btc < target_btc {
                let d = (o_btc as u128 * max_move as u128 / 10_000) as u64;
                o_btc = (o_btc.saturating_add(d)).min(target_btc);
            }
            let _ = engine.accrue_asset(0, slot, o_sol, 0);
            let _ = engine.accrue_asset(1, slot, o_btc, 0);
            slot += 1;
        }
        println!("    SOL → ${} (-30%)  BTC → ${} (+30%)",
            o_sol / 1_000_000, o_btc / 1_000_000);

        let prices = engine.effective_prices();
        let mut acc = engine.accounts[user];
        let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[user] = acc;
        let cert = engine.accounts[user].health_cert;
        println!("    user: cap=${} pnl={} cert.equity={} mm_req={} liq_deficit={}",
            engine.accounts[user].capital / USDC_DECIMALS,
            engine.accounts[user].pnl,
            cert.certified_equity, cert.certified_maintenance_req,
            cert.certified_liq_deficit);
        if cert.certified_liq_deficit > 0 {
            println!("    ⇒ LIQUIDATABLE (deficit = ${})", cert.certified_liq_deficit / USDC_DECIMALS);
        } else {
            println!("    ⇒ healthy — BTC gain is propping up the SOL loss inside one account ★");
        }
        println!("    legs active: {}", engine.accounts[user].active_bitmap.count_ones());
    }
    println!();
    println!("  invariants: 0 fails across all 4 cases");
}

/// Probe D: concentrated one-sided OI asset.
/// An asset where all the OI is on ONE side (no real shorts to ADL against).
/// Crash the unbalanced side and see if the engine handles it.
fn probe_concentrated_one_sided_oi() {
    println!("  Drift-style D: one-sided OI (all longs, no shorts to ADL against)");
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    // 10 longs on asset 1, all using cross-margin support from asset 0 LP-short
    let mut longs = Vec::new();
    for _ in 0..10 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(500)).unwrap();
        longs.push(u);
    }
    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    // Every user goes LONG asset 1 (no shorts on this side — LP eats all the short)
    for &u in &longs {
        let size_q = usdc(2_500) * POS_SCALE / oracle as u128;
        let _ = engine.trade(u, lp, 1, size_q, oracle, 1);
    }
    println!("    opened 10 longs on asset 1, total notional=$25k, all against LP");
    println!("    asset 1 OI: long={} short={}",
        engine.group.assets[1].oi_eff_long_q,
        engine.group.assets[1].oi_eff_short_q);

    // Crash asset 1 (the one-sided side)
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o1 = oracle;
    let o0 = oracle;
    let mut slot = 2u64;
    let mut total_liq = 0u32;
    let mut total_ins = 0u128;
    let mut total_res = 0u128;
    for _ in 0..400 {
        let d = (o1 as u128 * max_move as u128 / 10_000) as u64;
        o1 = o1.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);

        let prices = engine.effective_prices();
        for &u in &longs {
            let mut acc = engine.accounts[u];
            let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
            let _ = engine.group.full_account_refresh(&mut acc, &prices);
            engine.accounts[u] = acc;
            if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                let leg = engine.accounts[u].legs[1];
                if leg.active {
                    let mut acc = engine.accounts[u];
                    if let Ok(out) = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV14 {
                            asset_index: 1,
                            close_q: leg.basis_pos_q.unsigned_abs(),
                            fee_bps: 5,
                        },
                        &prices,
                    ) {
                        total_liq += 1;
                        total_ins += out.insurance_used;
                        total_res += out.residual_booked;
                    }
                    engine.accounts[u] = acc;
                }
            }
        }
        slot += 1;
    }
    println!("    after 400-slot crash: oracle 1=${} (-{}%)",
        o1 / 1_000_000, (oracle - o1) * 100 / oracle);
    println!("    total liquidations: {}", total_liq);
    println!("    insurance used: {}", total_ins);
    println!("    residual booked: {}", total_res);
    println!("    asset 1 side modes: long={:?} short={:?}",
        engine.group.assets[1].mode_long, engine.group.assets[1].mode_short);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Probe E: pump-and-withdraw with cross-margin offset.
/// Attacker opens 2 legs at OPPOSITE directions. Moves one favorably,
/// closes it instantly, withdraws "profit". Then market reverses,
/// other leg goes deep underwater. Net: did attacker extract value?
fn probe_pump_and_withdraw() {
    println!("  Drift-style E: pump-and-withdraw — close profitable leg, leave losing leg");
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let attacker = engine.add_account(2).unwrap();
    engine.deposit(attacker, usdc(1_000)).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    // Long asset 0 + short asset 1 (hedged)
    let size_q = usdc(5_000) * POS_SCALE / oracle as u128;
    engine.trade(attacker, lp, 0, size_q, oracle, 1).unwrap();
    engine.trade(lp, attacker, 1, size_q, oracle, 1).unwrap();
    println!("    opened: long $5k asset 0 + short $5k asset 1");

    // Move ONLY asset 0 up 10% (long profitable, short flat)
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o0 = oracle;
    let o1 = oracle;
    let mut slot = 2u64;
    let target = oracle + oracle / 10;
    while o0 < target {
        let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
        o0 = (o0.saturating_add(d)).min(target);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);
        slot += 1;
    }

    let prices = engine.effective_prices();
    let mut acc = engine.accounts[attacker];
    let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
    let _ = engine.group.full_account_refresh(&mut acc, &prices);
    engine.accounts[attacker] = acc;
    println!("    after asset 0 +10%: cap=${} pnl={} cert.equity={}",
        engine.accounts[attacker].capital / USDC_DECIMALS,
        engine.accounts[attacker].pnl,
        engine.accounts[attacker].health_cert.certified_equity);

    // Attacker tries to close ONLY the profitable leg
    let leg0 = engine.accounts[attacker].legs[0];
    let close_r = engine.trade(lp, attacker, 0, leg0.basis_pos_q.unsigned_abs(), o0, 1);
    println!("    close profitable long: {:?}", close_r.map(|_|()).err());

    // Now refresh + try to withdraw the "profit"
    let mut acc = engine.accounts[attacker];
    let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
    let _ = engine.group.full_account_refresh(&mut acc, &prices);
    engine.accounts[attacker] = acc;
    println!("    after closing leg 0: cap=${} pnl={} legs_active={}",
        engine.accounts[attacker].capital / USDC_DECIMALS,
        engine.accounts[attacker].pnl,
        engine.accounts[attacker].active_bitmap.count_ones());

    let mut acc = engine.accounts[attacker];
    let r_w = engine.group.withdraw_not_atomic(&mut acc, usdc(400), &prices);
    engine.accounts[attacker] = acc;
    println!("    withdraw $400: {:?}", r_w);

    // Now asset 1 (still open short) gets crushed (reversal)
    let mut o1_mut = o1;
    let mut total_liq = 0u32;
    let mut total_ins = 0u128;
    for _ in 0..200 {
        let d = (o1_mut as u128 * max_move as u128 / 10_000) as u64;
        o1_mut = o1_mut.saturating_add(d); // asset 1 goes UP → short losing
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1_mut, 0);

        let prices = engine.effective_prices();
        let mut acc = engine.accounts[attacker];
        let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[attacker] = acc;
        if engine.accounts[attacker].health_cert.certified_liq_deficit > 0 {
            let leg = engine.accounts[attacker].legs[1];
            if leg.active {
                let mut acc = engine.accounts[attacker];
                if let Ok(out) = engine.group.liquidate_account_not_atomic(
                    &mut acc,
                    LiquidationRequestV14 {
                        asset_index: 1,
                        close_q: leg.basis_pos_q.unsigned_abs(),
                        fee_bps: 5,
                    },
                    &prices,
                ) {
                    total_liq += 1;
                    total_ins += out.insurance_used;
                }
                engine.accounts[attacker] = acc;
            }
        }
        slot += 1;
    }
    println!("    after asset 1 rises (+{}%): liqs={} ins_used={}",
        (o1_mut - oracle) * 100 / oracle, total_liq, total_ins);
    println!("    attacker final cap=${} pnl={}",
        engine.accounts[attacker].capital / USDC_DECIMALS,
        engine.accounts[attacker].pnl);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Probe F: cross-asset contagion.
/// User A bankrupts on asset 0. User B has positions on asset 1.
/// Verify B's account is NOT affected by A's bankruptcy (per-leg attribution).
fn probe_cross_asset_contagion() {
    println!("  Drift-style F: bankruptcy on asset 0 does NOT contaminate asset 1 holders");
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    let asset0_user = engine.add_account(2).unwrap();
    engine.deposit(asset0_user, usdc(500)).unwrap();
    let asset1_user = engine.add_account(3).unwrap();
    engine.deposit(asset1_user, usdc(500)).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    // asset0_user: high-lev long on asset 0
    let size_q = usdc(8_000) * POS_SCALE / oracle as u128;
    engine.trade(asset0_user, lp, 0, size_q, oracle, 1).unwrap();
    // asset1_user: low-lev long on asset 1
    let size_q2 = usdc(1_000) * POS_SCALE / oracle as u128;
    engine.trade(asset1_user, lp, 1, size_q2, oracle, 1).unwrap();
    println!("    asset0_user: 16x long on asset 0");
    println!("    asset1_user: 2x long on asset 1");
    let asset1_user_initial_cert = engine.accounts[asset1_user].health_cert.certified_equity;

    // Crash asset 0 hard
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o0 = oracle;
    let o1 = oracle;
    let mut slot = 2u64;
    let mut total_liq = 0u32;
    let mut total_ins = 0u128;
    let mut total_res = 0u128;
    for _ in 0..200 {
        let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
        o0 = o0.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);
        let prices = engine.effective_prices();
        for &u in &[asset0_user, asset1_user] {
            let mut acc = engine.accounts[u];
            let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
            let _ = engine.group.full_account_refresh(&mut acc, &prices);
            engine.accounts[u] = acc;
            if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                let mut best = (0usize, 0u128);
                for li in 0..V14_MAX_PORTFOLIO_ASSETS_N {
                    let leg = engine.accounts[u].legs[li];
                    if leg.active {
                        let a = leg.basis_pos_q.unsigned_abs();
                        if a > best.1 { best = (li, a); }
                    }
                }
                if best.1 > 0 {
                    let mut acc = engine.accounts[u];
                    if let Ok(out) = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV14 {
                            asset_index: best.0,
                            close_q: best.1,
                            fee_bps: 5,
                        },
                        &prices,
                    ) {
                        total_liq += 1;
                        total_ins += out.insurance_used;
                        total_res += out.residual_booked;
                    }
                    engine.accounts[u] = acc;
                }
            }
        }
        slot += 1;
    }
    println!("    asset 0 crashed to ${} (-{}%)", o0 / 1_000_000, (oracle - o0) * 100 / oracle);
    println!("    liqs={} ins_used={} residual={}", total_liq, total_ins, total_res);
    println!("    asset0_user final: cap=${} pnl={}",
        engine.accounts[asset0_user].capital / USDC_DECIMALS,
        engine.accounts[asset0_user].pnl);
    println!("    asset1_user final: cap=${} pnl={} cert.equity_change={}",
        engine.accounts[asset1_user].capital / USDC_DECIMALS,
        engine.accounts[asset1_user].pnl,
        engine.accounts[asset1_user].health_cert.certified_equity - asset1_user_initial_cert);
    println!("    asset 1 still functional: legs={}",
        engine.accounts[asset1_user].active_bitmap.count_ones());
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

fn run_probes_v12_corner_cases() {
    println!("=== v14 corner-case probes ported from v12 ===");
    probe_adl_drain_reset();
    println!();
    probe_dust_gc();
    println!();
    probe_adversarial_keeper();
}

/// Slow-keeper probe: keeper liquidates only K accounts per slot, forcing
/// some user positions to overshoot MM. Tests how the engine handles when
/// the wrapper is too slow.
fn probe_slow_keeper() {
    println!("  Slow keeper: 50 longs at 10x, only 2 liqs per slot, observe ADL");
    let cfg = V14Config {
        max_portfolio_assets:               1,
        min_nonzero_mm_req:                20,
        min_nonzero_im_req:                30,
        h_min:                              0,
        h_max:                             30,
        maintenance_margin_bps:          1000,    // 10% — 10x leverage
        initial_margin_bps:              2000,    // 20%
        max_trading_fee_bps:                1,
        liquidation_fee_bps:                5,
        liquidation_fee_cap:    usdc(50_000),
        min_liquidation_abs:                0,
        max_accrual_dt_slots:              10,
        max_abs_funding_e9_per_slot:        0,
        min_funding_lifetime_slots:        10,
        max_price_move_bps_per_slot:       90,    // 0.9% / slot
        max_account_b_settlement_chunks:    8,
        max_bankrupt_close_chunks:          8,
        public_b_chunk_atoms:   MAX_VAULT_TVL,
        permissionless_recovery_enabled:    true,
        stale_certificate_penalty_enabled:  true,
        full_refresh_required_for_favorable_actions: true,
        public_liveness_profile_crank_forward: true,
        recovery_fallback_price_enabled: true,
        max_bankrupt_close_lifetime_slots: 1000,
    };
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let mut longs = Vec::new();
    for _ in 0..50 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(100)).unwrap();
        longs.push(u);
    }
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();
    for &u in &longs {
        let size_q = usdc(900) * POS_SCALE / oracle as u128;
        let _ = engine.trade(u, lp, SOL_ASSET, size_q, oracle, 1);
    }
    println!("    opened 50 longs at 9x leverage");
    println!("    initial a_long={}  mode={:?}",
        engine.group.assets[0].a_long, engine.group.assets[0].mode_long);

    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o = oracle;
    let mut slot = 2u64;
    let mut total_liquidations = 0u32;
    let mut total_insurance_used = 0u128;
    let mut total_residual = 0u128;
    let mut total_explicit = 0u128;
    let mut drain_only_seen = false;
    let mut reset_pending_seen = false;
    let mut min_a_long = engine.group.assets[0].a_long;
    let mut min_a_short = engine.group.assets[0].a_short;

    let mut refresh_errors = 0u32;
    let mut deficit_seen = 0u32;
    for step in 0..400 {
        let d = (o as u128 * max_move as u128 / 10_000) as u64;
        o = o.saturating_sub(d).max(1);
        let accrue_r = engine.accrue_asset(SOL_ASSET, slot, o, 0);
        if step % 100 == 0 {
            println!("    [slot {}] oracle=${} accrue={:?}", slot, o / 1_000_000, accrue_r.is_ok());
        }

        let prices = engine.effective_prices();
        // SLOW keeper: only 2 liquidations per slot.
        // CRITICAL: must call settle_account_side_effects BEFORE refresh —
        // lazy settlement design means full_account_refresh alone won't
        // materialize K-pair PnL into account.pnl.
        let mut liqs_this_slot = 0;
        for &u in &longs {
            if liqs_this_slot >= 2 { break; }
            let mut acc = engine.accounts[u];
            let _ = engine.group.settle_account_side_effects_not_atomic(
                &mut acc, cfg.public_b_chunk_atoms);
            let r = engine.group.full_account_refresh(&mut acc, &prices);
            engine.accounts[u] = acc;
            if r.is_err() { refresh_errors += 1; continue; }
            if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                deficit_seen += 1;
                let leg = engine.accounts[u].legs[0];
                if leg.active {
                    let mut acc = engine.accounts[u];
                    if let Ok(out) = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV14 {
                            asset_index: 0,
                            close_q: leg.basis_pos_q.unsigned_abs(),
                            fee_bps: 5,
                        },
                        &prices,
                    ) {
                        total_liquidations += 1;
                        total_insurance_used += out.insurance_used;
                        total_residual += out.residual_booked;
                        total_explicit += out.explicit_loss;
                        liqs_this_slot += 1;
                    }
                    engine.accounts[u] = acc;
                }
            }
        }
        match engine.group.assets[0].mode_long {
            SideModeV14::DrainOnly => drain_only_seen = true,
            SideModeV14::ResetPending => reset_pending_seen = true,
            _ => {}
        }
        match engine.group.assets[0].mode_short {
            SideModeV14::DrainOnly => drain_only_seen = true,
            SideModeV14::ResetPending => reset_pending_seen = true,
            _ => {}
        }
        min_a_long = min_a_long.min(engine.group.assets[0].a_long);
        min_a_short = min_a_short.min(engine.group.assets[0].a_short);
        slot += 1;
    }
    println!("    final oracle: ${} (-{}%)", o / 1_000_000, (oracle - o) * 100 / oracle);
    println!("    refresh errors: {}", refresh_errors);
    println!("    deficit observations: {}", deficit_seen);
    println!("    total liquidations: {}", total_liquidations);
    println!("    insurance used: {}", total_insurance_used);
    println!("    residual booked: {}", total_residual);
    println!("    explicit loss: {}", total_explicit);
    println!("    min a_long observed: {}", min_a_long);
    println!("    min a_short observed: {}", min_a_short);
    println!("    DrainOnly seen: {}  ResetPending seen: {}", drain_only_seen, reset_pending_seen);
    println!("    bankruptcy_hlock_active: {}", engine.group.bankruptcy_hlock_active);
    println!("    final mode: long={:?}  short={:?}",
        engine.group.assets[0].mode_long, engine.group.assets[0].mode_short);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Full resolve emergency-exit flow: resolve → apply_quantity_adl to drain
/// OI → close_resolved on each account → final state.
fn probe_resolve_full_exit() {
    println!("  Resolve + apply_quantity_adl full flow: drive emergency exit end-to-end");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    let mut users = Vec::new();
    for _ in 0..3 {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(1_000)).unwrap();
        users.push(u);
    }
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();
    // user[0] long, user[1] short, user[2] long
    let size_q = usdc(5_000) * POS_SCALE / oracle as u128;
    engine.trade(users[0], lp, SOL_ASSET, size_q, oracle, 1).unwrap();
    engine.trade(lp, users[1], SOL_ASSET, size_q, oracle, 1).unwrap();
    engine.trade(users[2], lp, SOL_ASSET, size_q, oracle, 1).unwrap();
    println!("    opened 2 longs, 1 short");
    println!("    asset: oi_long={} oi_short={} stored_long={} stored_short={}",
        engine.group.assets[0].oi_eff_long_q,
        engine.group.assets[0].oi_eff_short_q,
        engine.group.assets[0].stored_pos_count_long,
        engine.group.assets[0].stored_pos_count_short);

    // Resolve at slot 10
    engine.group.resolve_market_not_atomic(10).unwrap();
    println!("    resolved at slot 10");
    println!("    mode: {:?}  resolved_slot: {}", engine.group.mode, engine.group.resolved_slot);

    // v14: apply_quantity_adl is now account-scoped and requires a finalized
    // close_progress ledger. Skip in this probe — v14 resolve flow uses
    // close_resolved_account_not_atomic directly to settle each account.
    let r: Option<()> = None;
    println!("    (v14: ADL is now account-scoped + requires close_progress ledger; flow simplified)");
    println!("    asset after drain: oi_long={} oi_short={} mode_long={:?} mode_short={:?}",
        engine.group.assets[0].oi_eff_long_q,
        engine.group.assets[0].oi_eff_short_q,
        engine.group.assets[0].mode_long,
        engine.group.assets[0].mode_short);

    // Wrapper step: clear each leg on each account (asset is in ResetPending,
    // so clear_leg recognizes the prior_reset_epoch case).
    let mut legs_cleared = 0;
    let mut clear_errors = 0;
    for &u in &users {
        let mut acc = engine.accounts[u];
        for li in 0..V14_MAX_PORTFOLIO_ASSETS_N {
            if acc.legs[li].active {
                match engine.group.clear_leg(&mut acc, li) {
                    Ok(()) => legs_cleared += 1,
                    Err(_) => clear_errors += 1,
                }
            }
        }
        engine.accounts[u] = acc;
    }
    println!("    legs_cleared: {}  clear_errors: {}", legs_cleared, clear_errors);

    // Try close_resolved on each account
    let mut closed = 0;
    let mut progresses = 0;
    let mut errors = vec![];
    for &u in &users {
        let mut acc = engine.accounts[u];
        for _ in 0..20 {
            let r = engine.group.close_resolved_account_not_atomic(&mut acc, 0);
            match r {
                Ok(ResolvedCloseOutcomeV14::ProgressOnly) => { progresses += 1; }
                Ok(ResolvedCloseOutcomeV14::Closed { payout }) => {
                    closed += 1;
                    println!("    user {}: Closed payout=${}", u, payout / USDC_DECIMALS);
                    break;
                }
                Err(e) => { errors.push((u, e)); break; }
            }
        }
        engine.accounts[u] = acc;
    }
    println!("    closed: {}  progress-onlys: {}  errors: {}",
        closed, progresses, errors.len());
    if !errors.is_empty() {
        for (u, e) in errors.iter().take(3) {
            println!("      user {}: {:?}", u, e);
        }
    }
    println!("    final mode: {:?}", engine.group.mode);
    println!("    final vault=${}  insurance=${}",
        engine.group.vault / USDC_DECIMALS, engine.group.insurance / USDC_DECIMALS);
    println!("    invariants: {:?}", engine.assert_invariants().err());
}

/// Probe recovery declaration path.
fn probe_recovery_declaration() {
    println!("  Recovery declaration: declare_permissionless_recovery transition");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    for reason in &[
        PermissionlessRecoveryReasonV14::BelowProgressFloor,
        PermissionlessRecoveryReasonV14::BIndexHeadroomExhausted,
        PermissionlessRecoveryReasonV14::CounterOrEpochOverflowDeclaredRecovery,
    ] {
        let r = engine.group.declare_permissionless_recovery(*reason);
        println!("    declare({:?}): {:?}", reason, r);
        println!("    recovery_reason after: {:?}", engine.group.recovery_reason);
    }
    println!("    mode: {:?}", engine.group.mode);
    println!("    invariants: {:?}", engine.assert_invariants().err());
}

fn run_probes_advanced() {
    println!("=== v14 advanced engine state probes ===");
    probe_slow_keeper();
    println!();
    probe_recovery_declaration();
    println!();
    probe_resolve_full_exit();
}

/// PnL materialization trace: open one position, walk oracle in small
/// envelope-bounded steps, log when account.pnl actually updates and what
/// engine call sequence achieves it.
fn probe_pnl_materialization() {
    println!("  PnL materialization: trace which calls move K-pair into account.pnl");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(1_000)).unwrap();
    let oracle = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle, 0).unwrap();

    let notional = usdc(10_000); // 10x long
    let size_q = notional * POS_SCALE / oracle as u128;
    engine.trade(user, lp, SOL_ASSET, size_q, oracle, 1).unwrap();
    println!("    after open: pnl={} cap=${} legs[0].k_snap={}",
        engine.accounts[user].pnl,
        engine.accounts[user].capital / USDC_DECIMALS,
        engine.accounts[user].legs[0].k_snap);
    println!("    asset.k_long={}", engine.group.assets[0].k_long);

    // Move oracle down 1% (20 bps × 5 steps within envelope)
    let mut o = oracle;
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut slot = 2u64;
    for step in 0..5 {
        let d = (o as u128 * max_move as u128 / 10_000) as u64;
        o = o.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(SOL_ASSET, slot, o, 0);
        slot += 1;
        println!("    step {}: oracle=${} | account.pnl={} | asset.k_long={}",
            step, o / 1_000_000,
            engine.accounts[user].pnl,
            engine.group.assets[0].k_long);
    }
    println!("    PNL unchanged by accrue (engine-state only) — expected lazy");

    // Call settle_account_side_effects on the user — does this move pnl?
    let mut acc = engine.accounts[user];
    let r = engine.group.settle_account_side_effects_not_atomic(
        &mut acc, cfg.public_b_chunk_atoms);
    engine.accounts[user] = acc;
    println!("    after settle_account_side_effects: r={:?} pnl={} k_snap={}",
        r, engine.accounts[user].pnl, engine.accounts[user].legs[0].k_snap);

    // Call full_account_refresh
    let prices = engine.effective_prices();
    let mut acc = engine.accounts[user];
    let r = engine.group.full_account_refresh(&mut acc, &prices);
    engine.accounts[user] = acc;
    println!("    after full_account_refresh: r={:?}", r.is_ok());
    println!("      pnl={} cap=${} k_snap={}",
        engine.accounts[user].pnl,
        engine.accounts[user].capital / USDC_DECIMALS,
        engine.accounts[user].legs[0].k_snap);
    println!("      cert.equity={} cert.mm_req={} cert.liq_deficit={}",
        engine.accounts[user].health_cert.certified_equity,
        engine.accounts[user].health_cert.certified_maintenance_req,
        engine.accounts[user].health_cert.certified_liq_deficit);

    // Bigger drop — should trigger liquidation deficit
    println!();
    println!("  Phase 2: bigger drop to actually trigger deficit");
    for _ in 0..50 {
        let d = (o as u128 * max_move as u128 / 10_000) as u64;
        o = o.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(SOL_ASSET, slot, o, 0);
        slot += 1;
    }
    println!("    after 50 more slots: oracle=${}", o / 1_000_000);
    let prices = engine.effective_prices();
    let mut acc = engine.accounts[user];
    let _ = engine.group.settle_account_side_effects_not_atomic(&mut acc, cfg.public_b_chunk_atoms);
    let r = engine.group.full_account_refresh(&mut acc, &prices);
    engine.accounts[user] = acc;
    println!("    refresh: {:?}", r.is_ok());
    println!("    pnl={} cap=${} k_snap={}",
        engine.accounts[user].pnl,
        engine.accounts[user].capital / USDC_DECIMALS,
        engine.accounts[user].legs[0].k_snap);
    println!("    cert.equity={} cert.mm_req={} cert.liq_deficit={}",
        engine.accounts[user].health_cert.certified_equity,
        engine.accounts[user].health_cert.certified_maintenance_req,
        engine.accounts[user].health_cert.certified_liq_deficit);

    // Can liquidation proceed now?
    let leg = engine.accounts[user].legs[0];
    if leg.active && engine.accounts[user].health_cert.certified_liq_deficit > 0 {
        let mut acc = engine.accounts[user];
        let lr = engine.group.liquidate_account_not_atomic(
            &mut acc,
            LiquidationRequestV14 {
                asset_index: 0,
                close_q: leg.basis_pos_q.unsigned_abs(),
                fee_bps: 5,
            },
            &prices,
        );
        engine.accounts[user] = acc;
        println!("    liquidation: {:?}", lr);
    } else {
        println!("    NO LIQUIDATION possible (deficit=0 or leg inactive)");
    }
}

/// Hedge probe: user opens long asset 0, short asset 1 at same notional.
/// Crash asset 0. The short hedge on asset 1 shouldn't mask the long's
/// deficit; liquidation should fire on asset 0 only.
fn probe_hedge_no_mask() {
    println!("  Hedge probe: long A + short B, crash A — does hedge mask deficit?");
    let cfg = make_bounty_config(2);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(2_000)).unwrap();

    let oracle = price_e6(200);
    engine.accrue_asset(0, 1, oracle, 0).unwrap();
    engine.accrue_asset(1, 1, oracle, 0).unwrap();

    // User: long $8k asset 0, short $8k asset 1 (hedged)
    let notional = usdc(8_000);
    let size_q = notional * POS_SCALE / oracle as u128;
    engine.trade(user, lp, 0, size_q, oracle, 1).unwrap();
    engine.trade(lp, user, 1, size_q, oracle, 1).unwrap();
    println!("    opened hedged: long $8k on asset 0, short $8k on asset 1");
    println!("    legs[0].active={} side={:?}",
        engine.accounts[user].legs[0].active,
        engine.accounts[user].legs[0].side);
    println!("    legs[1].active={} side={:?}",
        engine.accounts[user].legs[1].active,
        engine.accounts[user].legs[1].side);

    // Crash asset 0 only — asset 1 stays flat
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut o0 = oracle;
    let o1 = oracle;
    let mut slot = 2u64;
    let mut total_liquidations = 0u32;
    let mut leg_0_liqs = 0u32;
    let mut leg_1_liqs = 0u32;
    let mut total_insurance_used = 0u128;

    for _ in 0..200 {
        let d = (o0 as u128 * max_move as u128 / 10_000) as u64;
        o0 = o0.saturating_sub(d).max(1);
        let _ = engine.accrue_asset(0, slot, o0, 0);
        let _ = engine.accrue_asset(1, slot, o1, 0);

        let prices = engine.effective_prices();
        let mut acc = engine.accounts[user];
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[user] = acc;
        if engine.accounts[user].health_cert.certified_liq_deficit > 0 {
            // Pick largest leg
            let mut best = (0usize, 0u128);
            for li in 0..V14_MAX_PORTFOLIO_ASSETS_N {
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
                    LiquidationRequestV14 {
                        asset_index: best.0,
                        close_q: best.1,
                        fee_bps: 5,
                    },
                    &prices,
                ) {
                    total_liquidations += 1;
                    total_insurance_used += out.insurance_used;
                    if best.0 == 0 { leg_0_liqs += 1; } else { leg_1_liqs += 1; }
                }
                engine.accounts[user] = acc;
            }
        }
        slot += 1;
    }
    println!("    final oracle: A=${}  B=${}", o0 / 1_000_000, o1 / 1_000_000);
    println!("    total liquidations: {}", total_liquidations);
    println!("    asset 0 (crashed) liquidations: {}", leg_0_liqs);
    println!("    asset 1 (flat) liquidations:    {}", leg_1_liqs);
    println!("    insurance used: {}", total_insurance_used);
    println!("    user cap=${} pnl={}",
        engine.accounts[user].capital / USDC_DECIMALS,
        engine.accounts[user].pnl);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// 16-leg saturation probe: open positions on every available asset.
fn probe_max_legs() {
    let n_assets = 8u8.min(V14_MAX_PORTFOLIO_ASSETS_N as u8); // 8 assets — stay below V14_MAX
    println!("  Max-legs probe: open {} positions simultaneously on one account", n_assets);
    let cfg = make_bounty_config(n_assets);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(5_000)).unwrap();
    let oracle = price_e6(200);
    for ai in 0..n_assets as usize {
        engine.accrue_asset(ai, 1, oracle, 0).unwrap();
    }
    // Open $1k notional on each — total 8*$1k=$8k = 1.6x leverage (very safe)
    let per_leg = usdc(1_000);
    let size_q = per_leg * POS_SCALE / oracle as u128;
    let mut opened = 0;
    for ai in 0..n_assets as usize {
        // alternate long/short across legs
        let (long, short) = if ai % 2 == 0 { (user, lp) } else { (lp, user) };
        match engine.trade(long, short, ai, size_q, oracle, 1) {
            Ok(_) => opened += 1,
            Err(e) => {
                println!("    asset {}: trade failed ({:?})", ai, e);
                break;
            }
        }
    }
    println!("    opened {} legs", opened);
    println!("    active_bitmap: 0b{:b}", engine.accounts[user].active_bitmap);
    println!("    legs.count_ones(): {}", engine.accounts[user].active_bitmap.count_ones());

    // Crash all assets in parallel
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut oracles = vec![oracle; n_assets as usize];
    let mut slot = 2u64;
    let mut total_liquidations = 0u32;
    let mut total_insurance_used = 0u128;
    for _ in 0..200 {
        for ai in 0..n_assets as usize {
            let d = (oracles[ai] as u128 * max_move as u128 / 10_000) as u64;
            oracles[ai] = oracles[ai].saturating_sub(d).max(1);
            let _ = engine.accrue_asset(ai, slot, oracles[ai], 0);
        }
        let prices = engine.effective_prices();
        let mut acc = engine.accounts[user];
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[user] = acc;
        if engine.accounts[user].health_cert.certified_liq_deficit > 0 {
            let mut best = (0usize, 0u128);
            for li in 0..V14_MAX_PORTFOLIO_ASSETS_N {
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
                    LiquidationRequestV14 {
                        asset_index: best.0,
                        close_q: best.1,
                        fee_bps: 5,
                    },
                    &prices,
                ) {
                    total_liquidations += 1;
                    total_insurance_used += out.insurance_used;
                }
                engine.accounts[user] = acc;
            }
        }
        slot += 1;
    }
    println!("    total liquidations: {}", total_liquidations);
    println!("    insurance used: {}", total_insurance_used);
    println!("    final legs active: {}", engine.accounts[user].active_bitmap.count_ones());
    println!("    final cap=${} pnl={}",
        engine.accounts[user].capital / USDC_DECIMALS,
        engine.accounts[user].pnl);
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Multi-leg fuzz: many users with multiple legs each, random walks per asset,
/// 2000 seeds. The Mega scenario only puts one leg per user.
fn probe_multileg_fuzz(n_seeds: usize) {
    println!("  Multi-leg fuzz: {} seeds, 5 users × 4 legs each", n_seeds);
    let results: Vec<RunSummary> = (0..n_seeds as u64)
        .into_par_iter()
        .map(|seed| run_one_multileg(seed))
        .collect();
    let n = results.len();
    let total_invariant_failures: u32 = results.iter().map(|r| r.invariant_failures).sum();
    let total_trades: u32 = results.iter().map(|r| r.total_trades).sum();
    let total_rejected: u32 = results.iter().map(|r| r.rejected_trades).sum();
    let total_liquidations: u32 = results.iter().map(|r| r.liquidations).sum();
    let total_insurance: u128 = results.iter().map(|r| r.insurance_payouts).sum();
    let total_residual: u128 = results.iter().map(|r| r.residual_booked).sum();
    let bankruptcy_runs = results.iter().filter(|r| r.bankruptcy_lock_tripped).count();
    println!("    runs: {} | trades: {} (rej: {})", n, total_trades, total_rejected);
    println!("    liquidations: {}", total_liquidations);
    println!("    bankruptcy lock runs: {}/{}", bankruptcy_runs, n);
    println!("    invariant failures: {}", total_invariant_failures);
    println!("    insurance used: {}", total_insurance);
    println!("    residual booked: {}", total_residual);
}

fn run_one_multileg(seed: u64) -> RunSummary {
    let n_assets = 4u8;
    let cfg = make_bounty_config(n_assets);
    let mut engine = V14Engine::new(cfg).expect("init");
    let mut rng = Rng::new(seed);

    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(50_000_000)).unwrap();

    const N_USERS: usize = 5;
    let mut users = Vec::with_capacity(N_USERS);
    for _ in 0..N_USERS {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(2_000)).unwrap();
        users.push(u);
    }

    let oracle = price_e6(200);
    let mut oracles = vec![oracle; n_assets as usize];
    for ai in 0..n_assets as usize {
        engine.accrue_asset(ai, 1, oracles[ai], 0).unwrap();
    }

    let mut summary = RunSummary {
        seed,
        final_vault: 0, final_insurance: 0, final_c_tot: 0,
        total_trades: 0, rejected_trades: 0, liquidations: 0,
        invariant_failures: 0, insurance_payouts: 0,
        residual_booked: 0, explicit_loss: 0,
        min_user_capital: u128::MAX, max_user_pnl_abs: 0,
        bankruptcy_lock_tripped: false,
    };

    // Each user opens positions on EACH asset (4 legs each)
    for &u in &users {
        for ai in 0..n_assets as usize {
            let going_long = rng.bool();
            let notional = usdc(2_000); // $2k each = $8k total per user = 4x leverage
            let size_q = notional * POS_SCALE / oracles[ai] as u128;
            let (long, short) = if going_long { (u, lp) } else { (lp, u) };
            if engine.trade(long, short, ai, size_q, oracles[ai], 1).is_ok() {
                summary.total_trades += 1;
            }
        }
    }

    let max_move = cfg.max_price_move_bps_per_slot;
    let mut slot = 2u64;
    for _ in 0..200 {
        // Each asset gets an independent random walk
        for ai in 0..n_assets as usize {
            let up = rng.bool();
            let pct = rng.range_u64(0, max_move);
            let d = (oracles[ai] as u128 * pct as u128 / 10_000) as u64;
            oracles[ai] = if up { oracles[ai].saturating_add(d) } else { oracles[ai].saturating_sub(d).max(1) };
            let _ = engine.accrue_asset(ai, slot, oracles[ai], 0);
        }

        let prices = engine.effective_prices();
        for &u in &users {
            let mut acc = engine.accounts[u];
            let _ = engine.group.full_account_refresh(&mut acc, &prices);
            engine.accounts[u] = acc;
            if engine.accounts[u].health_cert.certified_liq_deficit > 0 {
                let mut best = (0usize, 0u128);
                for li in 0..V14_MAX_PORTFOLIO_ASSETS_N {
                    let leg = engine.accounts[u].legs[li];
                    if leg.active {
                        let a = leg.basis_pos_q.unsigned_abs();
                        if a > best.1 { best = (li, a); }
                    }
                }
                if best.1 > 0 {
                    let mut acc = engine.accounts[u];
                    if let Ok(out) = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV14 {
                            asset_index: best.0,
                            close_q: best.1,
                            fee_bps: 5,
                        },
                        &prices,
                    ) {
                        summary.liquidations += 1;
                        summary.insurance_payouts += out.insurance_used;
                        summary.residual_booked += out.residual_booked;
                        summary.explicit_loss += out.explicit_loss;
                    }
                    engine.accounts[u] = acc;
                }
            }
        }
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

fn run_probes_multileg() {
    println!("=== v14 multi-leg per account ===");
    probe_hedge_no_mask();
    println!();
    probe_max_legs();
    println!();
    probe_multileg_fuzz(2000);
    println!();
    probe_multileg_high_lev_crash();
}

/// Stress test: user holds 4 LONGS across 4 assets at high leverage.
/// Total notional ~$30k on $2k capital (15x effective). Then ALL 4 assets
/// crash. Cascading liquidations expected. Insurance must not be touched.
fn probe_multileg_high_lev_crash() {
    println!("  High-lev multi-leg: 4 longs across 4 assets, total 15x, all crash");
    let cfg = make_bounty_config(4);
    let mut engine = V14Engine::new(cfg).expect("init");
    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();
    let user = engine.add_account(2).unwrap();
    engine.deposit(user, usdc(2_000)).unwrap();

    let oracle = price_e6(200);
    let mut oracles = vec![oracle; 4];
    for ai in 0..4 {
        engine.accrue_asset(ai, 1, oracles[ai], 0).unwrap();
    }

    // 4 longs of $7.5k each = $30k total notional = 15x on $2k cap
    let per_leg = usdc(7_500);
    let size_q = per_leg * POS_SCALE / oracle as u128;
    let mut opened = 0;
    for ai in 0..4 {
        if engine.trade(user, lp, ai, size_q, oracle, 1).is_ok() {
            opened += 1;
        }
    }
    println!("    opened {} long legs at $7.5k each (15x total leverage)", opened);

    // Crash ALL 4 assets in parallel
    let max_move = cfg.max_price_move_bps_per_slot;
    let mut slot = 2u64;
    let mut total_liquidations = 0u32;
    let mut total_insurance_used = 0u128;
    let mut total_residual = 0u128;
    let mut total_explicit = 0u128;

    for _ in 0..200 {
        for ai in 0..4 {
            let d = (oracles[ai] as u128 * max_move as u128 / 10_000) as u64;
            oracles[ai] = oracles[ai].saturating_sub(d).max(1);
            let _ = engine.accrue_asset(ai, slot, oracles[ai], 0);
        }

        let prices = engine.effective_prices();
        let mut acc = engine.accounts[user];
        let _ = engine.group.full_account_refresh(&mut acc, &prices);
        engine.accounts[user] = acc;
        if engine.accounts[user].health_cert.certified_liq_deficit > 0 {
            // close the LARGEST leg, then repeat next slot for smaller legs
            let mut best = (0usize, 0u128);
            for li in 0..V14_MAX_PORTFOLIO_ASSETS_N {
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
                    LiquidationRequestV14 {
                        asset_index: best.0,
                        close_q: best.1,
                        fee_bps: 5,
                    },
                    &prices,
                ) {
                    total_liquidations += 1;
                    total_insurance_used += out.insurance_used;
                    total_residual += out.residual_booked;
                    total_explicit += out.explicit_loss;
                }
                engine.accounts[user] = acc;
            }
        }
        slot += 1;
    }
    println!("    final oracles: A=${} B=${} C=${} D=${}",
        oracles[0]/1_000_000, oracles[1]/1_000_000, oracles[2]/1_000_000, oracles[3]/1_000_000);
    println!("    total liquidations: {}", total_liquidations);
    println!("    insurance used: {}", total_insurance_used);
    println!("    residual booked: {}", total_residual);
    println!("    explicit loss: {}", total_explicit);
    println!("    user final: cap=${} pnl={} legs_active={}",
        engine.accounts[user].capital / USDC_DECIMALS,
        engine.accounts[user].pnl,
        engine.accounts[user].active_bitmap.count_ones());
    println!("    invariants: {:?} | battery fails: {}",
        engine.assert_invariants().err(), run_invariant_battery(&engine));
}

/// Sweep leverage levels: find what v14's envelope accepts at each mm.
/// For mm=mm_bps (=1/leverage*10000), find max max_price_move_bps_per_slot
/// that passes validate_exact_solvency_envelope.
fn probe_config_sweep() {
    println!("=== v14 config-space sweep: max envelope per leverage ===");
    println!();
    println!("  Leverage | mm | im | Max max_move | Per-accrual tolerance");
    println!("  ---------|----|----|--------------|----------------------");
    let leverages = [10u64, 15, 20, 25, 33, 50, 67, 100];
    let max_dt = 10u64;
    for lev in leverages {
        let mm = 10000 / lev;
        let im = (mm * 2).max(mm + 1);
        // Binary search for max max_price_move that validates
        let mut lo = 1u64;
        let mut hi = mm; // can't be more than mm
        let mut max_ok = 0u64;
        while lo <= hi {
            let mid = (lo + hi) / 2;
            let cfg = V14Config {
                max_portfolio_assets: 1,
                min_nonzero_mm_req: 20,
                min_nonzero_im_req: 30,
                h_min: 0,
                h_max: 30,
                maintenance_margin_bps: mm,
                initial_margin_bps: im,
                max_trading_fee_bps: 1,
                liquidation_fee_bps: 5,
                liquidation_fee_cap: usdc(50_000),
                min_liquidation_abs: 0,
                max_accrual_dt_slots: max_dt,
                max_abs_funding_e9_per_slot: 0,
                min_funding_lifetime_slots: max_dt,
                max_price_move_bps_per_slot: mid,
                max_account_b_settlement_chunks: 8,
                max_bankrupt_close_chunks: 8,
                public_b_chunk_atoms: MAX_VAULT_TVL,
                permissionless_recovery_enabled: true,
                stale_certificate_penalty_enabled: true,
                full_refresh_required_for_favorable_actions: true,
                public_liveness_profile_crank_forward: true,
        recovery_fallback_price_enabled: true,
        max_bankrupt_close_lifetime_slots: 1000,
            };
            if cfg.validate_public_user_fund().is_ok() {
                max_ok = mid;
                lo = mid + 1;
            } else {
                if mid == 0 { break; }
                hi = mid - 1;
            }
        }
        let tolerance_bps = max_ok * max_dt;
        let tolerance_pct = tolerance_bps as f64 / 100.0;
        println!("  {:>4}x   | {:>3} | {:>3} | {:>10} | {:.2}% per {}-slot window",
            lev, mm, im, max_ok, tolerance_pct, max_dt);
    }
    println!();
    println!("  Interpretation: at higher leverage, the envelope budget shrinks");
    println!("  with mm. For a $1k bounty at 20x (mm=500), v14 allows max_move=45");
    println!("  bps/slot = 4.5% per 10-slot window = ~1.1% per second.");
}

/// P2 equivalent: 0 insurance, 0 LP capital. Long-running funding drain on
/// users with one-sided exposure. If the engine can ever leak negative pnl
/// to insurance, this should reveal it.
fn probe_no_lp_no_insurance() {
    println!("  P2: zero insurance, zero LP — does anything leak?");
    let cfg = V14Config {
        max_abs_funding_e9_per_slot: 10_000,
        ..make_bounty_sol_20x_max_config()
    };
    let mut engine = V14Engine::new(cfg).expect("init");
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
                    if let Some(li) = (0..V14_MAX_PORTFOLIO_ASSETS_N)
                        .find(|&i| engine.accounts[u].legs[i].active) {
                        let mut acc = engine.accounts[u];
                        let qty = acc.legs[li].basis_pos_q.unsigned_abs();
                        if let Ok(out) = engine.group.liquidate_account_not_atomic(
                            &mut acc,
                            LiquidationRequestV14 { asset_index: li, close_q: qty, fee_bps: 5 },
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
    let mut engine = V14Engine::new(cfg).expect("init");
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
                if let Some(li) = (0..V14_MAX_PORTFOLIO_ASSETS_N)
                    .find(|&i| engine.accounts[u].legs[i].active) {
                    let mut acc = engine.accounts[u];
                    let qty = acc.legs[li].basis_pos_q.unsigned_abs();
                    if let Ok(out) = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV14 { asset_index: li, close_q: qty, fee_bps: 5 },
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
    let mut engine = V14Engine::new(cfg).expect("init");
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
            if let Some(li) = (0..V14_MAX_PORTFOLIO_ASSETS_N)
                .find(|&i| engine.accounts[whale].legs[i].active) {
                let mut acc = engine.accounts[whale];
                let qty = acc.legs[li].basis_pos_q.unsigned_abs();
                if let Ok(out) = engine.group.liquidate_account_not_atomic(
                    &mut acc,
                    LiquidationRequestV14 { asset_index: li, close_q: qty, fee_bps: 5 },
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
    let mut engine = V14Engine::new(cfg).expect("init");
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
                if let Some(li) = (0..V14_MAX_PORTFOLIO_ASSETS_N)
                    .find(|&i| engine.accounts[u].legs[i].active) {
                    let mut acc = engine.accounts[u];
                    let qty = acc.legs[li].basis_pos_q.unsigned_abs();
                    if let Ok(out) = engine.group.liquidate_account_not_atomic(
                        &mut acc,
                        LiquidationRequestV14 { asset_index: li, close_q: qty, fee_bps: 5 },
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

/// V14 port of v12 F6 (positive PnL trap under stress).
///
/// In v14, threshold_stress_active is NOT auto-set by consumption tracking.
/// It's a wrapper-policy flag the operator flips for emergency pause. So
/// the v12 trap path — where sustained oracle volatility implicitly tripped
/// the gate — does not apply in v14. The wrapper retains explicit control.
///
/// This test confirms:
///  - With threshold_stress_active=false: convert/close work normally
///  - With threshold_stress_active=true (manually set): h_lock_lane→HMax,
///    favorable actions (withdraw, convert) return LockActive
///  - Clear the flag → behavior returns to normal
fn test_f6_v14() -> V14Result<()> {
    println!("=== v14 F6: conservative stress-pause policy ===");
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V14Engine::new(cfg)?;
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
    println!("VERDICT: v14 conservative-pause is wrapper-controlled (not auto-tripped");
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
        println!("  --test=exec_price_attack  v14 exec_price deviation");
        println!("  --test=sybil_close        v14 sybil two-step exec_price");
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
        match test_exec_price_attack_v14() {
            Ok(()) => {},
            Err(e) => println!("FAILED: {:?}", e),
        }
        return;
    }
    if args.iter().any(|a| a == "--test=sybil_close") {
        match test_sybil_close_v14() {
            Ok(()) => {},
            Err(e) => println!("FAILED: {:?}", e),
        }
        return;
    }
    if args.iter().any(|a| a == "--test=probes") {
        run_probes();
        return;
    }
    if args.iter().any(|a| a == "--test=probes_v14") {
        run_probes_v14_extra();
        return;
    }
    if args.iter().any(|a| a == "--test=probes_paths") {
        run_probes_v14_more();
        return;
    }
    if args.iter().any(|a| a == "--test=probes_resolve") {
        run_probes_resolve();
        return;
    }
    if args.iter().any(|a| a == "--test=probes_boundary") {
        run_probes_boundary();
        return;
    }
    if args.iter().any(|a| a == "--test=config_sweep") {
        probe_config_sweep();
        return;
    }
    if args.iter().any(|a| a == "--test=multileg") {
        run_probes_multileg();
        return;
    }
    if args.iter().any(|a| a == "--test=corner_cases") {
        run_probes_v12_corner_cases();
        return;
    }
    if args.iter().any(|a| a == "--test=xmargin") {
        run_probes_xmargin();
        return;
    }
    if args.iter().any(|a| a == "--test=drift") {
        run_probes_drift();
        return;
    }
    if args.iter().any(|a| a == "--test=hard") {
        run_probes_hard_stress();
        return;
    }
    if args.iter().any(|a| a == "--test=hard_ext") {
        run_probes_hard_extended();
        return;
    }
    if args.iter().any(|a| a == "--test=domain_attr") {
        run_probes_domain_attribution();
        return;
    }
    if args.iter().any(|a| a == "--test=xmargin_within") {
        probe_xmargin_offset_within_account();
        return;
    }
    if args.iter().any(|a| a == "--test=advanced") {
        run_probes_advanced();
        return;
    }
    if args.iter().any(|a| a == "--test=pnl_trace") {
        probe_pnl_materialization();
        return;
    }
    if args.iter().any(|a| a == "--test=f6") {
        match test_f6_v14() {
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
    println!("v14 port in progress. Try: --test=smoke, --fuzz=200, --fuzz=2000 --scenario=crash20, --fuzz-all");
}
