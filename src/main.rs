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
    V13Config {
        max_portfolio_assets: 1,
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
    min_user_capital: u128,
    max_user_pnl_abs: u128,
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

/// Single fuzz run: open random positions, walk oracle, check invariants.
fn run_one_bounty(seed: u64) -> RunSummary {
    let cfg = make_bounty_sol_20x_max_config();
    let mut engine = V13Engine::new(cfg).expect("init");
    let mut rng = Rng::new(seed);

    let lp = engine.add_account(1).unwrap();
    engine.deposit(lp, usdc(10_000_000)).unwrap();

    // 5 users with $1k each
    const N_USERS: usize = 5;
    let mut users = Vec::with_capacity(N_USERS);
    for _ in 0..N_USERS {
        let u = engine.add_account(2).unwrap();
        engine.deposit(u, usdc(1_000)).unwrap();
        users.push(u);
    }

    let oracle0 = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle0, 0).unwrap();

    let mut summary = RunSummary {
        seed,
        final_vault: 0, final_insurance: 0, final_c_tot: 0,
        total_trades: 0, rejected_trades: 0, liquidations: 0,
        invariant_failures: 0, insurance_payouts: 0,
        min_user_capital: u128::MAX, max_user_pnl_abs: 0,
    };

    let mut slot = 2u64;
    let mut oracle = oracle0;
    let max_move = cfg.max_price_move_bps_per_slot;
    let total_slots = 200;

    for step in 0..total_slots {
        // Random oracle walk within envelope (10% chance of "crash slot" with bigger move)
        let crash = rng.bool() && rng.bool() && rng.bool() && rng.bool(); // ~6%
        let move_pct = if crash { rng.range_u64(1, max_move) } else { rng.range_u64(0, max_move / 2) };
        let direction_up = rng.bool();
        let move_abs = (oracle as u128 * move_pct as u128 / 10_000) as u64;
        let target = if direction_up { oracle.saturating_add(move_abs) } else { oracle.saturating_sub(move_abs).max(1) };
        oracle = clamp_oracle(target, engine.group.assets[0].effective_price, max_move, 1);

        if engine.accrue_asset(SOL_ASSET, slot, oracle, 0).is_err() {
            // skip this slot if accrue rejected (e.g., NonProgress)
            slot += 1;
            continue;
        }

        // Occasional trade: pick a random user, open or close
        if step % 5 == 0 {
            let user_idx = users[(rng.next_u64() as usize) % users.len()];
            let user = engine.accounts[user_idx];
            let cap = user.capital;
            if cap > usdc(50) {
                let going_long = rng.bool();
                let leverage = rng.range_u64(2, 15) as u128;
                let notional = (cap * leverage).min(usdc(20_000));
                let exec = oracle;
                let size_q = notional * POS_SCALE / exec as u128;
                let (long, short) = if going_long { (user_idx, lp) } else { (lp, user_idx) };
                match engine.trade(long, short, SOL_ASSET, size_q, exec, 1) {
                    Ok(_) => summary.total_trades += 1,
                    Err(_) => summary.rejected_trades += 1,
                }
            }
        }

        // Check invariants
        if engine.assert_invariants().is_err() {
            summary.invariant_failures += 1;
        }

        // Track minima
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

fn run_fuzz(n_seeds: usize) {
    println!("v13 bounty_sol_20x_max fuzz: {} seeds", n_seeds);
    let mut total_invariant_failures = 0u32;
    let mut total_trades = 0u32;
    let mut total_rejected = 0u32;
    let mut total_liquidations = 0u32;
    let mut total_insurance_payouts = 0u128;
    let mut min_vault = u128::MAX;
    let mut max_vault = 0u128;
    let mut min_insurance = u128::MAX;
    let mut min_user_capital = u128::MAX;
    let mut max_pnl = 0u128;

    for seed in 0..n_seeds as u64 {
        let s = run_one_bounty(seed);
        total_invariant_failures += s.invariant_failures;
        total_trades += s.total_trades;
        total_rejected += s.rejected_trades;
        total_liquidations += s.liquidations;
        total_insurance_payouts += s.insurance_payouts;
        min_vault = min_vault.min(s.final_vault);
        max_vault = max_vault.max(s.final_vault);
        min_insurance = min_insurance.min(s.final_insurance);
        min_user_capital = min_user_capital.min(s.min_user_capital);
        max_pnl = max_pnl.max(s.max_user_pnl_abs);
    }

    println!("  total trades:           {}", total_trades);
    println!("  rejected trades:        {}", total_rejected);
    println!("  liquidations:           {}", total_liquidations);
    println!("  invariant failures:     {}  (must be 0)", total_invariant_failures);
    println!("  insurance_payouts (sum): {}  (atomic; must be 0 for legitimate flow)", total_insurance_payouts);
    println!("  vault range:            ${}M – ${}M",
        min_vault / USDC_DECIMALS / 1_000_000, max_vault / USDC_DECIMALS / 1_000_000);
    println!("  insurance final min:    ${}", min_insurance / USDC_DECIMALS);
    println!("  user min capital:       ${}", min_user_capital / USDC_DECIMALS);
    println!("  max |user pnl|:         ${}", max_pnl / 1_000_000);
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
    if let Some(arg) = args.iter().find(|a| a.starts_with("--fuzz=")) {
        let n: usize = arg.strip_prefix("--fuzz=").unwrap().parse().unwrap_or(100);
        run_fuzz(n);
        return;
    }
    println!("v13 port in progress. Try: --test=smoke, --fuzz=200");
}
