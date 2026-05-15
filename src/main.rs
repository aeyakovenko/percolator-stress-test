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
        // target_effective_lag stays false. The wrapper is responsible for
        // setting raw_oracle_target_price; a more realistic wrapper would
        // post the target and let cranks walk effective up to it.
        self.group.assets[asset_index].raw_oracle_target_price = effective_price;
        self.group.accrue_asset_to_not_atomic(
            asset_index,
            now_slot,
            effective_price,
            funding_rate_e9,
            false,
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

/// Aspirational bounty config (matches max_risk.md from v12). Currently
/// fails v13 validate_exact_solvency_envelope; need to investigate which
/// parameter the v13 envelope tightens vs v12.
#[allow(dead_code)]
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
        max_abs_funding_e9_per_slot: 10_000,
        min_funding_lifetime_slots: 10,
        max_price_move_bps_per_slot: 49,
        max_account_b_settlement_chunks: 8,
        max_bankrupt_close_chunks: 8,
        public_b_chunk_atoms: MAX_VAULT_TVL,
        permissionless_recovery_enabled: true,
        stale_certificate_penalty_enabled: true,
        full_refresh_required_for_favorable_actions: true,
        public_liveness_profile_crank_forward: true,
    }
}

/// Stage 1 smoke test: create engine, add LP + user, deposit, trade, accrue.
fn smoke_test() -> V13Result<()> {
    let cfg = make_full_margin_config();
    println!("V13 stage-1 smoke: full-margin (100% mm) config");
    cfg.validate_public_user_fund()?;
    println!("  config validated");

    let mut engine = V13Engine::new(cfg)?;
    let lp = engine.add_account(1)?;
    let user = engine.add_account(2)?;
    engine.deposit(lp, usdc(10_000_000))?;
    engine.deposit(user, usdc(1_000))?;
    println!("  accounts: lp=idx{}, user=idx{}", lp, user);
    println!("  vault=${}  c_tot=${}",
        engine.group.vault / USDC_DECIMALS,
        engine.group.c_tot / USDC_DECIMALS);

    // Set up oracle for the asset
    let oracle0 = price_e6(200);
    engine.accrue_asset(SOL_ASSET, 1, oracle0, 0)?;
    println!("  asset 0 oracle initialized to ${}", oracle0 / 1_000_000);

    // Open user-long against LP-short. Note: full-margin config (100% mm)
    // means a $500 notional requires $500 capital — user has $1000 so $500
    // notional is well within IM.
    let size_q = (usdc(500) * POS_SCALE / oracle0 as u128) as u128;
    println!("  attempting trade: size_q={}  exec=${}", size_q, oracle0 / 1_000_000);
    let outcome = engine.trade(user, lp, SOL_ASSET, size_q, oracle0, 0)?;
    println!("  trade ok: notional=${}  fee_a=${}  fee_b=${}",
        outcome.notional / USDC_DECIMALS,
        outcome.fee_a / USDC_DECIMALS,
        outcome.fee_b / USDC_DECIMALS);
    println!("  user.pnl={}  user.capital=${}",
        engine.accounts[user].pnl / 1_000_000,
        engine.accounts[user].capital / USDC_DECIMALS);

    engine.assert_invariants()?;
    println!("  invariants OK");
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("Usage:");
        println!("  --test=smoke    Run stage-1 smoke test");
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

    println!("v13 port in progress. Use --test=smoke. Full suite TBD.");
}
