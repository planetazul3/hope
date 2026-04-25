use anyhow::Result;
use hope::config::AppConfig;
use hope::fsm::{TradingFsm, TradingState};
use hope::risk::RiskManager;
use hope::strategy::{AnyModel, GaussianModel, SignalDirection, StrategyEngine};
use hope::tick_processor::{TickProcessor, TickSnapshot};
use hope::transformer::TransformerModel;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<()> {
    let config = AppConfig::load().expect("failed to load configuration");

    let csv_path = {
        let args: Vec<String> = std::env::args().collect();
        let mut iter = args.iter().skip(1);
        let mut path = "data/ticks.csv".to_string();
        while let Some(arg) = iter.next() {
            if arg == "--csv" {
                if let Some(p) = iter.next() {
                    path = p.clone();
                }
            }
        }
        path
    };
    let file = File::open(&csv_path).expect("failed to open ticks.csv");
    let reader = BufReader::new(file);

    let mut processor = TickProcessor::new();
    let mut fsm = TradingFsm::new();
    let mut risk = RiskManager::new(config.max_consecutive_losses); // Maximum consecutive losses before entering cooldown, from config

    // Task 5: Dynamic model instantiation based on config
    let model = match config.model_type {
        hope::config::ModelType::Gaussian => AnyModel::Gaussian(GaussianModel {
            duration_ticks: config.duration_ticks,
        }),
        hope::config::ModelType::Transformer => {
            let model_path = config
                .transformer_model_path
                .as_deref()
                .unwrap_or("model.onnx");
            AnyModel::Transformer(Box::new(
                TransformerModel::load(model_path, config.transformer_sequence_length)
                    .expect("failed to load transformer model"),
            ))
        }
    };

    let mut strategy = StrategyEngine::new(
        config.probability_threshold,
        model,
        config.min_trend_length,
        config.strategy_volatility_penalty,
        config.strategy_momentum_reward,
        config.strategy_min_return_ratio,
    );

    let mut total_ticks = 0;
    let mut cooldown_remaining = 0;
    let mut entry_price = 0.0;
    let mut entry_tick = 0;
    let mut signal_dir = SignalDirection::Up;

    let stake = config.stake;
    let payout_ratio = config.payout_ratio;
    let mut history_buffer = [TickSnapshot::default(); 64];

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }

        let epoch: u64 = parts[0].parse().unwrap_or(0);
        let quote: f64 = parts[1].parse().unwrap_or(0.0);
        let snapshot = processor.push(epoch, quote);
        total_ticks += 1;

        match fsm.state() {
            TradingState::Idle => {
                let count =
                    processor.last_n_into(config.transformer_sequence_length, &mut history_buffer);
                let history = &history_buffer[..count];
                let decision = strategy.evaluate(&snapshot, history, TradingState::Idle);

                if let Some(signal) = decision.signal {
                    entry_price = quote;
                    entry_tick = total_ticks;
                    signal_dir = signal;
                    fsm.transition(TradingState::OrderPending)?;
                    fsm.transition(TradingState::InPosition)?;
                }
            }
            TradingState::InPosition => {
                if total_ticks - entry_tick >= config.duration_ticks {
                    let profit = match signal_dir {
                        SignalDirection::Up => {
                            if quote > entry_price {
                                stake * payout_ratio
                            } else {
                                -stake
                            }
                        }
                        SignalDirection::Down => {
                            if quote < entry_price {
                                stake * payout_ratio
                            } else {
                                -stake
                            }
                        }
                    };

                    let outcome = risk.on_trade_closed(profit);
                    if outcome.enter_cooldown {
                        cooldown_remaining = config.cooldown_ticks;
                        fsm.transition(TradingState::Cooldown)?;
                    } else {
                        fsm.transition(TradingState::Idle)?;
                    }
                }
            }
            TradingState::Cooldown => {
                cooldown_remaining = cooldown_remaining.saturating_sub(1);
                if cooldown_remaining == 0 {
                    fsm.transition(TradingState::Idle)?;
                }
            }
            TradingState::OrderPending => {
                // Should not happen in backtest as we jump directly to InPosition
                fsm.transition(TradingState::InPosition)?;
            }
        }
    }

    let outcome = risk.stats();
    let win_rate = if outcome.total_trades > 0 {
        (outcome.wins as f64 / outcome.total_trades as f64) * 100.0
    } else {
        0.0
    };

    println!("--- Backtest Results ---");
    println!("Model:        {:?}", config.model_type);
    println!("Threshold:    {:.4}", config.probability_threshold);
    println!("Total Trades: {}", outcome.total_trades);
    println!("Wins:         {}", outcome.wins);
    println!("Losses:       {}", outcome.losses);
    println!("Win Rate:     {:.2}%", win_rate);
    println!("Total Profit: {:.2}", outcome.total_profit);
    println!("------------------------");

    Ok(())
}
