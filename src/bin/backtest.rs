use hope::fsm::TradingState;
use hope::strategy::{AnyModel, GaussianModel, SignalDirection, StrategyEngine};
use hope::tick_processor::{TickProcessor, TickSnapshot};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let csv_path = "data/ticks.csv";
    let file = File::open(csv_path).expect("failed to open ticks.csv");
    let reader = BufReader::new(file);

    let mut processor = TickProcessor::new();
    let model = AnyModel::Gaussian(GaussianModel { duration_ticks: 1 });
    let strategy = StrategyEngine::new(0.55, model, 5); // threshold 0.55

    let mut total_trades = 0;
    let mut wins = 0;
    let mut losses = 0;
    let mut total_profit = 0.0;

    let mut in_position: Option<(SignalDirection, f64, u64)> = None; // direction, entry_price, entry_epoch
    let stake = 1.0;
    let payout_ratio = 0.95;

    let mut history_buffer = [TickSnapshot::default(); 64];

    for line in reader.lines() {
        let line = line.expect("failed to read line");
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }

        let epoch: u64 = parts[0].parse().unwrap_or(0);
        let quote: f64 = parts[1].parse().unwrap_or(0.0);

        // 1. If in position, check result
        if let Some((dir, entry_price, _)) = in_position {
            let profit = if dir == SignalDirection::Up {
                if quote > entry_price {
                    stake * payout_ratio
                } else {
                    -stake
                }
            } else {
                if quote < entry_price {
                    stake * payout_ratio
                } else {
                    -stake
                }
            };

            total_trades += 1;
            if profit > 0.0 {
                wins += 1;
            } else {
                losses += 1;
            }
            total_profit += profit;

            in_position = None;
            // After trade, we don't evaluate signal on the same tick to simulate cooldown/settlement
            // In Deriv, "1 tick" means it settles on the next tick.
            continue;
        }

        // 2. Update processor
        let snapshot = processor.push(epoch, quote);

        // 3. Evaluate strategy
        let count = processor.last_n_into(16, &mut history_buffer);
        let history = &history_buffer[..count];
        let decision = strategy.evaluate(&snapshot, history, TradingState::Idle);

        if let Some(signal) = decision.signal {
            in_position = Some((signal, quote, epoch));
        }
    }

    let win_rate = if total_trades > 0 {
        (wins as f64 / total_trades as f64) * 100.0
    } else {
        0.0
    };

    println!("--- Backtest Results ---");
    println!("Total Trades: {}", total_trades);
    println!("Wins:         {}", wins);
    println!("Losses:       {}", losses);
    println!("Win Rate:     {:.2}%", win_rate);
    println!("Total Profit: {:.2}", total_profit);
    println!("------------------------");
}
