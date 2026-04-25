use hope::config::AppConfig;
use hope::fsm::TradingState;
use hope::strategy::{AnyModel, GaussianModel, StrategyEngine};
use hope::tick_processor::{TickProcessor, TickSnapshot};
use hope::transformer::TransformerModel;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let config = AppConfig::load().expect("failed to load configuration");

    let csv_path = "data/ticks.csv";
    let file = File::open(csv_path).expect("failed to open ticks.csv");
    let reader = BufReader::new(file);

    let mut processor = TickProcessor::new();

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

    let strategy =
        StrategyEngine::new(config.probability_threshold, model, config.min_trend_length);

    let mut total_trades = 0;
    let mut wins = 0;
    let mut losses = 0;
    let mut total_profit = 0.0;

    let stake = config.stake;
    let payout_ratio = 0.95;

    let mut in_position: Option<(hope::strategy::SignalDirection, f64, u64)> = None;
    let mut history_buffer = [TickSnapshot::default(); 64];

    for line in reader.lines() {
        let line = line.expect("failed to read line");
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }

        let epoch: u64 = parts[0].parse().unwrap_or(0);
        let quote: f64 = parts[1].parse().unwrap_or(0.0);

        if let Some((dir, entry_price, _)) = in_position {
            let profit = if dir == hope::strategy::SignalDirection::Up {
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
            continue;
        }

        let snapshot = processor.push(epoch, quote);

        // Task 6: Use dynamically loaded transformer_sequence_length
        let count = processor.last_n_into(config.transformer_sequence_length, &mut history_buffer);
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
    println!("Model:        {:?}", config.model_type);
    println!("Threshold:    {:.4}", config.probability_threshold);
    println!("Total Trades: {}", total_trades);
    println!("Wins:         {}", wins);
    println!("Losses:       {}", losses);
    println!("Win Rate:     {:.2}%", win_rate);
    println!("Total Profit: {:.2}", total_profit);
    println!("------------------------");
}
