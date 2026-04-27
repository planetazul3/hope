use hope::tick_processor::TickProcessor;

#[test]
fn verify_tick_processor_incremental_stats() {
    let mut processor = TickProcessor::new();
    let mut prices = Vec::new();

    // Simulate 100 ticks
    for i in 0..100 {
        let price = 100.0 + (i as f64).sin() * 10.0;
        prices.push(price);
        processor.push(i as u64, price);
    }

    let snap = processor.push(101, 105.0);
    prices.push(105.0);

    // Manual calculation of volatility over last 10 returns
    let returns: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();
    let window = &returns[returns.len() - 10..];
    let mean: f64 = window.iter().sum::<f64>() / 10.0;
    let variance: f64 = window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / 10.0;
    let std_dev = variance.sqrt();

    println!("Incremental Volatility: {:.10}", snap.volatility);
    println!("Manual Volatility:      {:.10}", std_dev);

    assert!(
        (snap.volatility - std_dev).abs() < 1e-9,
        "Volatility mismatch! diff: {}",
        (snap.volatility - std_dev).abs()
    );
}
