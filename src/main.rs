mod config;
mod engine;
mod execution;
mod fsm;
mod risk;
mod strategy;
mod tick_logger;
mod tick_processor;
mod websocket_client;

use anyhow::Result;
use config::AppConfig;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("LOG_LEVEL")
                .map(|level| format!("{level},hope=debug"))
                .unwrap_or_else(|_| "info,hope=debug".to_string()),
        )
        .with_target(false)
        .compact()
        .init();

    let config = AppConfig::load()?;
    info!(environment = ?config.deriv_environment, symbol = %config.symbol, "starting trading engine");
    engine::run(config).await
}
