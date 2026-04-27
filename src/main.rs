use anyhow::Result;
use hope::config::AppConfig;
use hope::engine;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("failed to install rustls crypto provider");

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
