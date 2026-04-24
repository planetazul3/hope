mod websocket_client;

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::{error, info};
use websocket_client::{DerivWebSocketClient, WebSocketClientConfig, WebSocketEvent};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,hope=debug".to_string()),
        )
        .with_target(false)
        .compact()
        .init();

    let config = WebSocketClientConfig::default();
    let client = DerivWebSocketClient::new(config);

    let (tx, mut rx) = mpsc::channel::<WebSocketEvent>(1024);
    let mut task = tokio::spawn(async move { client.run(tx).await });

    loop {
        tokio::select! {
            event = rx.recv() => {
                match event {
                    Some(WebSocketEvent::TickRaw(raw)) => info!(payload = %raw, "tick frame"),
                    Some(WebSocketEvent::ApiError { code, message, raw }) => {
                        error!(?code, ?message, payload = %raw, "api error frame");
                    }
                    Some(WebSocketEvent::Status(status)) => info!(?status, "websocket status"),
                    None => {
                        error!("event stream closed");
                        break;
                    }
                }
            }
            join = &mut task => {
                match join {
                    Ok(Ok(())) => info!("websocket client exited"),
                    Ok(Err(err)) => error!(error = %err, "websocket client failed"),
                    Err(err) => error!(error = %err, "websocket task join error"),
                }
                break;
            }
        }
    }

    Ok(())
}
