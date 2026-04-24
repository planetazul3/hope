use std::time::Duration;

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};
use url::Url;

#[derive(Debug, Clone)]
pub struct WebSocketClientConfig {
    pub endpoint: String,
    pub app_id: u32,
    pub symbol: String,
    pub reconnect_backoff: Duration,
}

impl Default for WebSocketClientConfig {
    fn default() -> Self {
        Self {
            endpoint: "wss://ws.derivws.com/websockets/v3".to_string(),
            app_id: 1089,
            symbol: "R_100".to_string(),
            reconnect_backoff: Duration::from_secs(1),
        }
    }
}

#[derive(Debug)]
pub enum WebSocketEvent {
    TickRaw(String),
    ApiError {
        code: Option<String>,
        message: Option<String>,
        raw: String,
    },
    Status(WebSocketStatus),
}

#[derive(Debug)]
pub enum WebSocketStatus {
    Connecting,
    Connected,
    Subscribed,
    Disconnected,
}

#[derive(Debug)]
pub struct DerivWebSocketClient {
    cfg: WebSocketClientConfig,
}

#[derive(Serialize)]
struct TickSubscriptionRequest<'a> {
    ticks: &'a str,
    subscribe: u8,
}

#[derive(Debug, Deserialize)]
struct DerivEnvelope {
    error: Option<DerivError>,
    msg_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DerivError {
    code: Option<String>,
    message: Option<String>,
}

impl DerivWebSocketClient {
    pub fn new(cfg: WebSocketClientConfig) -> Self {
        Self { cfg }
    }

    pub async fn run(self, tx: mpsc::Sender<WebSocketEvent>) -> Result<()> {
        let ws_url = self.ws_url()?;

        loop {
            Self::send_status(&tx, WebSocketStatus::Connecting).await;
            info!(url = %ws_url, "connecting to Deriv WebSocket");

            match connect_async(ws_url.as_str()).await {
                Ok((stream, _response)) => {
                    Self::send_status(&tx, WebSocketStatus::Connected).await;
                    info!("connected to Deriv WebSocket");
                    let (mut write, mut read) = stream.split();

                    let subscribe = TickSubscriptionRequest {
                        ticks: &self.cfg.symbol,
                        subscribe: 1,
                    };
                    let payload = serde_json::to_string(&subscribe)
                        .context("failed to encode tick subscription request")?;

                    if let Err(err) = write.send(Message::Text(payload)).await {
                        error!(error = %err, "failed to send tick subscription");
                        Self::send_status(&tx, WebSocketStatus::Disconnected).await;
                        tokio::time::sleep(self.cfg.reconnect_backoff).await;
                        continue;
                    }

                    Self::send_status(&tx, WebSocketStatus::Subscribed).await;
                    info!(symbol = %self.cfg.symbol, "subscribed to tick stream");

                    while let Some(frame) = read.next().await {
                        match frame {
                            Ok(Message::Text(text)) => {
                                if let Some(error_event) = Self::to_api_error_event(&text) {
                                    warn!("received API error frame");
                                    if tx.send(error_event).await.is_err() {
                                        warn!("receiver dropped; stopping websocket client");
                                        return Ok(());
                                    }
                                } else if tx.send(WebSocketEvent::TickRaw(text)).await.is_err() {
                                    warn!("receiver dropped; stopping websocket client");
                                    return Ok(());
                                }
                            }
                            Ok(Message::Ping(payload)) => {
                                debug!(size = payload.len(), "received ping");
                            }
                            Ok(Message::Pong(payload)) => {
                                debug!(size = payload.len(), "received pong");
                            }
                            Ok(Message::Binary(_)) => {
                                debug!("ignoring binary frame");
                            }
                            Ok(Message::Close(frame)) => {
                                warn!(?frame, "socket closed by server");
                                break;
                            }
                            Err(err) => {
                                error!(error = %err, "socket read error; reconnecting");
                                break;
                            }
                            _ => {}
                        }
                    }
                }
                Err(err) => {
                    error!(error = %err, "connection failed; will reconnect");
                }
            }

            Self::send_status(&tx, WebSocketStatus::Disconnected).await;
            tokio::time::sleep(self.cfg.reconnect_backoff).await;
        }
    }

    fn ws_url(&self) -> Result<Url> {
        let mut url = Url::parse(&self.cfg.endpoint).context("invalid websocket endpoint")?;
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("app_id", &self.cfg.app_id.to_string());
        }
        Ok(url)
    }

    fn to_api_error_event(raw: &str) -> Option<WebSocketEvent> {
        let parsed = serde_json::from_str::<DerivEnvelope>(raw).ok()?;
        let err = parsed.error?;

        Some(WebSocketEvent::ApiError {
            code: err.code,
            message: err.message,
            raw: raw.to_string(),
        })
    }

    async fn send_status(tx: &mpsc::Sender<WebSocketEvent>, status: WebSocketStatus) {
        let _ = tx.send(WebSocketEvent::Status(status)).await;
    }
}
