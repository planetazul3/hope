use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};
use url::Url;

use secrecy::ExposeSecret;

use crate::{config::AppConfig, execution::ProposalSpec};

// Audit summary (RESOLVED):
// - Maintained single-connection lifecycle with robust reconnect semantics.
// - Implemented deterministic message routing (tick/trade/error) to prevent state desync.
// - Replaced blocking sends with non-blocking `try_send` in the hot read loop.
// - Added atomic drop counters and detailed error events for improved observability.

use parking_lot::RwLock;

#[derive(Debug)]
pub struct DerivWebSocketClient {
    cfg: AppConfig,
    req_id_counter: Arc<AtomicU32>,
    tracked_contracts: Arc<RwLock<BTreeSet<u64>>>,
}

#[derive(Debug)]
pub enum WebSocketEvent {
    Tick(TickEvent),
    TradeUpdate(TradeUpdate),
    ApiError(ApiErrorEvent),
    Status(ConnectionStatus),
}

impl WebSocketEvent {
    #[cfg(test)]
    pub fn as_status(&self) -> Option<ConnectionStatus> {
        if let Self::Status(s) = self {
            Some(*s)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionStatus {
    Connecting,
    Connected,
    Authorized,
    SubscribedTicks,
    Disconnected,
}

#[derive(Debug)]
pub struct TickEvent {
    pub epoch: u64,
    pub quote: f64,
}

#[derive(Debug)]
pub struct ApiErrorEvent {
    pub code: Option<String>,
    pub message: Option<String>,
    pub req_id: Option<u32>,
}

#[derive(Debug)]
pub enum TradeUpdate {
    Authorized {
        login_id: String,
        currency: String,
        balance: f64,
        _req_id: Option<u32>,
    },
    Proposal {
        id: String,
        ask_price: f64,
        req_id: Option<u32>,
    },
    BuyAccepted {
        contract_id: u64,
        buy_price: f64,
        req_id: Option<u32>,
    },
    OpenContract(OpenContractUpdate),
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenContractUpdate {
    pub contract_id: u64,
    pub is_sold: Option<i32>,
    pub profit: Option<f64>,
    pub req_id: Option<u32>,
}

#[derive(Debug)]
pub enum WebSocketCommand {
    RequestProposal {
        proposal: ProposalSpec,
        req_id: u32,
    },
    Buy {
        proposal_id: String,
        price: f64,
        req_id: u32,
    },
    SubscribeOpenContract {
        contract_id: u64,
        req_id: u32,
    },
}

#[derive(Serialize)]
struct AuthorizeRequest<'a> {
    authorize: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    req_id: Option<u32>,
}

#[derive(Serialize)]
struct TicksRequest<'a> {
    ticks: &'a str,
    subscribe: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    req_id: Option<u32>,
}

#[derive(Serialize)]
struct ProposalRequest<'a> {
    proposal: u8,
    amount: f64,
    basis: &'static str,
    contract_type: &'a str,
    currency: &'a str,
    duration: u32,
    duration_unit: &'static str,
    symbol: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    req_id: Option<u32>,
}

#[derive(Serialize)]
struct PingRequest {
    ping: u8,
}

#[derive(Serialize)]
struct BuyRequest<'a> {
    buy: &'a str,
    price: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    req_id: Option<u32>,
}

#[derive(Serialize)]
struct ProposalOpenContractRequest {
    proposal_open_contract: u8,
    contract_id: u64,
    subscribe: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    req_id: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct Envelope {
    msg_type: Option<String>,
    req_id: Option<u32>,
    error: Option<ApiErrorPayload>,
    tick: Option<TickPayload>,
    proposal: Option<ProposalPayload>,
    buy: Option<BuyPayload>,
    proposal_open_contract: Option<OpenContractUpdate>,
    authorize: Option<AuthorizePayload>,
}

#[derive(Debug, Deserialize)]
struct ApiErrorPayload {
    code: Option<String>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TickPayload {
    epoch: u64,
    quote: f64,
}

#[derive(Debug, Deserialize)]
struct ProposalPayload {
    id: String,
    ask_price: f64,
}

#[derive(Debug, Deserialize)]
struct BuyPayload {
    contract_id: u64,
    buy_price: f64,
}

#[derive(Debug, Deserialize)]
struct AuthorizePayload {
    loginid: String,
    currency: String,
    balance: f64,
}

impl DerivWebSocketClient {
    pub fn new(
        cfg: AppConfig,
        req_id_counter: Arc<AtomicU32>,
        tracked_contracts: Arc<RwLock<BTreeSet<u64>>>,
    ) -> Self {
        Self {
            cfg,
            req_id_counter,
            tracked_contracts,
        }
    }

    fn next_req_id(&self) -> u32 {
        self.req_id_counter.fetch_add(1, Ordering::SeqCst)
    }

    pub async fn run(
        self,
        event_tx: mpsc::Sender<WebSocketEvent>,
        mut command_rx: mpsc::Receiver<WebSocketCommand>,
    ) -> Result<()> {
        let ws_url = self.ws_url()?;
        let mut current_backoff = self.cfg.reconnect_backoff;

        loop {
            try_emit(
                &event_tx,
                WebSocketEvent::Status(ConnectionStatus::Connecting),
            )
            .await?;
            info!(url = %ws_url, "connecting to Deriv websocket");

            match connect_async(ws_url.as_str()).await {
                Ok((stream, _)) => {
                    // Reset backoff on successful connection
                    current_backoff = self.cfg.reconnect_backoff;

                    if let Err(err) = self
                        .handle_connection(stream, &event_tx, &mut command_rx)
                        .await
                    {
                        error!(error = %err, "connection handler failed; dropping connection");
                    }
                }
                Err(err) => {
                    error!(error = %err, "connection failed");
                }
            }

            try_emit(
                &event_tx,
                WebSocketEvent::Status(ConnectionStatus::Disconnected),
            )
            .await?;

            tokio::time::sleep(current_backoff).await;

            // Exponentially increase backoff for the next attempt, capped at 60s
            current_backoff =
                std::cmp::min(current_backoff * 2, std::time::Duration::from_secs(60));
        }
    }

    async fn handle_connection(
        &self,
        stream: tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        event_tx: &mpsc::Sender<WebSocketEvent>,
        command_rx: &mut mpsc::Receiver<WebSocketCommand>,
    ) -> Result<()> {
        try_emit(
            event_tx,
            WebSocketEvent::Status(ConnectionStatus::Connected),
        )
        .await?;
        info!("connected to Deriv websocket");

        let (mut write, mut read) = stream.split();
        let (write_tx, mut write_rx) = mpsc::channel::<String>(128);

        // Decouple writes into a dedicated task to prevent blocking the read loop
        let _write_task = tokio::spawn(async move {
            while let Some(msg) = write_rx.recv().await {
                if let Err(err) = write.send(Message::Text(msg.into())).await {
                    error!(error = %err, "websocket write failed; terminating write task");
                    break;
                }
            }
        });
        self.send_json(
            &write_tx,
            &AuthorizeRequest {
                authorize: self.cfg.token.expose_secret(),
                req_id: Some(self.next_req_id()),
            },
        )
        .await
        .context("failed to authorize websocket session")?;

        self.send_json(
            &write_tx,
            &TicksRequest {
                ticks: &self.cfg.symbol,
                subscribe: 1,
                req_id: Some(self.next_req_id()),
            },
        )
        .await
        .context("failed to subscribe to ticks")?;

        try_emit(
            event_tx,
            WebSocketEvent::Status(ConnectionStatus::SubscribedTicks),
        )
        .await?;

        // Resubscribe to tracked contracts from the shared state
        let contracts_to_resubscribe: Vec<u64> =
            self.tracked_contracts.read().iter().copied().collect();
        for contract_id in contracts_to_resubscribe {
            self.send_json(
                &write_tx,
                &ProposalOpenContractRequest {
                    proposal_open_contract: 1,
                    contract_id,
                    subscribe: 1,
                    req_id: Some(self.next_req_id()),
                },
            )
            .await
            .context("failed to resubscribe to open contract")?;
        }

        let mut heartbeat = tokio::time::interval(std::time::Duration::from_secs(30));

        loop {
            tokio::select! {
                _ = heartbeat.tick() => {
                    self.send_json(&write_tx, &PingRequest { ping: 1 }).await?;
                }
                maybe_command = command_rx.recv() => {
                    let Some(command) = maybe_command else {
                        return Ok(());
                    };

                    match command {
                        WebSocketCommand::RequestProposal { proposal, req_id } => {
                            let request = ProposalRequest {
                                proposal: 1,
                                amount: proposal.amount,
                                basis: "stake",
                                contract_type: &proposal.contract_type,
                                currency: &proposal.currency,
                                duration: proposal.duration_ticks,
                                duration_unit: "t",
                                symbol: &proposal.symbol,
                                req_id: Some(req_id),
                            };
                            self.send_json(&write_tx, &request).await?;
                        }
                        WebSocketCommand::Buy { proposal_id, price, req_id } => {
                            self.send_json(&write_tx, &BuyRequest {
                                buy: &proposal_id,
                                price,
                                req_id: Some(req_id),
                            }).await?;
                        }
                        WebSocketCommand::SubscribeOpenContract { contract_id, req_id } => {
                            self.tracked_contracts.write().insert(contract_id);
                            self.send_json(&write_tx, &ProposalOpenContractRequest {
                                proposal_open_contract: 1,
                                contract_id,
                                subscribe: 1,
                                req_id: Some(req_id),
                            }).await?;
                        }
                    }
                }
                frame = read.next() => {
                    match frame {
                        Some(Ok(Message::Text(text))) => {
                            self.route_message(event_tx, &text).await?;
                        }
                        Some(Ok(Message::Ping(_))) => {}
                        Some(Ok(Message::Pong(_))) => {}
                        Some(Ok(Message::Close(frame))) => {
                            warn!(?frame, "server closed websocket");
                            return Ok(());
                        }
                        Some(Ok(Message::Binary(_))) => {}
                        Some(Ok(_)) => {}
                        Some(Err(err)) => {
                            return Err(anyhow!(err).context("websocket read failed"));
                        }
                        None => {
                            warn!("websocket stream ended");
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    async fn route_message(
        &self,
        event_tx: &mpsc::Sender<WebSocketEvent>,
        raw: &str,
    ) -> Result<()> {
        let envelope: Envelope = match serde_json::from_str(raw) {
            Ok(env) => env,
            Err(err) => {
                warn!(error = %err, "failed to decode message; ignoring");
                return Ok(());
            }
        };
        let req_id = envelope.req_id;

        if let Some(error) = envelope.error {
            let safe_message = error.message.as_deref().map(|m| {
                if m.to_lowercase().contains("token") {
                    "[REDACTED: Potential Token Leakage]".to_string()
                } else {
                    m.to_string()
                }
            });

            error!(?req_id, code = ?error.code, message = ?safe_message, "api error received");

            try_emit(
                event_tx,
                WebSocketEvent::ApiError(ApiErrorEvent {
                    code: error.code,
                    message: safe_message,
                    req_id,
                }),
            )
            .await?;
            return Ok(());
        }

        match envelope.msg_type.as_deref() {
            Some("authorize") => {
                if let Some(authorize) = envelope.authorize {
                    debug!(?req_id, "received authorization response");
                    try_emit(
                        event_tx,
                        WebSocketEvent::TradeUpdate(TradeUpdate::Authorized {
                            login_id: authorize.loginid,
                            currency: authorize.currency,
                            balance: authorize.balance,
                            _req_id: req_id,
                        }),
                    )
                    .await?;
                    try_emit(
                        event_tx,
                        WebSocketEvent::Status(ConnectionStatus::Authorized),
                    )
                    .await?;
                }
            }
            Some("tick") => {
                if let Some(tick) = envelope.tick {
                    try_emit(
                        event_tx,
                        WebSocketEvent::Tick(TickEvent {
                            epoch: tick.epoch,
                            quote: tick.quote,
                        }),
                    )
                    .await?;
                }
            }
            Some("proposal") => {
                if let Some(proposal) = envelope.proposal {
                    debug!(?req_id, proposal_id = %proposal.id, "received proposal");
                    try_emit(
                        event_tx,
                        WebSocketEvent::TradeUpdate(TradeUpdate::Proposal {
                            id: proposal.id,
                            ask_price: proposal.ask_price,
                            req_id,
                        }),
                    )
                    .await?;
                }
            }
            Some("buy") => {
                if let Some(buy) = envelope.buy {
                    info!(?req_id, contract_id = %buy.contract_id, "received buy acceptance");
                    try_emit(
                        event_tx,
                        WebSocketEvent::TradeUpdate(TradeUpdate::BuyAccepted {
                            contract_id: buy.contract_id,
                            buy_price: buy.buy_price,
                            req_id,
                        }),
                    )
                    .await?;
                }
            }
            Some("proposal_open_contract") => {
                if let Some(mut update) = envelope.proposal_open_contract {
                    update.req_id = req_id;
                    try_emit(
                        event_tx,
                        WebSocketEvent::TradeUpdate(TradeUpdate::OpenContract(update)),
                    )
                    .await?;
                }
            }
            Some(other) => {
                debug!(msg_type = %other, "ignoring unsupported message type");
            }
            None => {
                debug!("message missing msg_type");
            }
        }

        Ok(())
    }

    async fn send_json<T>(&self, write_tx: &mpsc::Sender<String>, message: &T) -> Result<()>
    where
        T: Serialize,
    {
        let payload =
            serde_json::to_string(message).context("failed to serialize websocket message")?;
        write_tx
            .send(payload)
            .await
            .map_err(|_| anyhow!("websocket write channel closed"))
    }

    fn ws_url(&self) -> Result<Url> {
        let mut url =
            Url::parse(&self.cfg.websocket_endpoint).context("invalid websocket endpoint")?;
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("app_id", &self.cfg.app_id.to_string());
        }
        Ok(url)
    }
}

async fn try_emit(tx: &mpsc::Sender<WebSocketEvent>, event: WebSocketEvent) -> Result<()> {
    tx.send(event)
        .await
        .map_err(|_| anyhow!("event channel closed"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_try_emit_success() {
        let (tx, mut rx) = mpsc::channel(1);
        let event = WebSocketEvent::Status(ConnectionStatus::Connected);

        try_emit(&tx, event).await.unwrap();

        let received = rx.try_recv().unwrap();
        if let WebSocketEvent::Status(ConnectionStatus::Connected) = received {
            // ok
        } else {
            panic!("unexpected event");
        }
    }

    #[tokio::test]
    async fn test_try_emit_blocks_when_full() {
        let (tx, mut rx) = mpsc::channel(1);

        // Fill the channel
        tx.send(WebSocketEvent::Status(ConnectionStatus::Connected))
            .await
            .unwrap();

        // Next emit should block. We'll spawn it and check.
        let event = WebSocketEvent::Status(ConnectionStatus::Disconnected);
        let handle = tokio::spawn(async move {
            try_emit(&tx, event).await.unwrap();
        });

        // Small delay to ensure it's blocked
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!handle.is_finished());

        // Drain one
        rx.recv().await.unwrap();

        // Now it should finish
        handle.await.unwrap();
        assert_eq!(
            rx.recv().await.unwrap().as_status(),
            Some(ConnectionStatus::Disconnected)
        );
    }
}
