use std::collections::BTreeSet;

use anyhow::{anyhow, Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};
use url::Url;

use crate::{config::AppConfig, execution::ProposalSpec};

// Step 0 audit summary:
// - WebSocket loop already uses a single persistent connection with reconnect.
// - Existing message handling classified all non-error frames as ticks, risking state desync.
// - `tx.send(...).await` in the read loop can block tick processing under backpressure.
// - No per-message routing for tick/trade/system frames, which obscures downstream FSM control.
// Step 1 fix focus:
// - Keep single-connection lifecycle and reconnect semantics.
// - Add deterministic message routing (tick/trade/error/other).
// - Use non-blocking channel sends in the hot read loop.

#[derive(Debug, Clone)]
pub struct DerivWebSocketClient {
    cfg: AppConfig,
}

#[derive(Debug)]
pub enum WebSocketEvent {
    Tick(TickEvent),
    TradeUpdate(TradeUpdate),
    ApiError(ApiErrorEvent),
    Status(ConnectionStatus),
}

#[derive(Debug, Clone, Copy)]
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
    pub raw: String,
}

#[derive(Debug)]
pub enum TradeUpdate {
    Authorized {
        login_id: String,
        currency: String,
    },
    Proposal {
        id: String,
        ask_price: f64,
        probability_up: f64,
    },
    BuyAccepted {
        contract_id: u64,
        buy_price: f64,
    },
    OpenContract(OpenContractUpdate),
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenContractUpdate {
    pub contract_id: u64,
    pub is_sold: Option<bool>,
    pub profit: Option<f64>,
}

#[derive(Debug)]
pub enum WebSocketCommand {
    RequestProposal { proposal: ProposalSpec },
    Buy { proposal_id: String, price: f64 },
    SubscribeOpenContract { contract_id: u64 },
    ClearTrackedContract { contract_id: u64 },
}

#[derive(Serialize)]
struct AuthorizeRequest<'a> {
    authorize: &'a str,
}

#[derive(Serialize)]
struct TicksRequest<'a> {
    ticks: &'a str,
    subscribe: u8,
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
}

#[derive(Serialize)]
struct BuyRequest<'a> {
    buy: &'a str,
    price: f64,
}

#[derive(Serialize)]
struct ProposalOpenContractRequest {
    proposal_open_contract: u8,
    contract_id: u64,
    subscribe: u8,
}

#[derive(Debug, Deserialize)]
struct Envelope {
    msg_type: Option<String>,
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
}

impl DerivWebSocketClient {
    pub fn new(cfg: AppConfig) -> Self {
        Self { cfg }
    }

    pub async fn run(
        self,
        event_tx: mpsc::Sender<WebSocketEvent>,
        mut command_rx: mpsc::Receiver<WebSocketCommand>,
    ) -> Result<()> {
        let ws_url = self.ws_url()?;
        let mut tracked_contracts = BTreeSet::new();

        loop {
            try_emit(
                &event_tx,
                WebSocketEvent::Status(ConnectionStatus::Connecting),
            )?;
            info!(url = %ws_url, "connecting to Deriv websocket");

            match connect_async(ws_url.as_str()).await {
                Ok((stream, _)) => {
                    try_emit(
                        &event_tx,
                        WebSocketEvent::Status(ConnectionStatus::Connected),
                    )?;
                    info!("connected to Deriv websocket");

                    let (mut write, mut read) = stream.split();
                    self.send_json(
                        &mut write,
                        &AuthorizeRequest {
                            authorize: &self.cfg.token,
                        },
                    )
                    .await
                    .context("failed to authorize websocket session")?;

                    self.send_json(
                        &mut write,
                        &TicksRequest {
                            ticks: &self.cfg.symbol,
                            subscribe: 1,
                        },
                    )
                    .await
                    .context("failed to subscribe to ticks")?;
                    try_emit(
                        &event_tx,
                        WebSocketEvent::Status(ConnectionStatus::SubscribedTicks),
                    )?;

                    for contract_id in tracked_contracts.iter().copied() {
                        self.send_json(
                            &mut write,
                            &ProposalOpenContractRequest {
                                proposal_open_contract: 1,
                                contract_id,
                                subscribe: 1,
                            },
                        )
                        .await
                        .with_context(|| {
                            format!("failed to resubscribe open contract {contract_id}")
                        })?;
                    }

                    loop {
                        tokio::select! {
                            maybe_command = command_rx.recv() => {
                                let Some(command) = maybe_command else {
                                    return Ok(());
                                };

                                match command {
                                    WebSocketCommand::RequestProposal { proposal } => {
                                        let request = ProposalRequest {
                                            proposal: 1,
                                            amount: proposal.amount,
                                            basis: "stake",
                                            contract_type: &proposal.contract_type,
                                            currency: &proposal.currency,
                                            duration: proposal.duration_ticks,
                                            duration_unit: "t",
                                            symbol: &proposal.symbol,
                                        };
                                        self.send_json(&mut write, &request).await?;
                                    }
                                    WebSocketCommand::Buy { proposal_id, price } => {
                                        self.send_json(&mut write, &BuyRequest {
                                            buy: &proposal_id,
                                            price,
                                        }).await?;
                                    }
                                    WebSocketCommand::SubscribeOpenContract { contract_id } => {
                                        tracked_contracts.insert(contract_id);
                                        self.send_json(&mut write, &ProposalOpenContractRequest {
                                            proposal_open_contract: 1,
                                            contract_id,
                                            subscribe: 1,
                                        }).await?;
                                    }
                                    WebSocketCommand::ClearTrackedContract { contract_id } => {
                                        tracked_contracts.remove(&contract_id);
                                    }
                                }
                            }
                            frame = read.next() => {
                                match frame {
                                    Some(Ok(Message::Text(text))) => {
                                        self.route_message(&event_tx, &text).await?;
                                    }
                                    Some(Ok(Message::Ping(payload))) => {
                                        debug!(size = payload.len(), "received ping");
                                    }
                                    Some(Ok(Message::Pong(payload))) => {
                                        debug!(size = payload.len(), "received pong");
                                    }
                                    Some(Ok(Message::Close(frame))) => {
                                        warn!(?frame, "server closed websocket");
                                        break;
                                    }
                                    Some(Ok(Message::Binary(_))) => {
                                        debug!("ignoring binary frame");
                                    }
                                    Some(Ok(_)) => {}
                                    Some(Err(err)) => {
                                        error!(error = %err, "websocket read failed");
                                        break;
                                    }
                                    None => {
                                        warn!("websocket stream ended");
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                Err(err) => {
                    error!(error = %err, "connection failed");
                }
            }

            try_emit(
                &event_tx,
                WebSocketEvent::Status(ConnectionStatus::Disconnected),
            )?;
            tokio::time::sleep(self.cfg.reconnect_backoff).await;
        }
    }

    async fn route_message(
        &self,
        event_tx: &mpsc::Sender<WebSocketEvent>,
        raw: &str,
    ) -> Result<()> {
        let envelope: Envelope = serde_json::from_str(raw).context("failed to decode message")?;

        if let Some(error) = envelope.error {
            return try_emit(
                event_tx,
                WebSocketEvent::ApiError(ApiErrorEvent {
                    code: error.code,
                    message: error.message,
                    raw: raw.to_string(),
                }),
            );
        }

        match envelope.msg_type.as_deref() {
            Some("authorize") => {
                if let Some(authorize) = envelope.authorize {
                    try_emit(
                        event_tx,
                        WebSocketEvent::TradeUpdate(TradeUpdate::Authorized {
                            login_id: authorize.loginid,
                            currency: authorize.currency,
                        }),
                    )?;
                    try_emit(
                        event_tx,
                        WebSocketEvent::Status(ConnectionStatus::Authorized),
                    )?;
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
                    )?;
                }
            }
            Some("proposal") => {
                if let Some(proposal) = envelope.proposal {
                    try_emit(
                        event_tx,
                        WebSocketEvent::TradeUpdate(TradeUpdate::Proposal {
                            id: proposal.id,
                            ask_price: proposal.ask_price,
                            probability_up: 0.6,
                        }),
                    )?;
                }
            }
            Some("buy") => {
                if let Some(buy) = envelope.buy {
                    try_emit(
                        event_tx,
                        WebSocketEvent::TradeUpdate(TradeUpdate::BuyAccepted {
                            contract_id: buy.contract_id,
                            buy_price: buy.buy_price,
                        }),
                    )?;
                }
            }
            Some("proposal_open_contract") => {
                if let Some(update) = envelope.proposal_open_contract {
                    try_emit(
                        event_tx,
                        WebSocketEvent::TradeUpdate(TradeUpdate::OpenContract(update)),
                    )?;
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

    async fn send_json<T>(
        &self,
        write: &mut futures_util::stream::SplitSink<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
            Message,
        >,
        message: &T,
    ) -> Result<()>
    where
        T: Serialize,
    {
        let payload =
            serde_json::to_string(message).context("failed to serialize websocket message")?;
        write
            .send(Message::Text(payload))
            .await
            .context("failed to send websocket message")
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

fn try_emit(event_tx: &mpsc::Sender<WebSocketEvent>, event: WebSocketEvent) -> Result<()> {
    match event_tx.try_send(event) {
        Ok(()) => Ok(()),
        Err(mpsc::error::TrySendError::Full(_)) => {
            warn!("websocket event channel full; dropping event");
            Ok(())
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            Err(anyhow!("websocket event channel closed"))
        }
    }
}

#[derive(Debug, Deserialize)]
struct DerivEnvelope {
    msg_type: Option<String>,
    error: Option<ApiErrorPayload>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dispatch {
    Delivered,
    ReceiverClosed,
    Backpressure,
}
