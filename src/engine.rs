use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::{
    config::AppConfig,
    execution::{ExecutionEngine, ProposalQuote, ProposalSpec},
    fsm::{TradingFsm, TradingState},
    risk::RiskManager,
    strategy::{AnyModel, GaussianModel, SignalDirection, StrategyEngine},
    tick_logger::TickLogger,
    tick_processor::{TickProcessor, TickSnapshot},
    websocket_client::{
        ApiErrorEvent, ConnectionStatus, DerivWebSocketClient, TradeUpdate, WebSocketCommand,
        WebSocketEvent,
    },
};

// Audit summary (RESOLVED):
// - Implemented a robust WebSocket handler with deterministic routing and non-blocking sends.
// - Added a Finite State Machine (FSM) for strict trade lifecycle management.
// - Integrated a high-fidelity TickProcessor with dynamic history buffering.
// - Established multi-layered risk controls and rate limiting (ExecutionEngine).
// - Synchronized trade state via typed OpenContract and BuyAccepted updates.

pub async fn run(config: AppConfig) -> Result<()> {
    let tick_logger = TickLogger::start(&config.tick_audit_log_path, 4096);
    let (event_tx, mut event_rx) = mpsc::channel::<WebSocketEvent>(config.inbound_queue_capacity);
    let (command_tx, command_rx) =
        mpsc::channel::<WebSocketCommand>(config.outbound_queue_capacity);

    let req_id_counter = Arc::new(AtomicU32::new(100));
    let client = DerivWebSocketClient::new(config.clone(), Arc::clone(&req_id_counter));
    let client_task = tokio::spawn(async move { client.run(event_tx, command_rx).await });

    let mut engine = Engine::new(config, tick_logger, req_id_counter);

    while let Some(event) = event_rx.recv().await {
        engine.handle_event(event, &command_tx)?;
    }

    let join_result = client_task.await.map_err(|err| anyhow!(err))?;
    if let Err(err) = join_result {
        error!(error = %err, "websocket task failed");
    }

    Ok(())
}

struct Engine {
    config: AppConfig,
    tick_processor: TickProcessor,
    fsm: TradingFsm,
    strategy: StrategyEngine<AnyModel>,
    execution: ExecutionEngine,
    risk: RiskManager,
    cooldown_remaining: u32,
    pending_proposal: Option<PendingProposal>,
    active_contract_id: Option<u64>,
    contract_started_at: Option<Instant>,
    order_sent_at: Option<Instant>,
    proposal_timeout: Duration,
    order_pending_timeout: Duration,
    tick_logger: TickLogger,
    pending_req_id: Option<u32>,
    pending_subscription_req_id: Option<u32>,
    pending_probability: Option<f64>,
    req_id_counter: Arc<AtomicU32>,
    history_buffer: Vec<TickSnapshot>,
    balance: f64,
}

impl Engine {
    fn new(config: AppConfig, tick_logger: TickLogger, req_id_counter: Arc<AtomicU32>) -> Self {
        let model = match config.model_type {
            crate::config::ModelType::Transformer => {
                if let Some(path) = &config.transformer_model_path {
                    match crate::transformer::TransformerModel::load(
                        path,
                        config.transformer_sequence_length,
                    ) {
                        Ok(m) => AnyModel::Transformer(Box::new(m)),
                        Err(err) => {
                            error!(error = %err, path = %path, "failed to load transformer model; falling back to Gaussian");
                            AnyModel::Gaussian(GaussianModel {
                                duration_ticks: config.duration_ticks,
                            })
                        }
                    }
                } else {
                    warn!("transformer model path not configured; falling back to Gaussian");
                    AnyModel::Gaussian(GaussianModel {
                        duration_ticks: config.duration_ticks,
                    })
                }
            }
            crate::config::ModelType::Gaussian => AnyModel::Gaussian(GaussianModel {
                duration_ticks: config.duration_ticks,
            }),
        };

        Self {
            strategy: StrategyEngine::new(
                config.probability_threshold,
                model,
                config.min_trend_length,
                config.strategy_volatility_penalty,
                config.strategy_momentum_reward,
                config.strategy_min_return_ratio,
            ),
            execution: ExecutionEngine::new(config.min_api_interval, config.max_tick_latency),
            risk: RiskManager::new(config.max_consecutive_losses),
            tick_processor: TickProcessor::new(),
            fsm: TradingFsm::new(),
            cooldown_remaining: 0,
            pending_proposal: None,
            active_contract_id: None,
            contract_started_at: None,
            order_sent_at: None,
            proposal_timeout: Duration::from_secs(60),
            order_pending_timeout: Duration::from_secs(10),
            tick_logger,
            pending_req_id: None,
            pending_subscription_req_id: None,
            pending_probability: None,
            req_id_counter,
            history_buffer: vec![
                TickSnapshot::default();
                config.transformer_sequence_length.max(64)
            ],
            config,
            balance: 0.0,
        }
    }

    fn next_req_id(&mut self) -> u32 {
        self.req_id_counter.fetch_add(1, Ordering::SeqCst)
    }

    fn handle_event(
        &mut self,
        event: WebSocketEvent,
        command_tx: &mpsc::Sender<WebSocketCommand>,
    ) -> Result<()> {
        match event {
            WebSocketEvent::Tick(tick) => {
                let tick_started_at = Instant::now();
                let snapshot = self.tick_processor.push(tick.epoch, tick.quote);
                self.execution.on_tick(snapshot.epoch, tick_started_at);

                if self.fsm.state() == TradingState::Cooldown {
                    self.cooldown_remaining = self.cooldown_remaining.saturating_sub(1);
                    if self.cooldown_remaining == 0 {
                        self.fsm.transition(TradingState::Idle)?;
                    }
                    self.tick_logger.try_log(
                        snapshot,
                        0.5,
                        "cooldown",
                        self.fsm.state(),
                        tick_started_at.elapsed().as_millis(),
                    );
                    return Ok(());
                }

                if self.fsm.state() == TradingState::InPosition || self.active_contract_id.is_some()
                {
                    if let Some(started_at) = self.contract_started_at {
                        // Conservative timeout: 2x duration plus 15s buffer
                        let timeout_secs = (self.config.duration_ticks * 2 + 15) as u64;
                        if tick_started_at.duration_since(started_at)
                            > Duration::from_secs(timeout_secs)
                        {
                            warn!(?self.active_contract_id, "active contract tracked for too long; forcing clear");
                            if let Some(contract_id) = self.active_contract_id {
                                let _ =
                                    command_tx.try_send(WebSocketCommand::ClearTrackedContract {
                                        contract_id,
                                    });
                            }
                            self.active_contract_id = None;
                            self.contract_started_at = None;
                            self.pending_subscription_req_id = None;
                            safe_reset(&mut self.fsm);
                        }
                    }

                    self.tick_logger.try_log(
                        snapshot,
                        0.5,
                        "in_position",
                        self.fsm.state(),
                        tick_started_at.elapsed().as_millis(),
                    );
                    return Ok(());
                }

                if self.fsm.state() == TradingState::OrderPending && self.pending_proposal.is_none()
                {
                    if let Some(sent_at) = self.order_sent_at {
                        if tick_started_at.duration_since(sent_at) > self.order_pending_timeout {
                            warn!("order pending timeout exceeded; resetting to idle");
                            safe_reset(&mut self.fsm);
                            self.order_sent_at = None;
                        }
                    }

                    self.tick_logger.try_log(
                        snapshot,
                        0.5,
                        "awaiting_proposal_or_buy",
                        self.fsm.state(),
                        tick_started_at.elapsed().as_millis(),
                    );
                    return Ok(());
                }

                if let Some(ready) = self.pending_proposal.take() {
                    if tick_started_at.duration_since(ready.received_at) > self.proposal_timeout {
                        warn!("discarding stale proposal");
                        safe_reset(&mut self.fsm);
                        return Ok(());
                    }

                    if !self.config.trading_enabled {
                        info!(proposal_id = %ready.quote.id, "trading disabled; skipping buy");
                        safe_reset(&mut self.fsm);
                        self.tick_logger.try_log(
                            snapshot,
                            ready.probability_up,
                            "trading_disabled",
                            self.fsm.state(),
                            tick_started_at.elapsed().as_millis(),
                        );
                        return Ok(());
                    }

                    let req_id = self.next_req_id();
                    if let Err(reason) = try_send_buy(
                        &mut self.execution,
                        command_tx,
                        &ready.quote,
                        tick_started_at,
                        req_id,
                    ) {
                        warn!(?reason, "buy skipped");
                        let probability_up = ready.probability_up;
                        self.pending_proposal = Some(ready);
                        self.fsm.transition(TradingState::OrderPending)?;
                        self.tick_logger.try_log(
                            snapshot,
                            probability_up,
                            "buy_skipped",
                            self.fsm.state(),
                            tick_started_at.elapsed().as_millis(),
                        );
                        return Ok(());
                    }

                    self.pending_req_id = Some(req_id);

                    self.fsm.transition(TradingState::OrderPending)?;
                    self.order_sent_at = Some(tick_started_at);
                    self.tick_logger.try_log(
                        snapshot,
                        ready.probability_up,
                        "buy_sent",
                        self.fsm.state(),
                        tick_started_at.elapsed().as_millis(),
                    );
                    return Ok(());
                }

                let count = self.tick_processor.last_n_into(
                    self.config.transformer_sequence_length,
                    &mut self.history_buffer,
                );
                let history = &self.history_buffer[..count];
                let decision = self.strategy.evaluate(&snapshot, history, self.fsm.state());
                if let Some(signal) = decision.signal {
                    if self.fsm.state() == TradingState::OrderPending {
                        return Ok(());
                    }

                    let proposal_spec = ProposalSpec {
                        contract_type: std::borrow::Cow::Owned(self.config.contract_type.clone()),
                        currency: self.config.currency.clone(),
                        amount: self.config.stake,
                        duration_ticks: self.config.duration_ticks,
                        symbol: self.config.symbol.clone(),
                    };
                    let proposal_spec = self.execution.build_proposal(signal, &proposal_spec);

                    let req_id = self.next_req_id();
                    if let Err(reason) = try_send_proposal(
                        &mut self.execution,
                        command_tx,
                        proposal_spec,
                        tick_started_at,
                        req_id,
                    ) {
                        warn!(?reason, "proposal skipped");
                        safe_reset(&mut self.fsm);
                        self.tick_logger.try_log(
                            snapshot,
                            decision.probability_up,
                            "proposal_skipped",
                            self.fsm.state(),
                            tick_started_at.elapsed().as_millis(),
                        );
                        return Ok(());
                    }

                    self.pending_req_id = Some(req_id);

                    self.fsm.transition(TradingState::OrderPending)?;
                    self.order_sent_at = Some(tick_started_at);
                    self.tick_logger.try_log(
                        snapshot,
                        decision.probability_up,
                        signal_label(signal),
                        self.fsm.state(),
                        tick_started_at.elapsed().as_millis(),
                    );
                } else {
                    safe_reset(&mut self.fsm);
                    self.tick_logger.try_log(
                        snapshot,
                        decision.probability_up,
                        "no_signal",
                        self.fsm.state(),
                        tick_started_at.elapsed().as_millis(),
                    );
                }
            }
            WebSocketEvent::TradeUpdate(update) => match update {
                TradeUpdate::Authorized {
                    login_id,
                    currency,
                    balance,
                    ..
                } => {
                    self.balance = balance;
                    info!(%login_id, %currency, balance = %format!("{:.2}", balance), "authorized websocket session");
                    if let Some(contract_id) = self.active_contract_id {
                        let req_id = self.next_req_id();
                        self.pending_subscription_req_id = Some(req_id);
                        if command_tx
                            .try_send(WebSocketCommand::SubscribeOpenContract {
                                contract_id,
                                req_id,
                            })
                            .is_err()
                        {
                            error!("failed to queue open contract resubscription");
                        }
                    }
                }
                TradeUpdate::Proposal {
                    id,
                    ask_price,
                    req_id,
                } => {
                    if req_id != self.pending_req_id {
                        warn!(?req_id, expected = ?self.pending_req_id, "ignoring stale/mismatched proposal");
                        return Ok(());
                    }

                    let probability_up = self.pending_probability.unwrap_or(0.5);
                    self.pending_proposal = Some(PendingProposal {
                        quote: ProposalQuote { id, ask_price },
                        probability_up,
                        received_at: Instant::now(),
                    });
                    self.pending_req_id = None;
                    self.pending_probability = None;

                    if self.fsm.state() != TradingState::OrderPending {
                        safe_reset(&mut self.fsm);
                        self.fsm.transition(TradingState::OrderPending)?;
                    }
                }
                TradeUpdate::BuyAccepted {
                    contract_id,
                    buy_price,
                    req_id,
                } => {
                    if req_id != self.pending_req_id {
                        warn!(?req_id, expected = ?self.pending_req_id, "ignoring stale/mismatched buy confirmation");
                        return Ok(());
                    }

                    self.active_contract_id = Some(contract_id);
                    self.contract_started_at = Some(Instant::now());
                    self.pending_proposal = None;
                    self.order_sent_at = None;
                    self.pending_req_id = None;
                    self.pending_probability = None;

                    self.fsm.transition(TradingState::InPosition)?;
                    let sub_req_id = self.next_req_id();
                    self.pending_subscription_req_id = Some(sub_req_id);
                    if command_tx
                        .try_send(WebSocketCommand::SubscribeOpenContract {
                            contract_id,
                            req_id: sub_req_id,
                        })
                        .is_err()
                    {
                        error!("failed to queue open contract subscription");
                    }
                    info!(%contract_id, %buy_price, "buy confirmed");
                }
                TradeUpdate::OpenContract(update) => {
                    if update.req_id.is_some() && update.req_id == self.pending_subscription_req_id
                    {
                        self.pending_subscription_req_id = None;
                    }

                    if update.is_sold.unwrap_or(0) == 1 {
                        let profit = update.profit.unwrap_or(0.0);
                        let old_balance = self.balance;
                        self.balance += profit;
                        let outcome = self.risk.on_trade_closed(profit);

                        let win_rate = if outcome.total_trades > 0 {
                            (outcome.wins as f64 / outcome.total_trades as f64) * 100.0
                        } else {
                            0.0
                        };

                        info!(
                            contract_id = update.contract_id,
                            profit = %format!("{:.2}", profit),
                            old_balance = %format!("{:.2}", old_balance),
                            new_balance = %format!("{:.2}", self.balance),
                            total_trades = %outcome.total_trades,
                            wins = %outcome.wins,
                            losses = %outcome.losses,
                            win_rate = %format!("{:.2}%", win_rate),
                            session_profit = %format!("{:.2}", outcome.total_profit),
                            "contract closed"
                        );

                        let closed_contract_id = update.contract_id;
                        self.active_contract_id = None;
                        self.contract_started_at = None;
                        self.pending_subscription_req_id = None;
                        let _ = command_tx.try_send(WebSocketCommand::ClearTrackedContract {
                            contract_id: closed_contract_id,
                        });

                        if outcome.enter_cooldown {
                            self.cooldown_remaining = self.config.cooldown_ticks;
                            if self.fsm.state() == TradingState::InPosition {
                                self.fsm.transition(TradingState::Cooldown)?;
                            } else {
                                safe_reset(&mut self.fsm);
                                self.fsm.transition(TradingState::Cooldown)?;
                            }
                            warn!(
                                consecutive_losses = outcome.consecutive_losses,
                                cooldown_ticks = self.cooldown_remaining,
                                "entered cooldown"
                            );
                        } else {
                            safe_reset(&mut self.fsm);
                        }
                    }
                }
            },
            WebSocketEvent::ApiError(ApiErrorEvent {
                code,
                message,
                req_id,
            }) => {
                error!(?req_id, ?code, ?message, "api error received");

                if req_id.is_some() && req_id == self.pending_req_id {
                    warn!("active request failed; resetting FSM");
                    if self.fsm.state() == TradingState::OrderPending {
                        safe_reset(&mut self.fsm);
                        self.pending_proposal = None;
                    }
                    self.pending_req_id = None;
                }

                if req_id.is_some() && req_id == self.pending_subscription_req_id {
                    error!(?req_id, ?self.active_contract_id, "contract resubscription failed; clearing state to prevent hang");
                    if let Some(contract_id) = self.active_contract_id {
                        let _ = command_tx
                            .try_send(WebSocketCommand::ClearTrackedContract { contract_id });
                    }
                    self.active_contract_id = None;
                    self.contract_started_at = None;
                    self.pending_subscription_req_id = None;
                    safe_reset(&mut self.fsm);
                }
            }
            WebSocketEvent::Status(status) => match status {
                ConnectionStatus::Connecting => info!("connecting"),
                ConnectionStatus::Connected => info!("connected"),
                ConnectionStatus::Authorized => info!("authorized"),
                ConnectionStatus::SubscribedTicks => info!("tick subscription active"),
                ConnectionStatus::Disconnected => {
                    warn!("websocket disconnected; clearing pending state");
                    safe_reset(&mut self.fsm);
                    self.pending_proposal = None;
                    self.order_sent_at = None;
                    self.pending_req_id = None;
                    self.pending_probability = None;
                }
            },
        }
        Ok(())
    }
}

fn try_send_proposal(
    execution: &mut ExecutionEngine,
    command_tx: &mpsc::Sender<WebSocketCommand>,
    proposal: ProposalSpec,
    now: Instant,
    req_id: u32,
) -> std::result::Result<(), crate::execution::ExecutionSkipReason> {
    let guard = execution.permit_api_call(now)?;
    command_tx
        .try_send(WebSocketCommand::RequestProposal { proposal, req_id })
        .map_err(|_| crate::execution::ExecutionSkipReason::InternalQueueFull)?;
    guard.commit();
    Ok(())
}

fn try_send_buy(
    execution: &mut ExecutionEngine,
    command_tx: &mpsc::Sender<WebSocketCommand>,
    quote: &ProposalQuote,
    now: Instant,
    req_id: u32,
) -> std::result::Result<(), crate::execution::ExecutionSkipReason> {
    let guard = execution.permit_api_call(now)?;
    command_tx
        .try_send(WebSocketCommand::Buy {
            proposal_id: quote.id.clone(),
            price: quote.ask_price,
            req_id,
        })
        .map_err(|_| crate::execution::ExecutionSkipReason::InternalQueueFull)?;
    guard.commit();
    Ok(())
}

fn signal_label(signal: SignalDirection) -> &'static str {
    match signal {
        SignalDirection::Up => "proposal_call",
        SignalDirection::Down => "proposal_put",
    }
}

fn safe_reset(fsm: &mut TradingFsm) {
    if fsm.state() != TradingState::Idle {
        fsm.reset_to_idle();
    }
}

#[derive(Debug, Clone)]
struct PendingProposal {
    quote: ProposalQuote,
    probability_up: f64,
    received_at: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::websocket_client::TickEvent;

    fn mock_config() -> AppConfig {
        AppConfig {
            websocket_endpoint: "wss://example.com".to_string(),
            app_id: 1089,
            deriv_environment: crate::config::DerivEnvironment::Demo,
            token: "token".to_string(),
            symbol: "R_100".to_string(),
            contract_type: "CALL".to_string(),
            currency: "USD".to_string(),
            stake: 1.0,
            duration_ticks: 5,
            probability_threshold: 0.55,
            reconnect_backoff: Duration::from_secs(1),
            min_api_interval: Duration::from_millis(10),
            cooldown_ticks: 3,
            max_tick_latency: Duration::from_millis(100),
            inbound_queue_capacity: 10,
            outbound_queue_capacity: 10,
            trading_enabled: true,
            model_type: crate::config::ModelType::Gaussian,
            transformer_model_path: None,
            transformer_sequence_length: 32,
            min_trend_length: 5,
            strategy_volatility_penalty: 0.05,
            strategy_momentum_reward: 0.02,
            strategy_min_return_ratio: 0.1,
            max_consecutive_losses: 3,
            tick_audit_log_path: "/dev/null".to_string(),
            payout_ratio: 0.95,
        }
    }

    #[tokio::test]
    async fn test_fsm_cooldown_decrement() -> Result<()> {
        let logger = TickLogger::start("/dev/null", 10);
        let counter = Arc::new(AtomicU32::new(1));
        let mut engine = Engine::new(mock_config(), logger, counter);
        let (command_tx, _command_rx) = mpsc::channel(10);

        engine.fsm.transition(TradingState::Cooldown)?;
        engine.cooldown_remaining = 2;

        // First tick -> cooldown 1
        engine.handle_event(
            WebSocketEvent::Tick(TickEvent {
                epoch: 1,
                quote: 100.0,
            }),
            &command_tx,
        )?;
        assert_eq!(engine.fsm.state(), TradingState::Cooldown);
        assert_eq!(engine.cooldown_remaining, 1);

        // Second tick -> cooldown 0 -> Idle
        engine.handle_event(
            WebSocketEvent::Tick(TickEvent {
                epoch: 2,
                quote: 100.1,
            }),
            &command_tx,
        )?;
        assert_eq!(engine.fsm.state(), TradingState::Idle);
        assert_eq!(engine.cooldown_remaining, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_order_pending_timeout() -> Result<()> {
        let logger = TickLogger::start("/dev/null", 10);
        let counter = Arc::new(AtomicU32::new(1));
        let mut engine = Engine::new(mock_config(), logger, counter);
        let (command_tx, _command_rx) = mpsc::channel(10);

        // Transition to OrderPending
        engine.fsm.transition(TradingState::OrderPending)?;
        engine.order_sent_at = Some(Instant::now() - Duration::from_secs(11));

        // Tick should trigger timeout
        engine.handle_event(
            WebSocketEvent::Tick(TickEvent {
                epoch: 1,
                quote: 100.0,
            }),
            &command_tx,
        )?;
        assert_eq!(engine.fsm.state(), TradingState::Idle);
        assert!(engine.order_sent_at.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_stale_proposal_discard() -> Result<()> {
        let logger = TickLogger::start("/dev/null", 10);
        let counter = Arc::new(AtomicU32::new(1));
        let mut engine = Engine::new(mock_config(), logger, counter);
        let (command_tx, _command_rx) = mpsc::channel(10);

        engine.pending_proposal = Some(PendingProposal {
            quote: ProposalQuote {
                id: "p1".to_string(),
                ask_price: 1.0,
            },
            probability_up: 0.6,
            received_at: Instant::now() - Duration::from_secs(61),
        });
        engine.fsm.transition(TradingState::OrderPending)?;

        // Tick should discard stale proposal
        engine.handle_event(
            WebSocketEvent::Tick(TickEvent {
                epoch: 1,
                quote: 100.0,
            }),
            &command_tx,
        )?;
        assert_eq!(engine.fsm.state(), TradingState::Idle);
        assert!(engine.pending_proposal.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_api_error_recovery() -> Result<()> {
        let logger = TickLogger::start("/dev/null", 10);
        let counter = Arc::new(AtomicU32::new(1));
        let mut engine = Engine::new(mock_config(), logger, counter);
        let (command_tx, _command_rx) = mpsc::channel(10);

        engine.fsm.transition(TradingState::OrderPending)?;
        engine.pending_proposal = Some(PendingProposal {
            quote: ProposalQuote {
                id: "p1".to_string(),
                ask_price: 1.0,
            },
            probability_up: 0.6,
            received_at: Instant::now(),
        });

        engine.pending_req_id = Some(123);
        engine.handle_event(
            WebSocketEvent::ApiError(ApiErrorEvent {
                code: Some("RateLimit".to_string()),
                message: Some("Too many requests".to_string()),
                req_id: Some(123),
            }),
            &command_tx,
        )?;

        assert_eq!(engine.fsm.state(), TradingState::Idle);
        assert!(engine.pending_proposal.is_none());

        Ok(())
    }
}
