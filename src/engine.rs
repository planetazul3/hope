use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use parking_lot::RwLock;
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

    // Globally unique request ID counter, shared between the Engine and the WebSocket client.
    // This atomicity ensures that concurrent subscriptions (from the client) and trade
    // requests (from the engine) never share an ID, preventing routing collisions.
    let req_id_counter = Arc::new(AtomicU32::new(100));
    let tracked_contracts = Arc::new(RwLock::new(BTreeSet::new()));

    let client = DerivWebSocketClient::new(
        config.clone(),
        Arc::clone(&req_id_counter),
        Arc::clone(&tracked_contracts),
    );
    let client_task = tokio::spawn(async move { client.run(event_tx, command_rx).await });

    let mut engine = Engine::new(config, tick_logger, req_id_counter, tracked_contracts);

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
    balance: Option<f64>,
    buffered_close_event: Option<(u64, f64)>,
    tracked_contracts: Arc<RwLock<BTreeSet<u64>>>,
}

impl Engine {
    fn new(
        config: AppConfig,
        tick_logger: TickLogger,
        req_id_counter: Arc<AtomicU32>,
        tracked_contracts: Arc<RwLock<BTreeSet<u64>>>,
    ) -> Self {
        let model = match config.model_type {
            crate::config::ModelType::Transformer => {
                if let Some(path) = &config.transformer_model_path {
                    match crate::transformer::TransformerModel::load(
                        path,
                        config.transformer_sequence_length,
                        config.model_public_key.as_deref(),
                    ) {
                        Ok(m) => AnyModel::Transformer(Box::new(m)),
                        Err(err) => {
                            error!(error = %err, path = %path, "failed to load transformer model; falling back to Gaussian");
                            AnyModel::Gaussian(GaussianModel {
                                duration_ticks: config.duration_ticks,
                                snr_threshold: config.snr_threshold,
                            })
                        }
                    }
                } else {
                    warn!("transformer model path not configured; falling back to Gaussian");
                    AnyModel::Gaussian(GaussianModel {
                        duration_ticks: config.duration_ticks,
                        snr_threshold: config.snr_threshold,
                    })
                }
            }
            crate::config::ModelType::Gaussian => AnyModel::Gaussian(GaussianModel {
                duration_ticks: config.duration_ticks,
                snr_threshold: config.snr_threshold,
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
            proposal_timeout: Duration::from_secs(config.proposal_timeout_secs),
            order_pending_timeout: Duration::from_secs(config.order_pending_timeout_secs),
            tick_logger,
            pending_req_id: None,
            pending_subscription_req_id: None,
            pending_probability: None,
            req_id_counter,
            history_buffer: vec![
                TickSnapshot::default();
                config
                    .transformer_sequence_length
                    .max(crate::tick_processor::TickProcessor::CAPACITY)
            ],
            config,
            balance: None,
            buffered_close_event: None,
            tracked_contracts,
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

                if self.fsm.state() == TradingState::Recovery {
                    self.tick_logger.try_log(
                        snapshot,
                        0.5,
                        "recovery_active",
                        self.fsm.state(),
                        tick_started_at.elapsed().as_millis(),
                    );
                    return Ok(());
                }

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
                                self.tracked_contracts.write().remove(&contract_id);
                            }
                            self.active_contract_id = None;
                            self.contract_started_at = None;
                            self.pending_subscription_req_id = None;
                            self.safe_reset();
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
                            self.safe_reset();
                            self.order_sent_at = None;
                            self.pending_req_id = None;
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
                        self.safe_reset();
                        // Ensure any pending request ID associated with this lifecycle is cleared
                        self.pending_req_id = None;
                        return Ok(());
                    }

                    if !self.config.trading_enabled {
                        info!(proposal_id = %ready.quote.id, "trading disabled; skipping buy");
                        self.safe_reset();
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
                        self.safe_reset();
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
                    self.pending_probability = Some(decision.probability_up);

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
                    self.safe_reset();
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
                    self.balance = Some(balance);
                    info!(%login_id, %currency, balance = %format!("{:.2}", balance), "authorized websocket session");
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
                    let quote = ProposalQuote {
                        id: id.clone(),
                        ask_price,
                    };

                    self.pending_req_id = None;
                    self.pending_probability = None;

                    // Optimization: Attempt immediate buy to eliminate 1-tick latency
                    let buy_req_id = self.next_req_id();
                    if let Err(reason) = try_send_buy(
                        &mut self.execution,
                        command_tx,
                        &quote,
                        Instant::now(),
                        buy_req_id,
                    ) {
                        warn!(?reason, "immediate buy skipped; falling back to next tick");
                        self.pending_proposal = Some(PendingProposal {
                            quote,
                            probability_up,
                            received_at: Instant::now(),
                        });
                        if self.fsm.state() != TradingState::OrderPending {
                            self.safe_reset();
                            let _ = self.fsm.transition(TradingState::OrderPending);
                        }
                    } else {
                        info!(proposal_id = %id, "immediate buy sent");
                        self.pending_req_id = Some(buy_req_id);
                        self.order_sent_at = Some(Instant::now());
                        self.pending_proposal = None;
                        if self.fsm.state() != TradingState::OrderPending {
                            self.safe_reset();
                            let _ = self.fsm.transition(TradingState::OrderPending);
                        }
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

                    // Race Condition Check: If an OpenContract (closed) arrived BEFORE BuyAccepted,
                    // we need to process it now.
                    if let Some((closed_id, profit)) = self.buffered_close_event.take() {
                        if closed_id == contract_id {
                            info!(%contract_id, %profit, "processing buffered close event for contract");
                            self.process_contract_closure(contract_id, profit, command_tx)?;
                        } else {
                            // If it's a different contract, put it back or discard (should not happen in strict FSM)
                            self.buffered_close_event = Some((closed_id, profit));
                        }
                    }
                }
                TradeUpdate::OpenContract(update) => {
                    if update.req_id.is_some() && update.req_id == self.pending_subscription_req_id
                    {
                        self.pending_subscription_req_id = None;
                    }

                    if update.is_sold.unwrap_or(0) == 1 {
                        if Some(update.contract_id) != self.active_contract_id {
                            if self.fsm.state() == TradingState::OrderPending
                                || self.fsm.state() == TradingState::Recovery
                            {
                                info!(contract_id = update.contract_id, "buffering early close event for contract (BuyAccepted not yet received)");
                                self.buffered_close_event =
                                    Some((update.contract_id, update.profit.unwrap_or(0.0)));
                            } else {
                                warn!(contract_id = update.contract_id, "received close event for untracked or already closed contract; ignoring");
                            }
                            return Ok(());
                        }

                        let profit = update.profit.unwrap_or(0.0);
                        self.process_contract_closure(update.contract_id, profit, command_tx)?;
                    } else {
                        // Contract is still open. If we were in Recovery, transition to InPosition.
                        if self.fsm.state() == TradingState::Recovery {
                            info!(
                                contract_id = update.contract_id,
                                "contract confirmed open; recovering to InPosition state"
                            );
                            self.active_contract_id = Some(update.contract_id);
                            let _ = self.fsm.transition(TradingState::InPosition);
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
                    warn!("active request failed; transitioning to recovery");
                    if self.fsm.state() == TradingState::OrderPending {
                        let _ = self.fsm.transition(TradingState::Recovery);
                        self.pending_proposal = None;
                    } else {
                        self.safe_reset();
                    }
                    self.pending_req_id = None;
                }

                if req_id.is_some() && req_id == self.pending_subscription_req_id {
                    error!(?req_id, ?self.active_contract_id, "contract resubscription failed; relying on safety timeout");
                    self.pending_subscription_req_id = None;
                }
            }
            WebSocketEvent::Status(status) => match status {
                ConnectionStatus::Connecting => info!("connecting"),
                ConnectionStatus::Connected => info!("connected"),
                ConnectionStatus::Authorized => info!("authorized"),
                ConnectionStatus::SubscribedTicks => info!("tick subscription active"),
                ConnectionStatus::Disconnected => {
                    warn!("websocket disconnected; safeguarding pending state in recovery");
                    if self.fsm.state() == TradingState::OrderPending
                        || self.fsm.state() == TradingState::InPosition
                    {
                        let _ = self.fsm.transition(TradingState::Recovery);
                    } else if self.fsm.state() != TradingState::Cooldown {
                        self.safe_reset();
                    }
                    self.pending_proposal = None;
                    self.order_sent_at = None;
                    self.pending_req_id = None;
                    self.pending_probability = None;
                }
            },
        }
        Ok(())
    }

    fn process_contract_closure(
        &mut self,
        contract_id: u64,
        profit: f64,
        _command_tx: &mpsc::Sender<WebSocketCommand>,
    ) -> Result<()> {
        let outcome = self.risk.on_trade_closed(profit);

        if let Some(ref mut bal) = self.balance {
            let old_balance = *bal;
            *bal += profit;
            let win_rate = if outcome.total_trades > 0 {
                (outcome.wins as f64 / outcome.total_trades as f64) * 100.0
            } else {
                0.0
            };

            info!(
                contract_id,
                profit = %format!("{:.2}", profit),
                old_balance = %format!("{:.2}", old_balance),
                new_balance = %format!("{:.2}", *bal),
                total_trades = %outcome.total_trades,
                wins = %outcome.wins,
                losses = %outcome.losses,
                win_rate = %format!("{:.2}%", win_rate),
                session_profit = %format!("{:.2}", outcome.total_profit),
                "contract closed"
            );
        } else {
            warn!(
                contract_id,
                profit, "contract closed but balance not yet initialized"
            );
        }

        self.active_contract_id = None;
        self.contract_started_at = None;
        self.pending_subscription_req_id = None;
        self.tracked_contracts.write().remove(&contract_id);

        if outcome.enter_cooldown {
            self.cooldown_remaining = self.config.cooldown_ticks;
            if self.fsm.state() != TradingState::Cooldown {
                if self.fsm.transition(TradingState::Cooldown).is_err() {
                    self.fsm.reset_to_idle();
                    self.fsm.transition(TradingState::Cooldown)?;
                }
            }
            warn!(
                consecutive_losses = outcome.consecutive_losses,
                cooldown_ticks = self.cooldown_remaining,
                "entered cooldown"
            );
        } else {
            self.safe_reset();
        }

        Ok(())
    }

    fn safe_reset(&mut self) {
        if self.fsm.state() != TradingState::Idle {
            self.fsm.reset_to_idle();
        }
        self.buffered_close_event = None;
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
            token: "token".to_string().into(),
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
            model_public_key: None,
            transformer_sequence_length: 32,
            min_trend_length: 5,
            strategy_volatility_penalty: 0.05,
            strategy_momentum_reward: 0.02,
            strategy_min_return_ratio: 0.1,
            max_consecutive_losses: 3,
            tick_audit_log_path: "/dev/null".to_string(),
            payout_ratio: 0.95,
            order_pending_timeout_secs: 10,
            proposal_timeout_secs: 60,
            snr_threshold: 0.05,
        }
    }

    #[tokio::test]
    async fn test_fsm_cooldown_decrement() -> Result<()> {
        let logger = TickLogger::start("/dev/null", 10);
        let counter = Arc::new(AtomicU32::new(1));
        let tracked_contracts = Arc::new(RwLock::new(BTreeSet::new()));
        let mut engine = Engine::new(mock_config(), logger, counter, tracked_contracts);
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
        let tracked_contracts = Arc::new(RwLock::new(BTreeSet::new()));
        let mut engine = Engine::new(mock_config(), logger, counter, tracked_contracts);
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
        let tracked_contracts = Arc::new(RwLock::new(BTreeSet::new()));
        let mut engine = Engine::new(mock_config(), logger, counter, tracked_contracts);
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
        let tracked_contracts = Arc::new(RwLock::new(BTreeSet::new()));
        let mut engine = Engine::new(mock_config(), logger, counter, tracked_contracts);
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

        assert_eq!(engine.fsm.state(), TradingState::Recovery);
        assert!(engine.pending_proposal.is_none());

        Ok(())
    }
}
