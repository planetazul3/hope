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
    tick_processor::TickProcessor,
    websocket_client::{
        ApiErrorEvent, ConnectionStatus, DerivWebSocketClient, TradeUpdate, WebSocketCommand,
        WebSocketEvent,
    },
};

// Audit summary:
// - The original repository had a single websocket reader that forwarded raw text frames.
// - There was no FSM, no fixed-size tick processor, no rate limiting, and no risk controls.
// - The websocket task could await on bounded channel sends, which can block frame handling.
// - Trade state could not be synchronized because no typed buy/open-contract updates existed.

pub async fn run(config: AppConfig) -> Result<()> {
    let tick_logger = TickLogger::start("tick_audit.log", 4096);
    let (event_tx, mut event_rx) = mpsc::channel::<WebSocketEvent>(config.inbound_queue_capacity);
    let (command_tx, command_rx) =
        mpsc::channel::<WebSocketCommand>(config.outbound_queue_capacity);

    let client = DerivWebSocketClient::new(config.clone());
    let client_task = tokio::spawn(async move { client.run(event_tx, command_rx).await });

    let mut engine = Engine::new(config, tick_logger);

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
    order_sent_at: Option<Instant>,
    proposal_timeout: Duration,
    order_pending_timeout: Duration,
    tick_logger: TickLogger,
}

impl Engine {
    fn new(config: AppConfig, tick_logger: TickLogger) -> Self {
        let model = match config.model_type {
            crate::config::ModelType::Transformer => {
                if let Some(path) = &config.transformer_model_path {
                    match crate::transformer::TransformerModel::load(path, 16) {
                        Ok(m) => AnyModel::Transformer(m),
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
            strategy: StrategyEngine::new(config.probability_threshold, model),
            execution: ExecutionEngine::new(config.min_api_interval, config.max_tick_latency),
            risk: RiskManager::new(3),
            config,
            tick_processor: TickProcessor::new(),
            fsm: TradingFsm::new(),
            cooldown_remaining: 0,
            pending_proposal: None,
            active_contract_id: None,
            order_sent_at: None,
            proposal_timeout: Duration::from_secs(60),
            order_pending_timeout: Duration::from_secs(10),
            tick_logger,
        }
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

                if self.fsm.state() == TradingState::Idle {
                    self.fsm.transition(TradingState::Evaluating)?;
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

                    if let Err(reason) = try_send_buy(
                        &mut self.execution,
                        command_tx,
                        &ready.quote,
                        tick_started_at,
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

                let history = self.tick_processor.last_n_snapshots(16);
                let decision = self.strategy.evaluate(&snapshot, &history, self.fsm.state());
                if let Some(signal) = decision.signal {
                    let proposal_spec = ProposalSpec {
                        contract_type: self.config.contract_type.clone(),
                        currency: self.config.currency.clone(),
                        amount: self.config.stake,
                        duration_ticks: self.config.duration_ticks,
                        symbol: self.config.symbol.clone(),
                    };
                    let proposal_spec = self.execution.build_proposal(signal, &proposal_spec);

                    if let Err(reason) = try_send_proposal(
                        &mut self.execution,
                        command_tx,
                        proposal_spec,
                        tick_started_at,
                    ) {
                        warn!(?reason, "proposal skipped");
                        self.fsm.transition(TradingState::Idle)?;
                        self.tick_logger.try_log(
                            snapshot,
                            decision.probability_up,
                            "proposal_skipped",
                            self.fsm.state(),
                            tick_started_at.elapsed().as_millis(),
                        );
                        return Ok(());
                    }

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
                    self.fsm.transition(TradingState::Idle)?;
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
                TradeUpdate::Authorized { login_id, currency } => {
                    info!(%login_id, %currency, "authorized websocket session");
                }
                TradeUpdate::Proposal {
                    id,
                    ask_price,
                    probability_up,
                } => {
                    self.pending_proposal = Some(PendingProposal {
                        quote: ProposalQuote { id, ask_price },
                        probability_up,
                        received_at: Instant::now(),
                    });
                    if self.fsm.state() != TradingState::OrderPending {
                        safe_reset(&mut self.fsm);
                        self.fsm.transition(TradingState::OrderPending)?;
                    }
                }
                TradeUpdate::BuyAccepted {
                    contract_id,
                    buy_price,
                } => {
                    self.active_contract_id = Some(contract_id);
                    self.pending_proposal = None;
                    self.order_sent_at = None;
                    self.fsm.transition(TradingState::InPosition)?;
                    if command_tx
                        .try_send(WebSocketCommand::SubscribeOpenContract { contract_id })
                        .is_err()
                    {
                        error!("failed to queue open contract subscription");
                    }
                    info!(%contract_id, %buy_price, "buy confirmed");
                }
                TradeUpdate::OpenContract(update) => {
                    if update.is_sold.unwrap_or(0) == 1 {
                        let profit = update.profit.unwrap_or(0.0);
                        let outcome = self.risk.on_trade_closed(profit);
                        let closed_contract_id = update.contract_id;
                        self.active_contract_id = None;
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
            WebSocketEvent::ApiError(ApiErrorEvent { code, message, raw }) => {
                error!(?code, ?message, payload = %raw, "api error; skipping");
                if self.fsm.state() == TradingState::OrderPending {
                    safe_reset(&mut self.fsm);
                    self.pending_proposal = None;
                }
            }
            WebSocketEvent::Status(status) => match status {
                ConnectionStatus::Connecting => info!("connecting"),
                ConnectionStatus::Connected => info!("connected"),
                ConnectionStatus::Authorized => info!("authorized"),
                ConnectionStatus::SubscribedTicks => info!("tick subscription active"),
                ConnectionStatus::Disconnected => {
                    warn!("websocket disconnected");
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
) -> Result<(), crate::execution::ExecutionSkipReason> {
    execution.permit_api_call(now)?;
    command_tx
        .try_send(WebSocketCommand::RequestProposal { proposal })
        .map_err(|_| crate::execution::ExecutionSkipReason::ApiPerTickLimit)?;
    Ok(())
}

fn try_send_buy(
    execution: &mut ExecutionEngine,
    command_tx: &mpsc::Sender<WebSocketCommand>,
    quote: &ProposalQuote,
    now: Instant,
) -> Result<(), crate::execution::ExecutionSkipReason> {
    execution.permit_api_call(now)?;
    command_tx
        .try_send(WebSocketCommand::Buy {
            proposal_id: quote.id.clone(),
            price: quote.ask_price,
        })
        .map_err(|_| crate::execution::ExecutionSkipReason::ApiPerTickLimit)?;
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
        }
    }

    #[tokio::test]
    async fn test_fsm_cooldown_decrement() -> Result<()> {
        let logger = TickLogger::start("/dev/null", 10);
        let mut engine = Engine::new(mock_config(), logger);
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
        let mut engine = Engine::new(mock_config(), logger);
        let (command_tx, _command_rx) = mpsc::channel(10);

        // Transition to OrderPending via Evaluating
        engine.fsm.transition(TradingState::Evaluating)?;
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
        let mut engine = Engine::new(mock_config(), logger);
        let (command_tx, _command_rx) = mpsc::channel(10);

        engine.pending_proposal = Some(PendingProposal {
            quote: ProposalQuote {
                id: "p1".to_string(),
                ask_price: 1.0,
            },
            probability_up: 0.6,
            received_at: Instant::now() - Duration::from_secs(61),
        });
        engine.fsm.transition(TradingState::Evaluating)?;
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
        let mut engine = Engine::new(mock_config(), logger);
        let (command_tx, _command_rx) = mpsc::channel(10);

        engine.fsm.transition(TradingState::Evaluating)?;
        engine.fsm.transition(TradingState::OrderPending)?;
        engine.pending_proposal = Some(PendingProposal {
            quote: ProposalQuote {
                id: "p1".to_string(),
                ask_price: 1.0,
            },
            probability_up: 0.6,
            received_at: Instant::now(),
        });

        // API error should reset FSM and clear proposal
        engine.handle_event(
            WebSocketEvent::ApiError(ApiErrorEvent {
                code: Some("RateLimit".to_string()),
                message: Some("Too many requests".to_string()),
                raw: "{}".to_string(),
            }),
            &command_tx,
        )?;

        assert_eq!(engine.fsm.state(), TradingState::Idle);
        assert!(engine.pending_proposal.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_latency_based_skipping() -> Result<()> {
        let logger = TickLogger::start("/dev/null", 10);
        let mut config = mock_config();
        config.max_tick_latency = Duration::from_millis(5); // Aggressive latency limit
        let mut engine = Engine::new(config, logger);
        let (command_tx, mut command_rx) = mpsc::channel(10);

        // We need to ensure we have enough ticks to generate a signal
        // ConstantModel returns 0.6, threshold is 0.55
        // TickProcessor needs 2 ticks for streak
        engine.handle_event(
            WebSocketEvent::Tick(TickEvent {
                epoch: 1,
                quote: 100.0,
            }),
            &command_tx,
        )?;

        // Wait a bit to simulate latency before the next tick
        tokio::time::sleep(Duration::from_millis(10)).await;

        engine.handle_event(
            WebSocketEvent::Tick(TickEvent {
                epoch: 2,
                quote: 101.0,
            }),
            &command_tx,
        )?;

        // State should remain Idle because latency was exceeded
        assert_eq!(engine.fsm.state(), TradingState::Idle);

        // Command receiver should be empty
        assert!(command_rx.try_recv().is_err());

        Ok(())
    }
}
