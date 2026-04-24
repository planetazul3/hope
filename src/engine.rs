use std::time::Instant;

use anyhow::{anyhow, Result};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::{
    config::AppConfig,
    execution::{ExecutionEngine, ProposalQuote, ProposalSpec},
    fsm::{TradingFsm, TradingState},
    risk::RiskManager,
    strategy::{ConstantModel, SignalDirection, StrategyEngine},
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

    let mut tick_processor = TickProcessor::new();
    let mut fsm = TradingFsm::new();
    let strategy = StrategyEngine::new(config.probability_threshold, ConstantModel);
    let mut execution = ExecutionEngine::new(config.min_api_interval, config.max_tick_latency);
    let mut risk = RiskManager::new(3);
    let mut cooldown_remaining = 0_u32;
    let mut pending_proposal: Option<PendingProposal> = None;
    let mut active_contract_id: Option<u64> = None;

    while let Some(event) = event_rx.recv().await {
        match event {
            WebSocketEvent::Tick(tick) => {
                let tick_started_at = Instant::now();
                let snapshot = tick_processor.push(tick.epoch, tick.quote);
                execution.on_tick(snapshot.epoch, tick_started_at);

                if fsm.state() == TradingState::Cooldown {
                    cooldown_remaining = cooldown_remaining.saturating_sub(1);
                    if cooldown_remaining == 0 {
                        fsm.transition(TradingState::Idle)?;
                    }
                    tick_logger.try_log(
                        snapshot,
                        0.5,
                        "cooldown",
                        fsm.state(),
                        tick_started_at.elapsed().as_millis(),
                    );
                    continue;
                }

                if fsm.state() == TradingState::InPosition || active_contract_id.is_some() {
                    tick_logger.try_log(
                        snapshot,
                        0.5,
                        "in_position",
                        fsm.state(),
                        tick_started_at.elapsed().as_millis(),
                    );
                    continue;
                }

                if fsm.state() == TradingState::OrderPending && pending_proposal.is_none() {
                    tick_logger.try_log(
                        snapshot,
                        0.5,
                        "awaiting_proposal",
                        fsm.state(),
                        tick_started_at.elapsed().as_millis(),
                    );
                    continue;
                }

                if fsm.state() == TradingState::Idle {
                    fsm.transition(TradingState::Evaluating)?;
                }

                if let Some(ready) = pending_proposal.take() {
                    if fsm.state() == TradingState::OrderPending {
                        fsm.transition(TradingState::Evaluating)?;
                    }

                    if let Err(reason) =
                        try_send_buy(&mut execution, &command_tx, &ready.quote, tick_started_at)
                    {
                        warn!(?reason, "buy skipped");
                        let probability_up = ready.probability_up;
                        pending_proposal = Some(ready);
                        fsm.transition(TradingState::OrderPending)?;
                        tick_logger.try_log(
                            snapshot,
                            probability_up,
                            "buy_skipped",
                            fsm.state(),
                            tick_started_at.elapsed().as_millis(),
                        );
                        continue;
                    }

                    fsm.transition(TradingState::OrderPending)?;
                    tick_logger.try_log(
                        snapshot,
                        ready.probability_up,
                        "buy_sent",
                        fsm.state(),
                        tick_started_at.elapsed().as_millis(),
                    );
                    continue;
                }

                let decision = strategy.evaluate(&snapshot, fsm.state());
                if let Some(signal) = decision.signal {
                    let proposal_spec = ProposalSpec {
                        contract_type: config.contract_type.clone(),
                        currency: config.currency.clone(),
                        amount: config.stake,
                        duration_ticks: config.duration_ticks,
                        symbol: config.symbol.clone(),
                    };
                    let proposal_spec = execution.build_proposal(signal, &proposal_spec);

                    if let Err(reason) = try_send_proposal(
                        &mut execution,
                        &command_tx,
                        proposal_spec,
                        tick_started_at,
                    ) {
                        warn!(?reason, "proposal skipped");
                        fsm.transition(TradingState::Idle)?;
                        tick_logger.try_log(
                            snapshot,
                            decision.probability_up,
                            "proposal_skipped",
                            fsm.state(),
                            tick_started_at.elapsed().as_millis(),
                        );
                        continue;
                    }

                    fsm.transition(TradingState::OrderPending)?;
                    tick_logger.try_log(
                        snapshot,
                        decision.probability_up,
                        signal_label(signal),
                        fsm.state(),
                        tick_started_at.elapsed().as_millis(),
                    );
                } else {
                    fsm.transition(TradingState::Idle)?;
                    tick_logger.try_log(
                        snapshot,
                        decision.probability_up,
                        "no_signal",
                        fsm.state(),
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
                    pending_proposal = Some(PendingProposal {
                        quote: ProposalQuote { id, ask_price },
                        probability_up,
                    });
                    if fsm.state() != TradingState::OrderPending {
                        safe_reset(&mut fsm);
                        fsm.transition(TradingState::OrderPending)?;
                    }
                }
                TradeUpdate::BuyAccepted {
                    contract_id,
                    buy_price,
                } => {
                    active_contract_id = Some(contract_id);
                    pending_proposal = None;
                    fsm.transition(TradingState::InPosition)?;
                    if command_tx
                        .try_send(WebSocketCommand::SubscribeOpenContract { contract_id })
                        .is_err()
                    {
                        error!("failed to queue open contract subscription");
                    }
                    info!(%contract_id, %buy_price, "buy confirmed");
                }
                TradeUpdate::OpenContract(update) => {
                    if update.is_sold.unwrap_or(false) {
                        let profit = update.profit.unwrap_or(0.0);
                        let outcome = risk.on_trade_closed(profit);
                        let closed_contract_id = update.contract_id;
                        active_contract_id = None;
                        let _ = command_tx.try_send(WebSocketCommand::ClearTrackedContract {
                            contract_id: closed_contract_id,
                        });

                        if outcome.enter_cooldown {
                            cooldown_remaining = config.cooldown_ticks;
                            if fsm.state() == TradingState::InPosition {
                                fsm.transition(TradingState::Cooldown)?;
                            } else {
                                safe_reset(&mut fsm);
                                fsm.transition(TradingState::Cooldown)?;
                            }
                            warn!(
                                consecutive_losses = outcome.consecutive_losses,
                                cooldown_ticks = cooldown_remaining,
                                "entered cooldown"
                            );
                        } else {
                            safe_reset(&mut fsm);
                        }
                    }
                }
            },
            WebSocketEvent::ApiError(ApiErrorEvent { code, message, raw }) => {
                error!(?code, ?message, payload = %raw, "api error; skipping");
                if fsm.state() == TradingState::OrderPending {
                    safe_reset(&mut fsm);
                    pending_proposal = None;
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
    }

    let join_result = client_task.await.map_err(|err| anyhow!(err))?;
    if let Err(err) = join_result {
        error!(error = %err, "websocket task failed");
    }

    Ok(())
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
}
