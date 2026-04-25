use std::borrow::Cow;
use std::time::{Duration, Instant};

use crate::strategy::SignalDirection;

#[derive(Debug, Clone)]
pub struct ProposalSpec {
    pub contract_type: Cow<'static, str>,
    pub currency: String,
    pub amount: f64,
    pub duration_ticks: u32,
    pub symbol: String,
}

#[derive(Debug, Clone)]
pub struct ProposalQuote {
    pub id: String,
    pub ask_price: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionSkipReason {
    ApiPerTickLimit,
    RateLimited,
    LatencyExceeded,
    InternalQueueFull,
}

/// A RAII-style guard that manages an API call slot for the current tick.
///
/// If the guard is dropped without being `commit()`ed, it will automatically
/// revert the engine's internal counters, effectively "refunding" the slot.
/// This ensures that transient errors (like full outbound queues) don't
/// permanently consume a tick's limited API capacity.
#[derive(Debug)]
pub struct PermitGuard<'a> {
    engine: &'a mut ExecutionEngine,
    committed: bool,
}

impl<'a> PermitGuard<'a> {
    /// Commits the permit, permanently consuming the API slot for this tick.
    /// This should only be called after a successful `try_send` to the WebSocket.
    pub fn commit(mut self) {
        self.committed = true;
    }
}

impl<'a> Drop for PermitGuard<'a> {
    fn drop(&mut self) {
        if !self.committed {
            self.engine.api_calls_this_tick = self.engine.api_calls_this_tick.saturating_sub(1);
            self.engine.last_api_call_at = self.engine.last_prev_api_call_at;
        }
    }
}

#[derive(Debug)]
pub struct ExecutionEngine {
    min_api_interval: Duration,
    max_tick_latency: Duration,
    last_api_call_at: Option<Instant>,
    last_prev_api_call_at: Option<Instant>,
    current_tick_epoch: Option<u64>,
    api_calls_this_tick: u8,
    tick_started_at: Option<Instant>,
}

impl ExecutionEngine {
    pub fn new(min_api_interval: Duration, max_tick_latency: Duration) -> Self {
        Self {
            min_api_interval,
            max_tick_latency,
            last_api_call_at: None,
            last_prev_api_call_at: None,
            current_tick_epoch: None,
            api_calls_this_tick: 0,
            tick_started_at: None,
        }
    }

    pub fn on_tick(&mut self, epoch: u64, started_at: Instant) {
        if self.current_tick_epoch != Some(epoch) {
            self.current_tick_epoch = Some(epoch);
            self.api_calls_this_tick = 0;
        }
        self.tick_started_at = Some(started_at);
    }

    pub fn permit_api_call(
        &mut self,
        now: Instant,
    ) -> Result<PermitGuard<'_>, ExecutionSkipReason> {
        if self.api_calls_this_tick >= 1 {
            return Err(ExecutionSkipReason::ApiPerTickLimit);
        }

        if let Some(last_api_call_at) = self.last_api_call_at {
            if now.duration_since(last_api_call_at) < self.min_api_interval {
                return Err(ExecutionSkipReason::RateLimited);
            }
        }

        if let Some(tick_started_at) = self.tick_started_at {
            if now.duration_since(tick_started_at) > self.max_tick_latency {
                return Err(ExecutionSkipReason::LatencyExceeded);
            }
        }

        self.last_prev_api_call_at = self.last_api_call_at;
        self.api_calls_this_tick += 1;
        self.last_api_call_at = Some(now);

        Ok(PermitGuard {
            engine: self,
            committed: false,
        })
    }

    pub fn build_proposal(&self, signal: SignalDirection, spec: &ProposalSpec) -> ProposalSpec {
        let contract_type = match signal {
            SignalDirection::Up => Cow::Borrowed("CALL"),
            SignalDirection::Down => Cow::Borrowed("PUT"),
        };

        ProposalSpec {
            contract_type,
            currency: spec.currency.clone(),
            amount: spec.amount,
            duration_ticks: spec.duration_ticks,
            symbol: spec.symbol.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_second_api_call_in_same_tick() {
        let mut execution =
            ExecutionEngine::new(Duration::from_millis(250), Duration::from_millis(10));
        let now = Instant::now();
        execution.on_tick(10, now);

        let guard = execution.permit_api_call(now).unwrap();
        guard.commit(); // Commit to keep the state

        assert_eq!(
            execution.permit_api_call(now).unwrap_err(),
            ExecutionSkipReason::ApiPerTickLimit
        );
    }
}
