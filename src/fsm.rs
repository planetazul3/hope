use anyhow::{anyhow, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingState {
    Idle,
    Evaluating,
    OrderPending,
    InPosition,
    Cooldown,
}

#[derive(Debug)]
pub struct TradingFsm {
    state: TradingState,
}

impl TradingFsm {
    pub fn new() -> Self {
        Self {
            state: TradingState::Idle,
        }
    }

    pub fn state(&self) -> TradingState {
        self.state
    }

    pub fn transition(&mut self, next: TradingState) -> Result<()> {
        let valid = matches!(
            (self.state, next),
            (TradingState::Idle, TradingState::Evaluating)
                | (TradingState::Idle, TradingState::Cooldown)
                | (TradingState::Evaluating, TradingState::Idle)
                | (TradingState::Evaluating, TradingState::OrderPending)
                | (TradingState::Evaluating, TradingState::Cooldown)
                | (TradingState::OrderPending, TradingState::Evaluating)
                | (TradingState::OrderPending, TradingState::InPosition)
                | (TradingState::OrderPending, TradingState::Idle)
                | (TradingState::OrderPending, TradingState::Cooldown)
                | (TradingState::InPosition, TradingState::Idle)
                | (TradingState::InPosition, TradingState::Cooldown)
                | (TradingState::Cooldown, TradingState::Idle)
                | (TradingState::Cooldown, TradingState::Evaluating)
        );

        if !valid {
            return Err(anyhow!(
                "invalid state transition: {:?} -> {:?}",
                self.state,
                next
            ));
        }

        self.state = next;
        Ok(())
    }

    pub fn reset_to_idle(&mut self) {
        self.state = TradingState::Idle;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_transitions() {
        let mut fsm = TradingFsm::new();

        assert!(fsm.transition(TradingState::InPosition).is_err());
        assert_eq!(fsm.state(), TradingState::Idle);
    }

    #[test]
    fn accepts_happy_path() {
        let mut fsm = TradingFsm::new();

        fsm.transition(TradingState::Evaluating).unwrap();
        fsm.transition(TradingState::OrderPending).unwrap();
        fsm.transition(TradingState::InPosition).unwrap();
        fsm.transition(TradingState::Idle).unwrap();

        assert_eq!(fsm.state(), TradingState::Idle);
    }
}
