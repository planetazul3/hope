use anyhow::{anyhow, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingState {
    Idle,
    OrderPending,
    Recovery,
    InPosition,
    Cooldown,
}

#[derive(Debug)]
pub struct TradingFsm {
    state: TradingState,
}

impl Default for TradingFsm {
    fn default() -> Self {
        Self::new()
    }
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
        if self.state == next {
            return Ok(());
        }

        let valid = matches!(
            (self.state, next),
            (TradingState::Idle, TradingState::OrderPending)
                | (TradingState::Idle, TradingState::InPosition)
                | (TradingState::Idle, TradingState::Cooldown)
                | (TradingState::Idle, TradingState::Recovery)
                | (TradingState::OrderPending, TradingState::InPosition)
                | (TradingState::OrderPending, TradingState::Idle)
                | (TradingState::OrderPending, TradingState::Cooldown)
                | (TradingState::OrderPending, TradingState::Recovery)
                | (TradingState::Recovery, TradingState::Idle)
                | (TradingState::Recovery, TradingState::InPosition)
                | (TradingState::InPosition, TradingState::Idle)
                | (TradingState::InPosition, TradingState::Cooldown)
                | (TradingState::InPosition, TradingState::Recovery)
                | (TradingState::Cooldown, TradingState::Idle)
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
        fsm.transition(TradingState::Cooldown).unwrap();

        assert!(fsm.transition(TradingState::InPosition).is_err());
        assert_eq!(fsm.state(), TradingState::Cooldown);
    }

    #[test]
    fn accepts_happy_path() {
        let mut fsm = TradingFsm::new();

        fsm.transition(TradingState::OrderPending).unwrap();
        fsm.transition(TradingState::InPosition).unwrap();
        fsm.transition(TradingState::Idle).unwrap();

        assert_eq!(fsm.state(), TradingState::Idle);
    }

    #[test]
    fn accepts_idle_to_in_position_for_delayed_fills() {
        let mut fsm = TradingFsm::new();
        fsm.transition(TradingState::InPosition).unwrap();
        assert_eq!(fsm.state(), TradingState::InPosition);
    }
}
