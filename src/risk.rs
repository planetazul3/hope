#[derive(Debug, Clone, Copy)]
pub struct RiskOutcome {
    pub consecutive_losses: u32,
    pub enter_cooldown: bool,
}

#[derive(Debug)]
pub struct RiskManager {
    consecutive_losses: u32,
    max_consecutive_losses: u32,
}

impl RiskManager {
    pub fn new(max_consecutive_losses: u32) -> Self {
        Self {
            consecutive_losses: 0,
            max_consecutive_losses,
        }
    }

    pub fn on_trade_closed(&mut self, profit: f64) -> RiskOutcome {
        if profit < 0.0 {
            self.consecutive_losses += 1;
        } else {
            self.consecutive_losses = 0;
        }

        RiskOutcome {
            consecutive_losses: self.consecutive_losses,
            enter_cooldown: self.consecutive_losses >= self.max_consecutive_losses,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enters_cooldown_after_three_losses() {
        let mut risk = RiskManager::new(3);

        assert!(!risk.on_trade_closed(-1.0).enter_cooldown);
        assert!(!risk.on_trade_closed(-1.0).enter_cooldown);
        assert!(risk.on_trade_closed(-1.0).enter_cooldown);
    }
}
