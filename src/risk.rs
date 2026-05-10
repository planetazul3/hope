#[derive(Debug, Clone, Copy)]
pub struct RiskOutcome {
    pub consecutive_losses: u32,
    pub enter_cooldown: bool,
    pub total_trades: u32,
    pub wins: u32,
    pub losses: u32,
    pub total_profit: f64,
}

#[derive(Debug)]
pub struct RiskManager {
    consecutive_losses: u32,
    max_consecutive_losses: u32,
    total_trades: u32,
    wins: u32,
    losses: u32,
    total_profit: f64,
}

impl RiskManager {
    pub fn new(max_consecutive_losses: u32) -> Self {
        Self {
            consecutive_losses: 0,
            max_consecutive_losses,
            total_trades: 0,
            wins: 0,
            losses: 0,
            total_profit: 0.0,
        }
    }

    pub fn on_trade_closed(&mut self, profit: f64) -> RiskOutcome {
        self.total_trades += 1;
        self.total_profit += profit;

        if profit < 0.0 {
            self.consecutive_losses += 1;
            self.losses += 1;
        } else {
            // Break-even (profit == 0.0) counts as a streak reset — it interrupted
            // the losing run and should not allow cooldown to fire on the next loss.
            self.consecutive_losses = 0;
            if profit > 0.0 {
                self.wins += 1;
            }
        }

        RiskOutcome {
            consecutive_losses: self.consecutive_losses,
            enter_cooldown: self.consecutive_losses >= self.max_consecutive_losses,
            total_trades: self.total_trades,
            wins: self.wins,
            losses: self.losses,
            total_profit: self.total_profit,
        }
    }

    pub fn stats(&self) -> RiskOutcome {
        RiskOutcome {
            consecutive_losses: self.consecutive_losses,
            enter_cooldown: self.consecutive_losses >= self.max_consecutive_losses,
            total_trades: self.total_trades,
            wins: self.wins,
            losses: self.losses,
            total_profit: self.total_profit,
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
        let outcome = risk.on_trade_closed(-1.0);
        assert!(outcome.enter_cooldown);
        assert_eq!(outcome.total_trades, 3);
        assert_eq!(outcome.losses, 3);
    }

    #[test]
    fn break_even_resets_loss_streak() {
        let mut risk = RiskManager::new(3);
        risk.on_trade_closed(-1.0);
        risk.on_trade_closed(-1.0);
        // Break-even should reset the streak
        let outcome = risk.on_trade_closed(0.0);
        assert_eq!(outcome.consecutive_losses, 0);
        assert!(!outcome.enter_cooldown);
        // Next loss starts a fresh streak of 1, not 3
        let outcome = risk.on_trade_closed(-1.0);
        assert_eq!(outcome.consecutive_losses, 1);
        assert!(!outcome.enter_cooldown);
    }
}
