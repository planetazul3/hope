use crate::{
    fsm::TradingState,
    tick_processor::{Direction, TickSnapshot},
};

pub trait ProbabilityModel {
    fn probability_up(&self, tick: &TickSnapshot) -> f64;
}

#[derive(Debug, Clone, Copy)]
pub struct ConstantModel;

impl ProbabilityModel for ConstantModel {
    fn probability_up(&self, _tick: &TickSnapshot) -> f64 {
        0.6
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDirection {
    Up,
    Down,
}

#[derive(Debug, Clone, Copy)]
pub struct StrategyDecision {
    pub probability_up: f64,
    pub signal: Option<SignalDirection>,
}

pub struct StrategyEngine<M> {
    threshold: f64,
    model: M,
}

impl<M> StrategyEngine<M>
where
    M: ProbabilityModel,
{
    pub fn new(threshold: f64, model: M) -> Self {
        Self { threshold, model }
    }

    pub fn evaluate(&self, tick: &TickSnapshot, state: TradingState) -> StrategyDecision {
        if state != TradingState::Evaluating || tick.streak < 2 {
            return StrategyDecision {
                probability_up: 0.5,
                signal: None,
            };
        }

        let probability_up = self.model.probability_up(tick);
        let probability_down = 1.0 - probability_up;
        let signal = match tick.direction {
            Direction::Up if probability_up >= self.threshold => Some(SignalDirection::Up),
            Direction::Down if probability_down >= self.threshold => Some(SignalDirection::Down),
            _ => None,
        };

        StrategyDecision {
            probability_up,
            signal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tick_processor::{Direction, TickSnapshot};

    #[test]
    fn emits_signal_only_when_threshold_and_streak_match() {
        let strategy = StrategyEngine::new(0.55, ConstantModel);
        let snapshot = TickSnapshot {
            epoch: 2,
            price: 101.0,
            direction: Direction::Up,
            streak: 2,
        };

        let decision = strategy.evaluate(&snapshot, TradingState::Evaluating);

        assert_eq!(decision.probability_up, 0.6);
        assert_eq!(decision.signal, Some(SignalDirection::Up));
    }
}
