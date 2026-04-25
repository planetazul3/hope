use crate::{
    fsm::TradingState,
    tick_processor::{Direction, TickSnapshot},
};

pub trait ProbabilityModel {
    fn probability_up(&self, tick: &TickSnapshot, history: &[TickSnapshot]) -> f64;
}

#[derive(Debug, Clone, Copy)]
pub struct ConstantModel;

impl ProbabilityModel for ConstantModel {
    fn probability_up(&self, _tick: &TickSnapshot, _history: &[TickSnapshot]) -> f64 {
        0.6
    }
}

const VOLATILITY_EPSILON: f64 = 1e-8;

#[derive(Debug, Clone, Copy)]
pub struct GaussianModel {
    pub duration_ticks: u32,
}

impl ProbabilityModel for GaussianModel {
    fn probability_up(&self, tick: &TickSnapshot, _history: &[TickSnapshot]) -> f64 {
        use statrs::distribution::{ContinuousCDF, Normal};

        if tick.volatility <= 0.0 {
            return 0.5;
        }

        // SNR Guard: prevent signals when drift is not statistically meaningful
        let snr = tick.drift / tick.volatility;
        if snr.abs() < 0.05 {
            return 0.5;
        }

        // Use dampened volatility to avoid numerical instability
        let t_sqrt = (self.duration_ticks as f64).sqrt();
        let x = (tick.drift * t_sqrt) / (tick.volatility + VOLATILITY_EPSILON);

        let n = Normal::new(0.0, 1.0).unwrap();
        n.cdf(x)
    }
}

#[allow(dead_code)]
pub enum AnyModel {
    Constant(ConstantModel),
    Gaussian(GaussianModel),
    Transformer(Box<crate::transformer::TransformerModel>),
}

impl ProbabilityModel for AnyModel {
    fn probability_up(&self, tick: &TickSnapshot, history: &[TickSnapshot]) -> f64 {
        match self {
            Self::Constant(m) => m.probability_up(tick, history),
            Self::Gaussian(m) => m.probability_up(tick, history),
            Self::Transformer(m) => m.probability_up(tick, history),
        }
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
    min_trend_length: u32,
}

const MIN_RETURN_RATIO: f64 = 0.1;

impl<M> StrategyEngine<M>
where
    M: ProbabilityModel,
{
    pub fn new(threshold: f64, model: M, min_trend_length: u32) -> Self {
        Self {
            threshold,
            model,
            min_trend_length,
        }
    }

    pub fn evaluate(
        &self,
        tick: &TickSnapshot,
        history: &[TickSnapshot],
        state: TradingState,
    ) -> StrategyDecision {
        // Strategy: 2 ticks in the same direction AND matches general trend (drift)
        if state != TradingState::Idle || tick.streak < 2 {
            return StrategyDecision {
                probability_up: 0.5,
                signal: None,
            };
        }

        // Trend-length filter: require sustained directional run
        if tick.ticks_since_reversal < self.min_trend_length {
            return StrategyDecision {
                probability_up: 0.5,
                signal: None,
            };
        }

        // Return-magnitude filter: avoid noise-induced false signals
        if tick.return_magnitude < tick.volatility * MIN_RETURN_RATIO {
            return StrategyDecision {
                probability_up: 0.5,
                signal: None,
            };
        }

        let probability_up = self.model.probability_up(tick, history);
        let probability_down = 1.0 - probability_up;

        // General trend is positive if probability_up > 0.5 (driven by positive drift)
        // General trend is negative if probability_down > 0.5 (driven by negative drift)

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
        let snapshot = TickSnapshot {
            epoch: 2,
            price: 101.0,
            direction: Direction::Up,
            streak: 2,
            volatility: 0.5,
            drift: 0.1,
            ticks_since_reversal: 5,
            return_magnitude: 0.1,
            ..Default::default()
        };

        let strategy = StrategyEngine::new(0.55, ConstantModel, 5);
        let decision = strategy.evaluate(&snapshot, &[], TradingState::Idle);

        assert_eq!(decision.probability_up, 0.6);
        assert_eq!(decision.signal, Some(SignalDirection::Up));
    }

    #[test]
    fn rejects_short_trend() {
        let snapshot = TickSnapshot {
            epoch: 2,
            price: 101.0,
            direction: Direction::Up,
            streak: 2,
            volatility: 0.5,
            drift: 0.1,
            ticks_since_reversal: 4, // < 5
            return_magnitude: 0.1,
            ..Default::default()
        };

        let strategy = StrategyEngine::new(0.55, ConstantModel, 5);
        let decision = strategy.evaluate(&snapshot, &[], TradingState::Idle);

        assert_eq!(decision.probability_up, 0.5);
        assert_eq!(decision.signal, None);
    }
}
