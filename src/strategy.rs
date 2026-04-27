use crate::{
    fsm::TradingState,
    tick_processor::{Direction, TickSnapshot},
};
use once_cell::sync::Lazy;
use statrs::distribution::{ContinuousCDF, Normal};

static STD_NORMAL: Lazy<Normal> = Lazy::new(|| Normal::new(0.0, 1.0).expect("valid params"));

pub trait ProbabilityModel {
    fn probability_up(&self, tick: &TickSnapshot, history: &[TickSnapshot]) -> f64;
}

const VOLATILITY_EPSILON: f64 = 1e-8;

#[derive(Debug, Clone, Copy)]
pub struct GaussianModel {
    pub duration_ticks: u32,
    pub snr_threshold: f64,
}

impl ProbabilityModel for GaussianModel {
    fn probability_up(&self, tick: &TickSnapshot, _history: &[TickSnapshot]) -> f64 {
        if tick.volatility <= 0.0 {
            return 0.5;
        }

        let snr = tick.drift / tick.volatility;
        if snr.abs() < self.snr_threshold {
            return 0.5;
        }

        let t_sqrt = (self.duration_ticks as f64).sqrt();
        let x = (tick.drift * t_sqrt) / (tick.volatility + VOLATILITY_EPSILON);

        STD_NORMAL.cdf(x)
    }
}

pub enum AnyModel {
    Gaussian(GaussianModel),
    Transformer(Box<crate::transformer::TransformerModel>),
}

impl ProbabilityModel for AnyModel {
    fn probability_up(&self, tick: &TickSnapshot, history: &[TickSnapshot]) -> f64 {
        match self {
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

/// Core engine for evaluating tick data and generating trading signals.
/// Uses a probability model (Gaussian or Transformer) and applies multiple
/// filters and dynamic thresholds to minimize false positives.
pub struct StrategyEngine<M> {
    /// Base probability threshold (e.g., 0.55 for 55% confidence).
    threshold: f64,
    /// The underlying probability model used for inference.
    model: M,
    /// Minimum number of ticks a trend must persist before considering a signal.
    min_trend_length: u32,
    /// Penalty added to the threshold during low-volatility regimes to avoid noise.
    volatility_penalty: f64,
    /// Reward subtracted from the threshold when strong momentum (long streaks) is detected.
    momentum_reward: f64,
    /// Minimum return magnitude as a ratio of volatility required to signal.
    min_return_ratio: f64,
}

impl<M> StrategyEngine<M>
where
    M: ProbabilityModel,
{
    pub fn new(
        threshold: f64,
        model: M,
        min_trend_length: u32,
        volatility_penalty: f64,
        momentum_reward: f64,
        min_return_ratio: f64,
    ) -> Self {
        Self {
            threshold,
            model,
            min_trend_length,
            volatility_penalty,
            momentum_reward,
            min_return_ratio,
        }
    }

    pub fn evaluate(
        &self,
        tick: &TickSnapshot,
        history: &[TickSnapshot],
        state: TradingState,
    ) -> StrategyDecision {
        let probability_up = self.model.probability_up(tick, history);
        let probability_down = 1.0 - probability_up;

        // Base signal check: require Idle state and minimum streak
        if state != TradingState::Idle || tick.streak < 2 {
            return StrategyDecision {
                probability_up,
                signal: None,
            };
        }

        // Trend-length filter: require sustained directional run
        if tick.ticks_since_reversal < self.min_trend_length {
            return StrategyDecision {
                probability_up,
                signal: None,
            };
        }

        // Explicit zero-volatility block to prevent AUD-006 bypass
        if tick.volatility < 1e-9 {
            return StrategyDecision {
                probability_up,
                signal: None,
            };
        }

        // Return-magnitude filter: avoid noise-induced false signals
        if tick.return_magnitude < tick.volatility * self.min_return_ratio {
            return StrategyDecision {
                probability_up,
                signal: None,
            };
        }

        // Task 8: Dynamic confidence threshold using fields
        let mut adjusted_threshold = self.threshold;

        // Reward strong momentum
        if tick.streak >= 4 {
            adjusted_threshold -= self.momentum_reward;
        }

        // Penalty for low volatility
        if tick.volatility < 0.0001 {
            adjusted_threshold += self.volatility_penalty;
        }

        // Clamp to ensure it doesn't fall below floor (never below 0.5)
        adjusted_threshold = adjusted_threshold.max(0.5_f64);

        // General trend is positive if probability_up > 0.5 (driven by positive drift)
        // General trend is negative if probability_down > 0.5 (driven by negative drift)

        let signal = match tick.direction {
            Direction::Up if probability_up >= adjusted_threshold => Some(SignalDirection::Up),
            Direction::Down if probability_down >= adjusted_threshold => {
                Some(SignalDirection::Down)
            }
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

    #[derive(Debug, Clone, Copy)]
    struct VariableModel(f64);

    impl ProbabilityModel for VariableModel {
        fn probability_up(&self, _tick: &TickSnapshot, _history: &[TickSnapshot]) -> f64 {
            self.0
        }
    }

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

        let strategy = StrategyEngine::new(0.55, VariableModel(0.6), 5, 0.05, 0.02, 0.1);
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

        let strategy = StrategyEngine::new(0.55, VariableModel(0.6), 5, 0.05, 0.02, 0.1);
        let decision = strategy.evaluate(&snapshot, &[], TradingState::Idle);

        assert_eq!(decision.probability_up, 0.6);
        assert_eq!(decision.signal, None);
    }

    #[test]
    fn applies_dynamic_threshold_on_low_volatility() {
        let snapshot = TickSnapshot {
            epoch: 2,
            price: 101.0,
            direction: Direction::Up,
            streak: 2,
            volatility: 0.00005, // < 0.0001
            drift: 0.1,
            ticks_since_reversal: 5,
            return_magnitude: 0.1,
            ..Default::default()
        };

        // Case 1: VariableModel(0.6) returns 0.6.
        // Base threshold 0.58. Adjusted is 0.63.
        // 0.6 >= 0.58 but 0.6 < 0.63. Should be None.
        let strategy = StrategyEngine::new(0.58, VariableModel(0.6), 5, 0.05, 0.02, 0.1);
        let decision = strategy.evaluate(&snapshot, &[], TradingState::Idle);
        assert_eq!(decision.signal, None);

        // Case 2: Volatility is high. Base threshold 0.58. Adjusted is 0.58.
        // 0.6 >= 0.58. Should be Some(Up).
        let mut snapshot_high_vol = snapshot;
        snapshot_high_vol.volatility = 0.5;
        let decision_high_vol = strategy.evaluate(&snapshot_high_vol, &[], TradingState::Idle);
        assert_eq!(decision_high_vol.signal, Some(SignalDirection::Up));
    }

    #[test]
    fn test_threshold_floor_clamping() {
        let snapshot = TickSnapshot {
            epoch: 2,
            price: 101.0,
            direction: Direction::Up,
            streak: 4,           // Triggers momentum reward
            volatility: 0.00005, // Triggers volatility penalty
            drift: 0.1,
            ticks_since_reversal: 5,
            return_magnitude: 0.1,
            ..Default::default()
        };

        // Case: Base 0.51, Reward 0.05, Penalty 0.02.
        // Adjusted: 0.51 - 0.05 + 0.02 = 0.48.
        // Floor should clamp to 0.5.
        let strategy = StrategyEngine::new(0.51, VariableModel(0.49), 5, 0.02, 0.05, 0.1);
        let decision = strategy.evaluate(&snapshot, &[], TradingState::Idle);

        // Model returns 0.49. Since adjusted_threshold is 0.5, 0.49 < 0.5 -> No signal.
        // If it didn't clamp, 0.49 >= 0.48 -> Signal.
        assert_eq!(decision.signal, None);
    }
}
