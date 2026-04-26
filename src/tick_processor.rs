#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    Up,
    Down,
    #[default]
    Flat,
}

impl Direction {
    pub fn as_i8(self) -> i8 {
        match self {
            Self::Up => 1,
            Self::Down => -1,
            Self::Flat => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TickSnapshot {
    pub epoch: u64,
    pub price: f64,
    pub direction: Direction,
    pub streak: u32,
    pub volatility: f64,
    pub drift: f64,
    pub return_magnitude: f64,
    pub ticks_since_reversal: u32,
    pub db2_a1: f64,
    pub db2_d1: f64,
    pub long_term_volatility: f64,
    pub vol_ratio: f64,
}

#[derive(Debug)]
pub struct TickProcessor {
    ring: [TickSnapshot; Self::CAPACITY],
    next_index: usize,
    len: usize,
    last_price: Option<f64>,
    last_trend_direction: Direction,
    last_direction: Direction,
    last_streak: u32,
    ticks_since_reversal: u32,
    sum_returns: f64,
    sq_sum_returns: f64,
    long_sum_returns: f64,
    long_sq_sum_returns: f64,
}

const DB2_H: [f64; 4] = [
    0.482962913144690,
    0.836516303737469,
    0.224143868041857,
    -0.129409522550921,
];
const DB2_G: [f64; 4] = [
    -0.129409522550921,
    -0.224143868041857,
    0.836516303737469,
    -0.482962913144690,
];

// Reserved for future DB3 (6-tap) support
#[allow(dead_code)]
const DB3_H: [f64; 6] = [
    0.332670552950,
    0.806891509311,
    0.459877502118,
    -0.135011020010,
    -0.085441273882,
    0.035226291885,
];
#[allow(dead_code)]
const DB3_G: [f64; 6] = [
    0.035226291885,
    0.085441273882,
    -0.135011020010,
    -0.459877502118,
    0.806891509311,
    -0.332670552950,
];

impl Default for TickProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl TickProcessor {
    pub const CAPACITY: usize = 256;
    /// Window size for volatility and drift calculation.
    /// ADR 0006 specifies this horizon for short-term responsive trend detection.
    pub const VOLATILITY_WINDOW: usize = 10;
    pub const LONG_VOLATILITY_WINDOW: usize = 50;

    pub fn new() -> Self {
        Self {
            ring: [TickSnapshot::default(); Self::CAPACITY],
            next_index: 0,
            len: 0,
            last_price: None,
            last_trend_direction: Direction::Flat,
            last_direction: Direction::Flat,
            last_streak: 0,
            ticks_since_reversal: 0,
            sum_returns: 0.0,
            sq_sum_returns: 0.0,
            long_sum_returns: 0.0,
            long_sq_sum_returns: 0.0,
        }
    }

    pub fn push(&mut self, epoch: u64, price: f64) -> TickSnapshot {
        let direction = match self.last_price {
            Some(previous) if price > previous => Direction::Up,
            Some(previous) if price < previous => Direction::Down,
            Some(_) => Direction::Flat,
            None => Direction::Flat,
        };

        let streak = if matches!(direction, Direction::Flat) {
            0
        } else if direction == self.last_direction {
            self.last_streak + 1
        } else {
            1
        };

        let current_return = match self.last_price {
            Some(prev) => price - prev,
            None => 0.0,
        };

        // Reversal is defined as a flip between Up and Down, even if Flat is in between.
        if (direction == Direction::Up && self.last_trend_direction == Direction::Down)
            || (direction == Direction::Down && self.last_trend_direction == Direction::Up)
        {
            self.ticks_since_reversal = 1;
        } else if direction != Direction::Flat {
            self.ticks_since_reversal += 1;
        }

        if direction != Direction::Flat {
            self.last_trend_direction = direction;
        }

        let (db2_a1, db2_d1) = if self.len >= 4 {
            let p0 = price;
            let p1 = self.ring[(self.next_index + Self::CAPACITY - 1) % Self::CAPACITY].price;
            let p2 = self.ring[(self.next_index + Self::CAPACITY - 2) % Self::CAPACITY].price;
            let p3 = self.ring[(self.next_index + Self::CAPACITY - 3) % Self::CAPACITY].price;

            let a1 = p0 * DB2_H[0] + p1 * DB2_H[1] + p2 * DB2_H[2] + p3 * DB2_H[3];
            let d1 = p0 * DB2_G[0] + p1 * DB2_G[1] + p2 * DB2_G[2] + p3 * DB2_G[3];
            (a1, d1)
        } else {
            (0.0, 0.0)
        };

        let snapshot_without_stats = TickSnapshot {
            epoch,
            price,
            direction,
            streak,
            volatility: 0.0,
            drift: 0.0,
            return_magnitude: current_return.abs(),
            ticks_since_reversal: self.ticks_since_reversal,
            db2_a1,
            db2_d1,
            long_term_volatility: 0.0,
            vol_ratio: 0.0,
        };

        self.ring[self.next_index] = snapshot_without_stats;
        self.next_index = (self.next_index + 1) % Self::CAPACITY;
        self.len = self.len.saturating_add(1).min(Self::CAPACITY);

        // O(1) Incremental Stats Update
        if let Some(_) = self.last_price {
            // Short-term window: subtract oldest return only if we have more returns than the window size
            if self.len > Self::VOLATILITY_WINDOW + 1 {
                let idx_out = (self.next_index + Self::CAPACITY - Self::VOLATILITY_WINDOW - 2)
                    % Self::CAPACITY;
                let next_idx_out = (idx_out + 1) % Self::CAPACITY;
                let ret_out = self.ring[next_idx_out].price - self.ring[idx_out].price;
                self.sum_returns -= ret_out;
                self.sq_sum_returns -= ret_out.powi(2);
            }
            self.sum_returns += current_return;
            self.sq_sum_returns += current_return.powi(2);

            // Long-term window: subtract oldest return only if we have more returns than the window size
            if self.len > Self::LONG_VOLATILITY_WINDOW + 1 {
                let idx_out = (self.next_index + Self::CAPACITY - Self::LONG_VOLATILITY_WINDOW - 2)
                    % Self::CAPACITY;
                let next_idx_out = (idx_out + 1) % Self::CAPACITY;
                let ret_out = self.ring[next_idx_out].price - self.ring[idx_out].price;
                self.long_sum_returns -= ret_out;
                self.long_sq_sum_returns -= ret_out.powi(2);
            }
            self.long_sum_returns += current_return;
            self.long_sq_sum_returns += current_return.powi(2);
        }

        self.last_price = Some(price);
        self.last_direction = direction;
        self.last_streak = streak;

        // Calculate volatility and drift over the available history
        let (volatility, drift, _, _) = self.calculate_stats(db2_a1, db2_d1);
        let long_term_volatility = self.calculate_long_term_volatility();
        let vol_ratio = volatility / (long_term_volatility + 1e-8);

        let mut snapshot = snapshot_without_stats;
        snapshot.volatility = volatility;
        snapshot.drift = drift;
        snapshot.long_term_volatility = long_term_volatility;
        snapshot.vol_ratio = vol_ratio;

        // Update the ring buffer entry with the calculated stats
        let last_index = (self.next_index + Self::CAPACITY - 1) % Self::CAPACITY;
        self.ring[last_index] = snapshot;

        snapshot
    }

    fn calculate_stats(&self, a1: f64, d1: f64) -> (f64, f64, f64, f64) {
        let n = (self.len.saturating_sub(1)).min(Self::VOLATILITY_WINDOW) as f64;
        if n < 1.0 {
            return (0.0, 0.0, a1, d1);
        }
        let mean = self.sum_returns / n;
        let variance = (self.sq_sum_returns / n) - mean.powi(2);
        let std_dev = variance.max(0.0).sqrt();

        (std_dev, mean, a1, d1)
    }

    fn calculate_long_term_volatility(&self) -> f64 {
        let n = (self.len.saturating_sub(1)).min(Self::LONG_VOLATILITY_WINDOW) as f64;
        if n < 1.0 {
            return 0.0;
        }
        let mean = self.long_sum_returns / n;
        let variance = (self.long_sq_sum_returns / n) - mean.powi(2);
        variance.max(0.0).sqrt()
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn last_n_into(&self, n: usize, out: &mut [TickSnapshot]) -> usize {
        let count = n.min(self.len).min(out.len());
        for (i, item) in out.iter_mut().enumerate().take(count) {
            let idx = (self.next_index + Self::CAPACITY - count + i) % Self::CAPACITY;
            *item = self.ring[idx];
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_direction_and_streak_without_growth() {
        let mut processor = TickProcessor::new();

        let first = processor.push(1, 100.0);
        let second = processor.push(2, 101.0);
        let third = processor.push(3, 102.0);
        let fourth = processor.push(4, 101.0);

        assert_eq!(first.direction.as_i8(), 0);
        assert_eq!(second.direction.as_i8(), 1);
        assert_eq!(second.streak, 1);
        assert_eq!(third.streak, 2);
        assert_eq!(fourth.direction.as_i8(), -1);
        assert_eq!(fourth.streak, 1);

        for index in 0..TickProcessor::CAPACITY {
            processor.push(index as u64, index as f64);
        }
        assert_eq!(processor.len(), TickProcessor::CAPACITY);
    }

    #[test]
    fn tracks_magnitude_and_reversal_timing() {
        let mut processor = TickProcessor::new();

        // 1. Initial tick
        let s1 = processor.push(1, 100.0);
        assert_eq!(s1.return_magnitude, 0.0);
        assert_eq!(s1.ticks_since_reversal, 0);

        // 2. Trend starts (Up)
        let s2 = processor.push(2, 101.0);
        assert_eq!(s2.return_magnitude, 1.0);
        assert_eq!(s2.ticks_since_reversal, 1);

        // 3. Trend continues (Up)
        let s3 = processor.push(3, 103.0);
        assert_eq!(s3.return_magnitude, 2.0);
        assert_eq!(s3.ticks_since_reversal, 2);

        // 4. Flat (should NOT reset reversal)
        let s4 = processor.push(4, 103.0);
        assert_eq!(s4.return_magnitude, 0.0);
        assert_eq!(s4.ticks_since_reversal, 2); // Still 2 from the last active direction

        // 5. Reversal (Down)
        let s5 = processor.push(5, 102.0);
        assert_eq!(s5.return_magnitude, 1.0);
        assert_eq!(s5.ticks_since_reversal, 1); // Reset!

        // 6. Trend continues (Down)
        let s6 = processor.push(6, 101.0);
        assert_eq!(s6.ticks_since_reversal, 2);
    }

    #[test]
    fn drift_equals_population_mean() {
        let mut proc = TickProcessor::new();
        // Push 11 prices to fill window (10 returns)
        for i in 0..11 {
            proc.push(i as u64, 100.0 + i as f64);
        }
        let snap = proc.push(11, 111.0);
        let expected_drift = 1.0_f64; // all returns are 1.0
        assert!((snap.drift - expected_drift).abs() < 1e-9);
    }
}
