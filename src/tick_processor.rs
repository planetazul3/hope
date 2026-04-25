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
    return_sum: f64,
    return_sq_sum: f64,
}

impl Default for TickProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl TickProcessor {
    pub const CAPACITY: usize = 64;
    /// Window size for volatility and drift calculation.
    /// ADR 0006 specifies this horizon for short-term responsive trend detection.
    pub const VOLATILITY_WINDOW: usize = 10;

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
            return_sum: 0.0,
            return_sq_sum: 0.0,
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

        // Maintain running sums for O(1) mean and variance calculation.
        // If the buffer is full relative to the VOLATILITY_WINDOW, subtract the return
        // that is about to be evicted from the horizon.
        if self.len >= Self::VOLATILITY_WINDOW {
            let expired_idx =
                (self.next_index + Self::CAPACITY - Self::VOLATILITY_WINDOW) % Self::CAPACITY;
            let expired_next_idx = (expired_idx + 1) % Self::CAPACITY;
            let expired_return = self.ring[expired_next_idx].price - self.ring[expired_idx].price;
            self.return_sum -= expired_return;
            self.return_sq_sum -= expired_return.powi(2);
        }

        self.return_sum += current_return;
        self.return_sq_sum += current_return.powi(2);

        let snapshot_without_stats = TickSnapshot {
            epoch,
            price,
            direction,
            streak,
            volatility: 0.0,
            drift: 0.0,
            return_magnitude: current_return.abs(),
            ticks_since_reversal: self.ticks_since_reversal,
        };

        self.ring[self.next_index] = snapshot_without_stats;
        self.next_index = (self.next_index + 1) % Self::CAPACITY;
        self.len = self.len.saturating_add(1).min(Self::CAPACITY);
        self.last_price = Some(price);
        self.last_direction = direction;
        self.last_streak = streak;

        // Calculate volatility and drift over the available history
        let (volatility, drift) = self.calculate_stats();

        let mut snapshot = snapshot_without_stats;
        snapshot.volatility = volatility;
        snapshot.drift = drift;

        // Update the ring buffer entry with the calculated stats
        let last_index = (self.next_index + Self::CAPACITY - 1) % Self::CAPACITY;
        self.ring[last_index] = snapshot;

        snapshot
    }

    fn calculate_stats(&self) -> (f64, f64) {
        let count = self.len.min(Self::VOLATILITY_WINDOW);
        if count < 2 {
            return (0.0, 0.0);
        }

        let n = (count - 1) as f64;
        let mean = self.return_sum / n;
        let variance = (self.return_sq_sum / n) - mean.powi(2);
        let std_dev = variance.max(0.0).sqrt();

        (std_dev, mean)
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

        for index in 0..128 {
            processor.push(index, index as f64);
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
}
