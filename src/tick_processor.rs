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
}

#[derive(Debug)]
pub struct TickProcessor {
    ring: [TickSnapshot; Self::CAPACITY],
    next_index: usize,
    len: usize,
    last_price: Option<f64>,
    last_direction: Direction,
    last_streak: u32,
}

impl TickProcessor {
    pub const CAPACITY: usize = 64;

    pub fn new() -> Self {
        Self {
            ring: [TickSnapshot::default(); Self::CAPACITY],
            next_index: 0,
            len: 0,
            last_price: None,
            last_direction: Direction::Flat,
            last_streak: 0,
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

        let snapshot = TickSnapshot {
            epoch,
            price,
            direction,
            streak,
        };

        self.ring[self.next_index] = snapshot;
        self.next_index = (self.next_index + 1) % Self::CAPACITY;
        self.len = self.len.saturating_add(1).min(Self::CAPACITY);
        self.last_price = Some(price);
        self.last_direction = direction;
        self.last_streak = streak;

        snapshot
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.len
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
}
