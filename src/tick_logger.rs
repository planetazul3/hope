use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    sync::mpsc::{self, SyncSender, TrySendError},
    thread,
};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use tracing::{error, warn};

use crate::{fsm::TradingState, tick_processor::TickSnapshot};

#[derive(Debug, Clone)]
pub struct TickLogRecord {
    pub timestamp: u64,
    pub price: f64,
    pub direction: i8,
    pub streak: u32,
    pub probability: f64,
    pub decision: &'static str,
    pub state: TradingState,
    pub latency_ms: u128,
}

#[derive(Clone)]
pub struct TickLogger {
    tx: SyncSender<TickLogRecord>,
}

impl TickLogger {
    pub fn start(path: &str, capacity: usize) -> Self {
        let (tx, rx) = mpsc::sync_channel::<TickLogRecord>(capacity);
        let path = path.to_string();

        thread::spawn(move || {
            let mut options = OpenOptions::new();
            options.create(true).append(true);
            #[cfg(unix)]
            options.mode(0o600);

            let file = match options.open(&path) {
                Ok(file) => file,
                Err(err) => {
                    error!(path = %path, error = %err, "failed to open tick log file");
                    return;
                }
            };
            let mut writer = BufWriter::new(file);

            while let Ok(record) = rx.recv() {
                let line = format!(
                    "{},{:.5},{},{},{:.4},{:?},\"{}\",{}\n",
                    record.timestamp,
                    record.price,
                    record.direction,
                    record.streak,
                    record.probability,
                    record.state,
                    record.decision,
                    record.latency_ms
                );
                if writer.write_all(line.as_bytes()).is_err() {
                    break;
                }
                // Smart Flush: If the channel is empty, flush the buffer to disk.
                if rx.try_recv().is_err() {
                    let _ = writer.flush();
                }
            }
        });

        Self { tx }
    }

    pub fn try_log(
        &self,
        tick: TickSnapshot,
        probability: f64,
        decision: &'static str,
        state: TradingState,
        latency_ms: u128,
    ) {
        let record = TickLogRecord {
            timestamp: tick.epoch,
            price: tick.price,
            direction: tick.direction.as_i8(),
            streak: tick.streak,
            probability,
            decision,
            state,
            latency_ms,
        };

        if let Err(TrySendError::Full(_)) = self.tx.try_send(record) {
            warn!("tick log channel full; dropping tick audit line");
        }
    }
}
