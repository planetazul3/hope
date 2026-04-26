use crossbeam_channel::{bounded, Sender, TrySendError};
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    thread,
};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use tracing::{error, info, warn};

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

pub struct TickLogger {
    tx: Option<Sender<TickLogRecord>>,
    _handle: Option<thread::JoinHandle<()>>,
}

impl Clone for TickLogger {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            _handle: None, // Only the original/main instance needs the handle
        }
    }
}

impl TickLogger {
    pub fn start(path: &str, capacity: usize) -> Self {
        let (tx, rx) = bounded::<TickLogRecord>(capacity);
        let path = path.to_string();

        let _handle = thread::spawn(move || {
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
                let mut write_record = |rec: &TickLogRecord| {
                    let line = format!(
                        "{},{:.5},{},{},{:.4},{:?},\"{}\",{}\n",
                        rec.timestamp,
                        rec.price,
                        rec.direction,
                        rec.streak,
                        rec.probability,
                        rec.state,
                        rec.decision,
                        rec.latency_ms
                    );
                    writer.write_all(line.as_bytes())
                };

                if write_record(&record).is_err() {
                    break;
                }

                // Drain remaining without blocking
                for extra_record in rx.try_iter() {
                    if write_record(&extra_record).is_err() {
                        break;
                    }
                }
                let _ = writer.flush();
            }
            // Final flush on shutdown
            let _ = writer.flush();
        });

        Self {
            tx: Some(tx),
            _handle: Some(_handle),
        }
    }

    /// Gracefully stops the logger and ensures all pending records are flushed to disk.
    pub fn stop(&mut self) {
        // Drop the sender to signal the background thread to exit
        self.tx.take();
        if let Some(handle) = self._handle.take() {
            info!("waiting for tick logger thread to join...");
            let _ = handle.join();
        }
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

        if let Some(ref tx) = self.tx {
            if let Err(err) = tx.try_send(record) {
                match err {
                    TrySendError::Full(_) => {
                        warn!("tick log channel full; dropping tick audit line")
                    }
                    TrySendError::Disconnected(_) => {
                        warn!("tick log channel disconnected; dropping tick audit line")
                    }
                }
            }
        }
    }
}
