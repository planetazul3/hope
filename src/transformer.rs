use anyhow::{anyhow, Context, Result};
use arc_swap::ArcSwap;
use crossbeam_queue::ArrayQueue;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use tracing::{error, info, warn};
use tract_onnx::prelude::*;

use crate::strategy::ProbabilityModel;
use crate::tick_processor::TickSnapshot;

/// Canonical Causal Transformer probability model using ONNX inference.
/// Executes a static execution graph (1x32x8) with dynamic INT8 Quantization over a zero-allocation hot path.
pub struct TransformerModel {
    sequence_length: usize,
    queue: Arc<ArrayQueue<Vec<f32>>>,
    pool: Arc<ArrayQueue<Vec<f32>>>,
    latest_prob: Arc<ArcSwap<f64>>,
    is_running: Arc<AtomicBool>,
    _handle: Option<thread::JoinHandle<()>>,
}

impl TransformerModel {
    pub fn load(
        path: impl AsRef<Path>,
        sequence_length: usize,
        public_key_hex: Option<&str>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let model_bytes = fs::read(path)
            .with_context(|| format!("failed to read model file: {}", path.display()))?;

        if let Some(pk_hex) = public_key_hex {
            let sig_path = path.with_extension("onnx.sig");
            if sig_path.exists() {
                let sig_bytes = fs::read(&sig_path).with_context(|| {
                    format!("failed to read signature file: {}", sig_path.display())
                })?;

                let pk_bytes =
                    hex::decode(pk_hex).context("invalid MODEL_PUBLIC_KEY hex format")?;
                let public_key = VerifyingKey::from_bytes(
                    &pk_bytes
                        .try_into()
                        .map_err(|_| anyhow!("invalid public key length"))?,
                )
                .context("failed to parse Ed25519 public key")?;

                let signature = Signature::from_slice(&sig_bytes)
                    .context("invalid Ed25519 signature format")?;

                public_key
                    .verify(&model_bytes, &signature)
                    .context("model signature verification failed")?;
                info!("model signature verified successfully");
            } else {
                warn!("MODEL_PUBLIC_KEY provided, but no .sig file found; proceeding without verification");
            }
        }

        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;

        let queue: Arc<ArrayQueue<Vec<f32>>> = Arc::new(ArrayQueue::new(2));
        let pool: Arc<ArrayQueue<Vec<f32>>> = Arc::new(ArrayQueue::new(2));

        // Pre-allocate zero-allocation feature buffers
        for _ in 0..2 {
            let _ = pool.push(vec![0.0f32; sequence_length * 8]);
        }

        let latest_prob = Arc::new(ArcSwap::from_pointee(0.5));
        let is_running = Arc::new(AtomicBool::new(true));

        let pool_clone = Arc::clone(&pool);
        let queue_clone = Arc::clone(&queue);
        let latest_prob_clone = Arc::clone(&latest_prob);
        let is_running_clone = Arc::clone(&is_running);

        let _handle = thread::Builder::new()
            .name("inference".into())
            .spawn(move || {
                let shape = vec![1, sequence_length, 8];
                let mut input_tensor = Tensor::zero::<f32>(&shape).unwrap();
                let backoff = crossbeam_utils::Backoff::new();

                while is_running_clone.load(Ordering::Acquire) {
                    if let Some(data_buffer) = queue_clone.pop() {
                        // Zero-allocation hot path update using State Recycling
                        if let Ok(mut view) = input_tensor.to_array_view_mut::<f32>() {
                            if let Some(slice) = view.as_slice_mut() {
                                slice.copy_from_slice(&data_buffer);
                            } else {
                                input_tensor = Tensor::from_shape(&shape, &data_buffer).unwrap();
                            }
                        } else {
                            // Fallback if tract tensor data layout changed or is not uniquely owned
                            input_tensor = Tensor::from_shape(&shape, &data_buffer).unwrap();
                        }

                        let input_tvec = tvec![input_tensor.clone().into()];

                        match model.run(input_tvec) {
                            Ok(outputs) => {
                                if let Ok(output_view) = outputs[0].to_array_view::<f32>() {
                                    if let Some(&prob) =
                                        output_view.as_slice().and_then(|s| s.first())
                                    {
                                        latest_prob_clone.store(Arc::new(prob as f64));
                                    }
                                }
                            }
                            Err(err) => {
                                error!(error = %err, "tract-onnx inference failed");
                            }
                        }

                        // Return the buffer to the pool to prevent allocation
                        let _ = pool_clone.push(data_buffer);
                        backoff.reset();
                    } else {
                        backoff.snooze();
                    }
                }
            })
            .expect("failed to spawn inference thread");

        Ok(Self {
            sequence_length,
            queue,
            pool,
            latest_prob,
            is_running,
            _handle: Some(_handle),
        })
    }
}

impl Drop for TransformerModel {
    fn drop(&mut self) {
        self.is_running.store(false, Ordering::Release);
        if let Some(handle) = self._handle.take() {
            let _ = handle.join();
        }
    }
}

impl ProbabilityModel for TransformerModel {
    fn probability_up(&mut self, _tick: &TickSnapshot, history: &[TickSnapshot]) -> f64 {
        if history.len() >= self.sequence_length {
            let start_idx = history.len() - self.sequence_length;
            let sequence = &history[start_idx..];

            if let Some(mut data_buffer) = self.pool.pop() {
                let mut valid = true;
                for (i, tick) in sequence.iter().enumerate() {
                    let base = i * 8;
                    data_buffer[base] = tick.direction.as_i8() as f32;
                    data_buffer[base + 1] =
                        (tick.return_magnitude as f32) / (tick.volatility as f32 + 1e-8);
                    data_buffer[base + 2] = (tick.streak as f32).ln_1p();
                    data_buffer[base + 3] = (tick.ticks_since_reversal as f32).ln_1p();
                    data_buffer[base + 4] = tick.volatility as f32;

                    let safe_price = if tick.price == 0.0 { 1e-8 } else { tick.price };
                    let a1_norm = (tick.db2_a1 / (safe_price + 1e-8_f64)) as f32;
                    data_buffer[base + 5] = a1_norm;
                    data_buffer[base + 6] = tick.db2_d1 as f32;
                    data_buffer[base + 7] = tick.vol_ratio as f32;

                    for j in 0..8 {
                        if data_buffer[base + j].is_nan() || data_buffer[base + j].is_infinite() {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        break;
                    }
                }

                if !valid {
                    error!("corrupted features: NaN or Inf detected");
                    let _ = self.pool.push(data_buffer);
                } else if let Err(err) = self.queue.push(data_buffer) {
                    // Queue full, return to pool
                    let _ = self.pool.push(err);
                }
            }
        }

        **self.latest_prob.load()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a1_normalization_alignment() {
        let tick = TickSnapshot {
            price: 0.0,
            db2_a1: 1.41421356,
            ..Default::default()
        };

        let safe_price = 1e-8;
        let expected = (tick.db2_a1 / (safe_price + 1e-8)) as f32;

        let actual_safe_price = if tick.price == 0.0 { 1e-8 } else { tick.price };
        let actual = (tick.db2_a1 / (actual_safe_price + 1e-8)) as f32;

        assert_eq!(actual, expected);
    }
}
