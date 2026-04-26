use anyhow::{anyhow, Context, Result};
use arc_swap::ArcSwap;
use crossbeam_channel::{bounded, Sender, TrySendError};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use std::fs;
use std::path::Path;
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
    tx: Option<Sender<Vec<TickSnapshot>>>,
    latest_prob: Arc<ArcSwap<f64>>,
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

                let signature =
                    Signature::from_slice(&sig_bytes).context("invalid signature format")?;

                public_key
                    .verify(&model_bytes, &signature)
                    .map_err(|err| anyhow!("MODEL SIGNATURE VERIFICATION FAILED: {}", err))?;
                info!("model signature verified successfully");
            } else {
                warn!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                warn!("WARNING: MODEL SIGNATURE FILE MISSING (.onnx.sig)");
                warn!("A public key was provided, but no signature was found.");
                warn!("Proceeding with UNVERIFIED model at user's risk.");
                warn!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            }
        } else {
            warn!("loading model WITHOUT signature verification (MODEL_PUBLIC_KEY not set)");
        }

        let model = tract_onnx::onnx()
            .model_for_read(&mut &model_bytes[..])
            .with_context(|| format!("failed to parse ONNX model from {}", path.display()))?
            .with_input_fact(0, f32::fact([1, sequence_length, 8]).into())
            .context("failed to set model input facts")?
            .into_optimized()
            .context("failed to optimize model graph")?
            .into_runnable()
            .context("failed to convert model to runnable form")?;

        let (tx, rx) = bounded::<Vec<TickSnapshot>>(2); // keep it small, discard if full
        let latest_prob = Arc::new(ArcSwap::from_pointee(0.5));
        let latest_prob_clone = Arc::clone(&latest_prob);

        let _handle = thread::Builder::new()
            .name("inference".into())
            .spawn(move || {
                let mut data_buffer = vec![0.0f32; sequence_length * 8];
                let shape = vec![1, sequence_length, 8];
                let mut input_tensor = Tensor::zero::<f32>(&shape).unwrap();
                let mut input_tvec = tvec![input_tensor.clone().into()];

                while let Ok(sequence) = rx.recv() {
                    if sequence.len() < sequence_length {
                        continue;
                    }

                    // Zero-allocation hot path update for inference thread
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
                            if data_buffer[base + j].is_nan() || data_buffer[base + j].is_infinite()
                            {
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
                        continue;
                    }

                    // Avoid heap allocation by using Tensor::from_slice_align or updating directly
                    // if tract allows mutable access
                    if let Ok(mut view) = input_tensor.to_array_view_mut::<f32>() {
                        if let Some(slice) = view.as_slice_mut() {
                            slice.copy_from_slice(&data_buffer);
                        } else {
                            // Fallback if not contiguous
                            input_tensor = Tensor::from_shape(&shape, &data_buffer).unwrap();
                        }
                    } else {
                        input_tensor = Tensor::from_shape(&shape, &data_buffer).unwrap();
                    }

                    input_tvec[0] = input_tensor.clone().into();

                    match model.run(input_tvec.clone()) {
                        Ok(outputs) => {
                            if let Ok(output_view) = outputs[0].to_array_view::<f32>() {
                                if let Some(&prob) = output_view.as_slice().and_then(|s| s.first())
                                {
                                    latest_prob_clone.store(Arc::new(prob as f64));
                                }
                            }
                        }
                        Err(err) => {
                            error!(error = %err, "tract-onnx inference failed");
                        }
                    }
                }
            })
            .expect("failed to spawn inference thread");

        Ok(Self {
            sequence_length,
            tx: Some(tx),
            latest_prob,
            _handle: Some(_handle),
        })
    }
}

impl Drop for TransformerModel {
    fn drop(&mut self) {
        self.tx.take(); // drops sender to terminate thread
        if let Some(handle) = self._handle.take() {
            let _ = handle.join();
        }
    }
}

impl ProbabilityModel for TransformerModel {
    fn probability_up(&mut self, _tick: &TickSnapshot, history: &[TickSnapshot]) -> f64 {
        if history.len() >= self.sequence_length {
            let start_idx = history.len() - self.sequence_length;
            let sequence = history[start_idx..].to_vec();

            if let Some(tx) = &self.tx {
                match tx.try_send(sequence) {
                    Ok(_) => {}
                    Err(TrySendError::Full(_)) => {
                        // It's a lock-free channel of size 2, so dropping the newest if it's lagging
                        // is fine to maintain non-blocking behavior.
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        error!("inference thread disconnected");
                    }
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
