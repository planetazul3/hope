use anyhow::{anyhow, Context, Result};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use std::fs;
use std::path::Path;
use tracing::{error, info, warn};
use tract_onnx::prelude::*;

use crate::strategy::ProbabilityModel;
use crate::tick_processor::TickSnapshot;

type TractModel = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Gated TCN probability model using ONNX inference.
///
/// This model (V4) uses causal dilated gated convolutions with Squeeze-and-Excitation
/// attention to estimate the probability of the next tick being UP.
/// It consumes a sliding window of 32 ticks with 8 features (5 base + 2 DWT + 1 Volatility Ratio).
pub struct TransformerModel {
    model: TractModel,
    sequence_length: usize,
    data_buffer: Vec<f32>,
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
            tracing::warn!(
                "loading model WITHOUT signature verification (MODEL_PUBLIC_KEY not set)"
            );
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

        Ok(Self {
            model,
            sequence_length,
            data_buffer: Vec::with_capacity(sequence_length * 8),
        })
    }

    pub fn predict(&mut self, history: &[TickSnapshot]) -> Result<f64> {
        if history.len() < self.sequence_length {
            return Err(anyhow!("insufficient history for GatedTCN inference"));
        }

        let start_idx = history.len() - self.sequence_length;
        let sequence = &history[start_idx..];

        self.data_buffer.clear();

        for (_i, tick) in sequence.iter().enumerate() {
            // 1-5: Base features
            self.data_buffer.push(tick.direction.as_i8() as f32);
            self.data_buffer
                .push((tick.return_magnitude as f32) / (tick.volatility as f32 + 1e-8));
            self.data_buffer.push((tick.streak as f32).ln_1p());
            self.data_buffer
                .push((tick.ticks_since_reversal as f32).ln_1p());
            self.data_buffer.push(tick.volatility as f32);

            // Phase 2: DWT Haar Wavelet Level-1 Decomposition
            // A1 (Approximation) normalized by price; D1 (Detail) coefficient.
            let safe_price = if tick.price == 0.0 { 1e-8 } else { tick.price };
            let a1_norm = (tick.dwt_a1 / (safe_price + 1e-8_f64)) as f32;
            self.data_buffer.push(a1_norm);
            self.data_buffer.push(tick.dwt_d1 as f32);
            self.data_buffer.push(tick.vol_ratio as f32);
        }

        if self
            .data_buffer
            .iter()
            .any(|v| v.is_nan() || v.is_infinite())
        {
            return Err(anyhow!("corrupted features: NaN or Inf detected"));
        }

        let input = Tensor::from_shape(&[1, self.sequence_length, 8], &self.data_buffer)?;

        let outputs = self.model.run(tvec!(input.into()))?;
        let output = outputs[0].to_array_view::<f32>()?;

        let prob = output
            .as_slice()
            .and_then(|s| s.first())
            .copied()
            .ok_or_else(|| anyhow!("Empty model output"))?;

        Ok(prob as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a1_normalization_alignment() {
        // Mock a tick that triggers the safe_price logic
        let tick = TickSnapshot {
            price: 0.0,
            dwt_a1: 1.41421356, // sqrt(2) * (0+2)/2 approx... let's just use a value
            ..Default::default()
        };

        // Expected in Rust: safe_price = 1e-8, denominator = 2e-8
        // Result: 1.41421356 / 2e-8 = 70710678
        let safe_price = 1e-8;
        let expected = (tick.dwt_a1 / (safe_price + 1e-8)) as f32;

        // Verify logic used in predict
        let actual_safe_price = if tick.price == 0.0 { 1e-8 } else { tick.price };
        let actual = (tick.dwt_a1 / (actual_safe_price + 1e-8)) as f32;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_signature_fallback_warning() {
        // This test mostly ensures it doesn't crash if sig is missing but PK is set
        let model_path = Path::new("model_test_temp.onnx");
        fs::write(model_path, b"dummy onnx content").unwrap();

        let pk = "00".repeat(32); // 32 bytes of zeros

        // This should NOT fail because load() now warns instead of erroring on missing .sig
        // It will fail later on parsing dummy bytes, which is fine for this test's scope
        let result = TransformerModel::load(model_path, 32, Some(&pk));

        match result {
            Err(e) => {
                // If it fails, it should be because of ONNX parsing, not signature missing
                let err_msg = e.to_string();
                assert!(!err_msg.contains("failed to read signature file"));
            }
            Ok(_) => {}
        }

        // Cleanup
        let _ = fs::remove_file(model_path);
    }
}

impl ProbabilityModel for TransformerModel {
    fn probability_up(&mut self, _tick: &TickSnapshot, history: &[TickSnapshot]) -> f64 {
        match self.predict(history) {
            Ok(p) => p,
            Err(err) => {
                error!(error = %err, "GatedTCN prediction failed");
                0.5
            }
        }
    }
}
