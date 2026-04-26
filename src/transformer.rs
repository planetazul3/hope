use anyhow::{anyhow, Context, Result};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use std::fs;
use std::path::Path;
use tracing::{error, info};
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
            info!(path = %path.display(), "verifying model signature");
            let sig_path = path.with_extension("onnx.sig");
            let sig_bytes = fs::read(&sig_path).with_context(|| {
                format!("failed to read signature file: {}", sig_path.display())
            })?;

            let pk_bytes = hex::decode(pk_hex).context("invalid MODEL_PUBLIC_KEY hex format")?;
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

        for (i, tick) in sequence.iter().enumerate() {
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
            let a1_norm = if tick.price == 0.0 {
                0.0
            } else {
                (tick.dwt_a1 / tick.price) as f32
            };
            self.data_buffer.push(a1_norm);
            self.data_buffer.push(tick.dwt_d1 as f32);
            self.data_buffer.push(tick.vol_ratio as f32);
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
