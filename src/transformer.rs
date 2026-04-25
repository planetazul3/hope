use anyhow::{anyhow, Context, Result};
use std::path::Path;
use tracing::error;
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
}

impl TransformerModel {
    pub fn load(path: impl AsRef<Path>, sequence_length: usize) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path.as_ref())
            .with_context(|| format!("failed to read ONNX model from {}", path.as_ref().display()))?
            .with_input_fact(0, f32::fact([1, sequence_length, 8]).into())
            .context("failed to set model input facts")?
            .into_optimized()
            .context("failed to optimize model graph")?
            .into_runnable()
            .context("failed to convert model to runnable form")?;

        Ok(Self {
            model,
            sequence_length,
        })
    }

    pub fn predict(&self, history: &[TickSnapshot]) -> Result<f64> {
        if history.len() < self.sequence_length {
            return Err(anyhow!("insufficient history for GatedTCN inference"));
        }

        let start_idx = history.len() - self.sequence_length;
        let sequence = &history[start_idx..];

        let mut data = Vec::with_capacity(self.sequence_length * 8);

        for (i, tick) in sequence.iter().enumerate() {
            // 1-5: Base features
            data.push(tick.direction.as_i8() as f32);
            data.push((tick.return_magnitude as f32) / (tick.volatility as f32 + 1e-8));
            data.push((tick.streak as f32).ln_1p());
            data.push((tick.ticks_since_reversal as f32).ln_1p());
            data.push(tick.volatility as f32);

            // 6: Haar A1 approximation normalized by price
            // 7: Haar D1 detail coefficient
            // 8: Short-term to long-term volatility ratio
            let global_idx = start_idx + i;

            // Phase 2: DWT Haar Wavelet Level-1 Decomposition
            // A1 (Approximation) = (x_t + x_t-1) / sqrt(2)
            // D1 (Detail) = (x_t - x_t-1) / sqrt(2)
            if global_idx >= 1 {
                // The first tick in the history cannot produce DWT coefficients and receives 0 padding for DWT
                let x_t = history[global_idx].price;
                let x_prev = history[global_idx - 1].price;

                let a1 = (x_t + x_prev) / 2.0_f64.sqrt();
                let d1 = (x_t - x_prev) / 2.0_f64.sqrt();

                // Normalizing A1 by price level to keep it scale-invariant
                data.push((a1 / x_t) as f32);
                data.push(d1 as f32);
            } else {
                data.push(0.0);
                data.push(0.0);
            }
            data.push(tick.vol_ratio as f32);
        }

        let input = Tensor::from_shape(&[1, self.sequence_length, 8], &data)?;

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
    fn probability_up(&self, _tick: &TickSnapshot, history: &[TickSnapshot]) -> f64 {
        match self.predict(history) {
            Ok(p) => p,
            Err(err) => {
                error!(error = %err, "GatedTCN prediction failed");
                0.5
            }
        }
    }
}
