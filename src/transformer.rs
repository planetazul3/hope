use anyhow::{anyhow, Result};
use std::path::Path;
use tract_onnx::prelude::*;

use crate::strategy::ProbabilityModel;
use crate::tick_processor::TickSnapshot;

pub struct TransformerModel {
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    sequence_length: usize,
}

impl TransformerModel {
    pub fn load(path: impl AsRef<Path>, sequence_length: usize) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact(&[1, sequence_length, 5]).into())?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self {
            model,
            sequence_length,
        })
    }

    pub fn predict(&self, history: &[TickSnapshot]) -> Result<f64> {
        if history.len() < self.sequence_length {
            return Err(anyhow!("insufficient history for transformer inference"));
        }

        let start_idx = history.len() - self.sequence_length;
        let sequence = &history[start_idx..];

        let mut data = Vec::with_capacity(self.sequence_length * 5);
        for tick in sequence {
            data.push(tick.direction.as_i8() as f32);
            data.push(tick.return_magnitude as f32);
            data.push(tick.streak as f32);
            data.push(tick.ticks_since_reversal as f32);
            data.push(tick.volatility as f32);
        }

        let input = tract_ndarray::Array3::from_shape_vec((1, self.sequence_length, 5), data)
            .map_err(|err| anyhow!("failed to build input tensor: {err}"))?;
        
        let result = self.model.run(tvec!(input.into_tensor().into()))?;
        let output = result[0].to_array_view::<f32>()?;
        
        // Assume output is a single float representing P(Up)
        Ok(output[0] as f64)
    }
}

// Since ProbabilityModel trait only takes a single TickSnapshot, we need to adapt it.
// Or we need to change the trait to take history.
// However, the current strategy boundary says "Strategy consumes normalized tick state".
// Let's stick to the trait and maybe the model itself tracks the history?
// Or we pass the processor to the evaluator?
// Actually, TickProcessor already has the history in its ring buffer.

impl ProbabilityModel for TransformerModel {
    fn probability_up(&self, _tick: &TickSnapshot, history: &[TickSnapshot]) -> f64 {
        self.predict(history).unwrap_or(0.5)
    }
}
