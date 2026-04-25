use anyhow::{anyhow, Context, Result};
use std::cell::RefCell;
use std::path::Path;
use tract_onnx::prelude::*;

use crate::strategy::ProbabilityModel;
use crate::tick_processor::TickSnapshot;

type TractModel = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct TransformerModel {
    model: TractModel,
    sequence_length: usize,
    features_buffer: RefCell<Vec<f32>>,
}

impl TransformerModel {
    pub fn load(path: impl AsRef<Path>, sequence_length: usize) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path.as_ref())
            .with_context(|| format!("failed to read ONNX model from {}", path.as_ref().display()))?
            .with_input_fact(0, f32::fact([1, sequence_length, 5]).into())
            .context("failed to set model input facts")?
            .into_optimized()
            .context("failed to optimize model graph")?
            .into_runnable()
            .context("failed to convert model to runnable form")?;

        Ok(Self {
            model,
            sequence_length,
            features_buffer: RefCell::new(Vec::with_capacity(sequence_length * 5)),
        })
    }

    pub fn predict(&self, history: &[TickSnapshot]) -> Result<f64> {
        if history.len() < self.sequence_length {
            return Err(anyhow!("insufficient history for transformer inference"));
        }

        let start_idx = history.len() - self.sequence_length;
        let sequence = &history[start_idx..];

        let mut data = self.features_buffer.borrow_mut();
        data.clear();
        for tick in sequence {
            data.push(tick.direction.as_i8() as f32);
            data.push(tick.return_magnitude as f32);
            data.push(tick.streak as f32);
            data.push(tick.ticks_since_reversal as f32);
            data.push(tick.volatility as f32);
        }

        let input = Tensor::from_shape(&[1, self.sequence_length, 5], &data)?;

        let outputs = self.model.run(tvec!(input.into()))?;
        let output = outputs[0].to_array_view::<f32>()?;

        // Safe indexing to avoid rank mismatch panics
        let prob = output
            .as_slice()
            .and_then(|s| s.first())
            .copied()
            .ok_or_else(|| anyhow!("Empty model output"))?;

        Ok(prob as f64)
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
