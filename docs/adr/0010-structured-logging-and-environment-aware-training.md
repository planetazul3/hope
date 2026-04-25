# 0010: Structured Logging and Environment-Aware Training

*   **Status**: Accepted
*   **Date**: 2026-04-24

## Context

As the machine learning training pipeline grew in complexity, observability and portability became critical bottlenecks. Training runs lacked structured logs for retrospective analysis, and the training notebook required manual adjustments when switching between local execution, Google Colab, and Kaggle. Additionally, the model architecture lacked flexibility in how it aggregated sequence information.

## Decision

We have implemented a suite of observability and portability enhancements:

1.  **Structured Logging**: Replaced transient `print` statements with a dual-target `logging` system. Logs are now persisted to `training.log` with timestamps and severity levels, while maintaining real-time console output.
2.  **Environment-Aware Logic**: Introduced automatic environment detection (Colab/Kaggle/Local) in the training notebook. This allows for zero-config data path resolution and platform-specific artifacts handling (e.g., automatic ONNX downloads in Colab).
3.  **Configurable Model Pooling**: Updated the `SimpleTransformer` architecture to support both `mean` pooling (averaging sequence hidden states) and `last` pooling (using the final hidden state). This is toggleable via the `pooling` parameter.
4.  **Hardware Auditing**: The training entrypoint now explicitly logs PyTorch versions, CUDA availability, and the specific GPU device name to ensure hardware acceleration is correctly utilized.
5.  **Interactive Progress Visualization**: Integrated `tqdm` progress bars for both training and validation loops, providing real-time feedback on batch loss and estimated completion time.
6.  **ONNX Integrity Validation**: Added a mandatory post-export validation step using the `onnx` library to verify that the generated `model.onnx` file is valid and compliant with the ONNX specification before deployment.
7.  **Classification [CLS] Token**: Adopted the standard Transformer classification pattern by prepending a learned `[CLS]` token to the input sequence. This token serves as the primary aggregator for classification features, replacing simple sequence averaging.
8.  **Explicit Weight Initialization**: Implemented Kaiming Normal initialization for linear layers and controlled standard deviation for the `[CLS]` token to ensure stable gradient flow from the start of training.
9.  **Z-Score Feature Standardization**: Transitioned to `StandardScaler` for all input features to ensure zero mean and unit variance, optimizing the sensitivity of the dot-product attention mechanism.
10. **Learning Rate Warmup**: Integrated a 5-epoch linear warmup schedule to prevent early divergence and allow attention layers to stabilize before reaching the base learning rate.

## Consequences

*   **Convergence Stability**: The combination of warmup, explicit initialization, and standardization significantly reduces the risk of the model collapsing into predicting a single class (the "majority class" problem).
*   **Feature Signal**: The `[CLS]` token improves the model's ability to focus on critical temporal patterns across the entire sequence.
*   **Traceability**: Every training session now produces a durable audit trail in `training.log`.
*   **Efficiency**: Researchers can move between cloud environments (Colab/Kaggle) and local workstations without modifying code or manually searching for data files.
*   **Flexibility**: The model can be easily adapted to different time-series aggregation strategies by simply changing a configuration parameter.
*   **Reliability**: The ONNX validation step prevents corrupt or incompatible models from entering the production deployment pipeline.
