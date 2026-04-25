# ADR 0014 — Cloud-Only Training Enforcement

**Status**: Accepted
**Date**: 2026-04-25

## Context

Running model training on local hardware produces lower-quality models due to insufficient GPU memory and training time. Local machines typically lack the CUDA compute required to complete a full two-phase curriculum (contrastive pre-training + supervised fine-tuning) in a reasonable time, and memory constraints prevent batch sizes large enough for stable InfoNCE contrastive loss. Additionally, an unconstrained local training path increases the risk of accidental deployments of under-trained models.

Project policy mandates cloud GPU training (Google Colab T4 or Kaggle P100/T4) to ensure consistent model quality and reproducible training conditions.

## Decision

1. All model training must be performed exclusively in Google Colab or Kaggle by uploading `data/ticks.csv` and executing `notebooks/train_transformer.ipynb`.
2. All training entry points (`scripts/train_fixed.py`) must include a runtime environment check at the very beginning of `main()` that inspects the `COLAB_GPU` and `KAGGLE_URL_BASE` environment variables. If neither is present, the script must print a descriptive error message and call `sys.exit(1)`.
3. The notebook's Environment Detection cell (Cell 1) must raise a `RuntimeError` in the `else` branch if neither a Colab nor a Kaggle environment is detected, preventing any subsequent cell from running.
4. The `make train` Makefile target must not invoke any Python training script locally. It must print instructions directing the operator to a cloud environment.
5. This policy is codified in `AGENTS.md`, `GEMINI.md`, and `.github/copilot-instructions.md` so all AI coding agents enforce it.

## Consequences

- Consistent model quality is enforced by hardware access requirements.
- Clear error messages prevent accidental local training runs.
- The policy is codified in the ADR alongside the three existing agent instruction files, making it auditable and durable.
- Operators must have access to Google Colab or Kaggle to train new models, which is a deliberate constraint.
