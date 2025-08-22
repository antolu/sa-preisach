# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SA-Preisach implements Self-Adaptive Differentiable Preisach models using PyTorch Lightning for modeling magnetic hysteresis in particle accelerator applications at CERN. This package extends the Differentiable Preisach model by R. Roussel et al. with neural network enhancements for improved adaptability.

## Common Commands

### Development Environment
```bash
# Install package in development mode
pip install -e . --config-settings editable_mode=compat

# Install with development dependencies  
pip install -e ".[dev,test]"
```

### Code Quality
```bash
# Run linting with auto-fixes
ruff check --fix --unsafe-fixes --preview .
ruff format .

# Type checking
mypy sa_preisach/

# Run pre-commit hooks
pre-commit run --all-files
```

### Testing
```bash
# Run tests (basic placeholder test exists)
pytest

# Run specific test
pytest tests/test_placeholder.py

# Run with coverage
pytest --cov=sa_preisach
```

### Model Training
```bash
# Train Self-Adaptive Preisach model
cd sample
sa_preisach fit -c config.yml

# Train original Differentiable Preisach model  
sa_preisach fit -c config_diff_preisach.yml

# Train neural network enhanced model
sa_preisach fit -c config_diff_preisach_nn.yml

# Verbose training
sa_preisach fit -vv -c config.yml

# With experiment name
sa_preisach fit -c config.yml --experiment-name my_experiment
```

### Model Prediction/Inference
```bash
# Run prediction
sa_preisach predict --config config.yml --ckpt_path path/to/checkpoint.ckpt
```

## Architecture Overview

### Core Components

1. **Models (`sa_preisach.models/`)**
   - `BaseModule`: Lightning base class with model compilation and validation tracking
   - `SelfAdaptivePreisach`: Novel self-adaptive variant with neural network components
   - `DifferentiablePreisach`: Original implementation by Roussel et al.
   - `DifferentiablePreisachNN`: Neural network enhanced version with ResNet MLPs

2. **Data Pipeline (`sa_preisach.data/`)**
   - `PreisachDataModule`: Lightning data module for hysteresis data loading
   - Supports Parquet file inputs with downsampling
   - Integrates with TransformerTF transform system for preprocessing

3. **Neural Network Components (`sa_preisach.nn/`)**
   - `ResNetMLP`: Residual network MLP for function approximation
   - `BinaryParameter`: Binary-constrained parameters for physical modeling
   - `ConstrainedParameter`: General parameter constraints
   - `GPyConstrainedParameter`: GPyTorch-based parameter constraints

4. **Utilities (`sa_preisach.utils/`)**
   - `create_triangle_mesh`: Triangle mesh generation for Preisach plane
   - `get_states`/`set_states`: Magnetic state management
   - `make_mesh_size_function`: Mesh density function generation
   - Gradient manipulation utilities

5. **Callbacks (`sa_preisach.callbacks/`)**
   - `PlotHysteresisCallback`: Plots hysteresis loops during training

### Key Design Patterns

- **Lightning CLI Integration**: Uses PyTorch Lightning CLI with OmegaConf for YAML configuration
- **Model Compilation**: Supports `torch.compile` for performance optimization
- **Magnetic State Tracking**: Maintains magnetic states across batch sequences
- **Constrained Parameters**: Physics-informed parameter constraints
- **Mesh-based Modeling**: Triangle mesh representation of Preisach plane

### Model Hierarchy

- `BaseModule` (Lightning base)
  - `SelfAdaptivePreisach`: Adaptive variant with learnable mesh density
  - `DifferentiablePreisach`: Original formulation
  - `DifferentiablePreisachNN`: Enhanced with neural networks

### Configuration System

Uses Lightning CLI with hyperparameter linking:
- `data.n_train_samples` → `model.init_args.n_train_samples`
- Sample configurations in `sample/` directory demonstrate different model variants
- Supports TensorBoard and Neptune logging

### Data Flow

1. Magnetic field/current data loaded from Parquet files
2. Optional downsampling and transform preprocessing
3. Triangle mesh generation for Preisach plane discretization
4. Model forward pass with magnetic state propagation
5. Loss computation using masked MSE for variable-length sequences
6. Lightning handles training loop with automatic checkpointing

## Dependencies

- PyTorch >= 2.5.1 for core tensor operations and compilation
- Lightning >= 2.2.2 for training framework
- TransformerTF >= 0.13 for data pipeline and transforms
- NumPy for mesh generation and scientific computing
- Pandas for data loading and manipulation
- GPyTorch for parameter constraints
- Matplotlib for plotting callbacks

## Development Notes

- Pre-commit hooks enforce code quality with ruff, mypy
- Single-batch training/validation requirement enforced in BaseModule
- Model compilation supported via `compile_model` hyperparameter
- Checkpoints saved with version directories for experiment tracking
- Neptune logger integration for experiment management