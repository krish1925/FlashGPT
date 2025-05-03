# FlashGPT Advanced Modules

This directory contains the modular implementation of FlashGPT with advanced features and optimizations.

## Directory Structure

```
.
├── modules/                    # Core modules
│   ├── attention.py           # Attention mechanisms
│   ├── config.py              # Configuration classes
│   ├── datasets.py            # Dataset handling
│   ├── embeddings.py          # Embedding implementations
│   ├── models.py              # Model architectures
│   ├── optimization.py        # Training optimizations
│   ├── tools.py              # Utility tools
│   ├── training.py           # Training loops
│   └── utils.py              # Helper functions
├── checkpoints/               # Model checkpoints
├── datasets/                  # Training data
├── logs/                      # Training logs
├── cache/                     # Cached computations
├── model_diagnostics.py       # Debugging utilities
├── train_flashgpt.py         # Training script
├── infer_flashgpt.py         # Inference script
└── FlashGPT-Advanced-Thinking.py  # Advanced reasoning implementation
```

## Module Descriptions

### Core Modules

1. **attention.py** (11KB)
   - Implementation of various attention mechanisms
   - ALiBi positional bias
   - Grouped Query Attention
   - Flash Attention support
   - Multi-head attention with optimizations

2. **config.py** (1KB)
   - Configuration classes
   - Model hyperparameters
   - Training settings
   - Hardware-specific configs

3. **datasets.py** (4KB)
   - Dataset processing
   - Data loading utilities
   - Tokenization handling
   - Batch preparation

4. **embeddings.py** (3KB)
   - Token embeddings
   - Positional embeddings
   - RoPE implementation
   - Embedding dropout

5. **models.py** (8KB)
   - Model architecture definitions
   - Transformer blocks
   - Layer implementations
   - Model initialization

6. **optimization.py** (3KB)
   - Training optimizations
   - Learning rate scheduling
   - Gradient handling
   - Memory optimizations

7. **tools.py** (6KB)
   - Utility functions
   - Calculator implementation
   - Debugging tools
   - Metrics calculation

8. **training.py** (17KB)
   - Training loop implementation
   - Loss computation
   - Gradient updates
   - Checkpoint management

9. **utils.py** (7KB)
   - Helper functions
   - Logging utilities
   - Data processing
   - Model utilities

### Scripts

1. **model_diagnostics.py** (13KB)
   - Model debugging tools
   - Performance profiling
   - Memory usage tracking
   - Training diagnostics

2. **train_flashgpt.py** (9KB)
   - Main training script
   - Command-line interface
   - Training configuration
   - Experiment management

3. **infer_flashgpt.py** (5KB)
   - Inference utilities
   - Text generation
   - Model evaluation
   - Batch inference

4. **FlashGPT-Advanced-Thinking.py** (3KB)
   - Advanced reasoning implementation
   - Tree of Thought processing
   - Multi-step reasoning
   - Tool integration

## Usage

### Training

```bash
python train_flashgpt.py \
    --config configs/default.yaml \
    --output_dir checkpoints/run1 \
    --batch_size 32 \
    --learning_rate 3e-4
```

### Inference

```bash
python infer_flashgpt.py \
    --model_path checkpoints/run1/best_model.pt \
    --prompt "Your prompt here" \
    --max_length 100
```

### Diagnostics

```bash
python model_diagnostics.py \
    --model_path checkpoints/run1/best_model.pt \
    --test_batch_size 16 \
    --profile_memory True
```

## Advanced Features

1. **Attention Mechanisms**
   - ALiBi positional bias
   - Grouped Query Attention
   - Flash Attention
   - Sliding Window Attention

2. **Optimization Techniques**
   - Mixed precision training
   - Gradient checkpointing
   - Memory-efficient attention
   - Dynamic batch sizing

3. **Advanced Reasoning**
   - Tree of Thought processing
   - Step-by-step reasoning
   - Tool integration
   - Self-verification

4. **Hardware Optimizations**
   - CUDA optimizations
   - MPS (Metal) support
   - CPU fallback
   - Multi-GPU training

## Logging and Monitoring

- Training logs in `logs/training.log`
- Validation metrics in `logs/validation.log`
- TensorBoard support
- Model diagnostics

## Dependencies

Required packages (versions to be specified in requirements.txt):
- PyTorch
- transformers
- datasets
- numpy
- tqdm
- pyyaml
- tensorboard 