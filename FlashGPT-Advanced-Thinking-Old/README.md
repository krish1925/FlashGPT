# FlashGPT Advanced Thinking (Legacy)

This directory contains the original implementation of FlashGPT with advanced reasoning capabilities. While this is considered the legacy version, it contains valuable implementations of reasoning modules and hardware optimizations that might be useful for reference.

## Directory Contents

```
.
├── FlashGPT-Advanced-Thinking.py  # Main model implementation
├── train_flashgpt.py              # Training script
└── infer_flashgpt.py             # Inference utilities
```

## Key Features

### 1. Advanced Reasoning Modules

- **Tree of Thought Processing**
  - Multi-step reasoning capabilities
  - Branching thought paths
  - Solution evaluation
  - Path optimization

- **Algorithmic Reasoning**
  - Register-based computation
  - Step-by-step problem solving
  - Verification mechanisms
  - Error handling

- **Calculator Integration**
  - Mathematical expression parsing
  - Safe computation handling
  - Result verification
  - Error reporting

### 2. Hardware Optimizations

- **Apple Silicon (MPS) Support**
  - Metal Performance Shaders optimization
  - Memory management
  - Performance profiling
  - Fallback mechanisms

- **Mixed Precision Training**
  - Dynamic scaling
  - Memory efficiency
  - Performance optimization
  - Gradient handling

### 3. Architecture Features

- **Attention Mechanisms**
  - ALiBi positional bias
  - RoPE embeddings
  - Grouped Query Attention
  - Flash Attention support

- **Model Components**
  - RMSNorm for stability
  - SwiGLU activation
  - Memory-efficient attention
  - Gradient checkpointing

## Implementation Details

### FlashGPT-Advanced-Thinking.py (146KB)
- Core model implementation
- Advanced reasoning modules
- Hardware optimizations
- Attention mechanisms
- Training utilities

### train_flashgpt.py (9.5KB)
- Training loop implementation
- Data handling
- Optimization setup
- Checkpoint management
- Logging utilities

### infer_flashgpt.py (5.2KB)
- Inference utilities
- Text generation
- Model evaluation
- Interactive mode
- Batch processing

## Key Differences from Current Version

1. **Architecture**
   - Monolithic implementation vs. modular design
   - Integrated reasoning vs. separate modules
   - Direct hardware optimizations vs. abstraction layer

2. **Features**
   - Combined reasoning modules
   - Integrated calculator
   - Direct MPS optimization
   - Custom attention implementations

3. **Training**
   - Single-file training implementation
   - Integrated optimization
   - Direct checkpoint handling
   - Custom data processing

## Usage Notes

While this is a legacy implementation, it contains valuable reference code for:
1. Hardware-specific optimizations
2. Advanced reasoning implementations
3. Efficient attention mechanisms
4. Training optimizations

## Dependencies

Core dependencies (versions as of original implementation):
- PyTorch (>=1.12.0)
- transformers
- numpy
- tqdm
- sympy (for calculator)

## Migration Guide

If you're using this version, consider:
1. Migrating to the new modular implementation
2. Extracting specific components for reuse
3. Adapting optimizations for current hardware
4. Updating dependencies

## Reference Implementation

This implementation serves as a reference for:
1. Advanced reasoning mechanisms
2. Hardware optimization techniques
3. Efficient attention implementations
4. Training optimization strategies

## Note on Maintenance

This is a legacy implementation maintained for reference. For new projects:
- Use the current modular implementation
- Reference this code for specific optimizations
- Adapt useful components to current architecture
- Consider hardware-specific requirements 