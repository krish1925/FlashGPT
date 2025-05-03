# FlashGPT

FlashGPT is a high-performance, modular implementation of the GPT architecture with advanced features like Tree-of-Thought reasoning, algorithmic reasoning modules, and hardware-specific optimizations.

## Project Structure

```
.
├── FlashGPT-Advanced-Modules/     # Advanced modular implementation
├── FlashGPT-Advanced-Thinking-Old/  # Legacy implementation with reasoning modules
├── datasets/                      # Training datasets
├── checkpoints/                   # Model checkpoints
├── checkpoints_turbo/            # Optimized model checkpoints
├── data/                         # Processed data
├── simplebooks_data/             # Book dataset
└── Notebooks/
    ├── FlashGPT-Base.ipynb           # Base implementation
    ├── FlashGPT-Base-Clean.ipynb     # Clean, documented base implementation
    ├── FlashGPT-Advanced.ipynb       # Advanced features implementation
    ├── FlashGPT-Advanced-Thinking.ipynb  # Reasoning-enhanced implementation
    ├── FlashGPT-Distributed-Training.ipynb  # Distributed training setup
    └── FlashGPT-distillation-basic.ipynb   # Model distillation

```

## Core Components

### Model Architecture
- Modern Transformer architecture with enhancements
- RMSNorm for improved stability
- SwiGLU activation functions
- Advanced attention mechanisms (ALiBi, RoPE, GQA)
- Optional Flash Attention support

### Advanced Features
- Tree of Thought reasoning
- Algorithmic reasoning modules
- Calculator tool integration
- Multi-step reasoning capabilities
- Hardware-specific optimizations (MPS, CUDA)

### Training Features
- Mixed precision training
- Distributed training support
- Model distillation capabilities
- Gradient checkpointing
- Dynamic batch sizing

## Key Files

### Python Modules
- `flashgpt_model.py`: Core model implementation
- `prompt_flashgpt.py`: Prompt engineering and processing
- `train_flashgpt.py`: Training utilities
- `infer_flashgpt.py`: Inference utilities

### Notebooks
Each notebook focuses on specific aspects of the model:
- **FlashGPT-Base**: Basic implementation and training
- **FlashGPT-Base-Clean**: Clean, documented implementation
- **FlashGPT-Advanced**: Advanced features and optimizations
- **FlashGPT-Distributed-Training**: Distributed training setup
- **FlashGPT-distillation-basic**: Model distillation techniques

## Getting Started

1. **Environment Setup**
   ```bash
   # Clone the repository
   git clone [repository-url]
   cd FlashGPT
   
   # Install dependencies (requirements.txt to be created)
   pip install -r requirements.txt
   ```

2. **Training**
   - Use notebooks for interactive development
   - Use Python scripts for production training
   - Checkpoints are saved in `checkpoints/` directory

3. **Inference**
   - Load pretrained models from checkpoints
   - Use inference scripts or notebooks
   - Support for various inference optimizations

## Hardware Support

- CUDA-enabled GPUs
- Apple Silicon (MPS) optimization
- CPU fallback with optimizations
- Distributed training across multiple devices

## Model Capabilities

1. **Base Capabilities**
   - Language modeling
   - Text generation
   - Context understanding

2. **Advanced Reasoning**
   - Tree of Thought processing
   - Multi-step reasoning
   - Calculator integration
   - Algorithmic problem solving

3. **Optimizations**
   - Memory efficient attention
   - Gradient checkpointing
   - Mixed precision training
   - Hardware-specific optimizations

## Contributing

Contributions are welcome! Please read the contribution guidelines before submitting pull requests.

## License

[License information to be added]

## Acknowledgments

- Implementation inspired by various transformer architectures
- Incorporates modern ML techniques and optimizations
