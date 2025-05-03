# FlashGPT Advanced Modules

This directory contains advanced training and inference scripts for the FlashGPT model, optimized for both CUDA and Apple Silicon (MPS) devices.

## Features

- Automatic device selection (CUDA/MPS/CPU)
- Efficient training with gradient checkpointing
- Automatic checkpointing every 120 seconds
- Validation logging with model outputs
- Dataset caching for faster training
- Configurable model parameters
- Interactive inference mode

## Directory Structure

```
FlashGPT-Advanced-Modules/
├── checkpoints/          # Directory for model checkpoints
├── datasets/            # Directory for cached datasets
├── logs/               # Directory for training and validation logs
├── train_flashgpt.py   # Training script
├── infer_flashgpt.py   # Inference script
└── README.md           # This file
```

## Installation

1. Ensure you have the required dependencies:
```bash
pip install torch transformers datasets tqdm
```

2. For Apple Silicon users, ensure you have the latest PyTorch version with MPS support.

## Training

### Basic Usage

```bash
# Train with default settings (MPS on Apple Silicon)
python train_flashgpt.py

# Train with CUDA
python train_flashgpt.py --cuda
```

### Advanced Options

```bash
# Customize training parameters
python train_flashgpt.py \
    --cuda \
    --batch_size 8 \
    --epochs 5 \
    --learning_rate 2e-4 \
    --max_seq_len 1024 \
    --checkpoint_interval 300 \
    --dataset "wikitext" \
    --dataset_config "wikitext-2-raw-v1"
```

### Arguments

- `--checkpoint_dir`: Directory to save checkpoints (default: "checkpoints")
- `--dataset`: Dataset to use for training (default: "wikitext")
- `--dataset_config`: Dataset configuration (default: "wikitext-2-raw-v1")
- `--cuda`: Use CUDA if available
- `--max_seq_len`: Maximum sequence length (default: 512)
- `--batch_size`: Training batch size (default: 4)
- `--epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--checkpoint_interval`: Checkpoint interval in seconds (default: 120)

## Inference

### Basic Usage

```bash
# Use latest checkpoint from checkpoints directory
python infer_flashgpt.py --checkpoint checkpoints/

# Use specific checkpoint
python infer_flashgpt.py --checkpoint checkpoints/checkpoint-epoch1-step1000.pt
```

### Advanced Options

```bash
# Customize generation parameters
python infer_flashgpt.py \
    --checkpoint checkpoints/ \
    --max_tokens 200 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.95
```

### Arguments

- `--checkpoint`: Path to model checkpoint or checkpoint directory
- `--cuda`: Use CUDA if available
- `--tokenizer`: Name or path to tokenizer (default: "gpt2")
- `--max_tokens`: Maximum tokens to generate (default: 100)
- `--temperature`: Generation temperature (default: 0.7)
- `--top_k`: Top-k sampling parameter (default: 40)
- `--top_p`: Top-p sampling parameter (default: 0.9)

## Logging

The training process generates several log files:

1. `training.log`: Contains training progress and metrics
2. `validation.log`: Contains validation prompts and model responses
3. `dataset_cache.log`: Contains information about cached datasets

Validation prompts are logged every 120 seconds during training, showing:
- Input prompt
- Model response
- Generation parameters
- Timestamp

## Checkpoints

Checkpoints are saved in the following format:
```
checkpoint-epoch{epoch}-step{step}.pt
```

Each checkpoint contains:
- Model state dictionary
- Model configuration
- Optimizer state
- Training progress (epoch and step)
- Timestamp

## Dataset Caching

Datasets are automatically cached in the `datasets/` directory to improve training performance. The cache includes:
- Tokenized dataset
- Dataset statistics
- Processing metadata

## Troubleshooting

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 2`
   - Reduce sequence length: `--max_seq_len 256`
   - Enable gradient checkpointing (default: enabled)

2. **MPS Performance Issues**
   - Ensure you're using the latest PyTorch version
   - Reduce batch size if experiencing memory issues
   - Monitor system memory usage

3. **Checkpoint Loading Issues**
   - Ensure the checkpoint file exists
   - Verify the model configuration matches
   - Check device compatibility (CUDA/MPS)

## Contributing

Feel free to submit issues and enhancement requests! 