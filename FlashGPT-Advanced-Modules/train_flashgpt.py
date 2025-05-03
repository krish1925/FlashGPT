import os
import sys
import time
import torch
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer
from datasets import load_dataset
from modules import (
    GPTConfig, GPTLMHeadModel, MPSMixedPrecision, optimize_for_apple_silicon,
    OptimizedDataset, MPSDataLoader, train_model_optimized, prompt_model_optimized,
    CalculatorTool, TreeOfThought, AlgorithmicReasoner
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

# Create validation logger
validation_logger = logging.getLogger('validation')
validation_logger.setLevel(logging.INFO)
validation_handler = logging.FileHandler('logs/validation.log')
validation_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
validation_logger.addHandler(validation_handler)

# Create prompting logger
prompting_logger = logging.getLogger('prompting')
prompting_logger.setLevel(logging.INFO)
prompting_handler = logging.FileHandler('logs/prompting.log')
prompting_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
prompting_logger.addHandler(prompting_handler)

# Ensure logs directory exists
Path('logs').mkdir(parents=True, exist_ok=True)

def get_device(use_cuda: bool = True) -> torch.device:
    """Get the appropriate device for training."""
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def save_checkpoint(model: GPTLMHeadModel, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, checkpoint_dir: str):
    """Save model checkpoint."""
    # Ensure checkpoint directory exists
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(model: GPTLMHeadModel, optimizer: torch.optim.Optimizer, 
                   checkpoint_path: str) -> tuple[int, float]:
    """Load model checkpoint."""
    # Ensure checkpoint directory exists
    Path(os.path.dirname(checkpoint_path)).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(checkpoint_path):
        logging.info(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        return 0, float('inf')
        
    try:
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))
        else:
            # Handle old format where the checkpoint is just the model state
            model.load_state_dict(checkpoint)
            return 0, float('inf')
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        return 0, float('inf')

def cache_dataset(dataset_name: str, dataset_config: str = "wikitext-2-raw-v1", cache_dir: str = "cache"):
    """Cache dataset for faster loading."""
    cache_path = Path(cache_dir) / f"{dataset_name}_{dataset_config}.json"
    if cache_path.exists():
        logging.info(f"Loading cached dataset from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    logging.info(f"Loading dataset {dataset_name} with config {dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config)
    
    # Process and cache the dataset
    processed_data = []
    for item in dataset['train']:
        # Check for different possible text field names
        text = None
        for field in ['text', 'content', 'article', 'sentence']:
            if field in item:
                text = item[field]
                break
        
        if text is not None:
            processed_data.append({'text': text})
    
    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(processed_data, f)
    
    logging.info(f"Cached dataset to {cache_path}")
    return processed_data

def run_validation(model: GPTLMHeadModel, tokenizer, device: torch.device):
    """Run validation with predefined prompts."""
    validation_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "Calculate 2 + 2 * 3 using the calculator tool.",
        "What are the main components of a transformer model?"
    ]
    
    for prompt in validation_prompts:
        try:
            response = prompt_model_optimized(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device
            )
            validation_logger.info(f"Prompt: {prompt}\nResponse: {response}\n{'='*50}")
        except Exception as e:
            validation_logger.error(f"Error during validation: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Train FlashGPT model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--dataset', type=str, default='wikitext',
                      help='Dataset to use for training')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1',
                      help='Dataset configuration (e.g., wikitext-2-raw-v1)')
    parser.add_argument('--cuda', action='store_true',
                      help='Use CUDA if available')
    parser.add_argument('--max_seq_len', type=int, default=512,
                      help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--checkpoint_interval', type=int, default=500,
                      help='Steps between checkpoints')
    parser.add_argument('--validation_interval', type=int, default=120,
                      help='Seconds between validation runs')
    args = parser.parse_args()

    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device(args.cuda)
    logging.info(f"Using device: {device}")
    
    # Load dataset and tokenizer
    dataset = cache_dataset(args.dataset, args.dataset_config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Initialize model configuration
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_kv_heads=12,  # Match n_head for standard attention
        dropout_prob=0.1,
        flash_attention=False,  # Disable for now
        use_gqa=False  # Disable for now
    )
    
    # Initialize model
    model = GPTLMHeadModel(config)
    model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Check for existing checkpoint
    checkpoint_path = Path(args.checkpoint_dir) / "latest_checkpoint.pt"
    start_epoch = 0
    if checkpoint_path.exists():
        start_epoch, _ = load_checkpoint(model, optimizer, str(checkpoint_path))
        logging.info(f"Resuming training from epoch {start_epoch}")
    
    # Create dataset and dataloader
    train_dataset = OptimizedDataset(
        dataset,
        tokenizer=tokenizer,
        max_length=args.max_seq_len
    )
    train_loader = MPSDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True
    )
    
    # Train model
    model = train_model_optimized(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        tokenizer_name_or_path="gpt2",
        accumulation_steps=1,  # Reduced for testing
        mixed_precision=False,  # Disabled for testing
        checkpoint_path=str(checkpoint_path),
        log_interval=10,
        eval_interval=100  # Reduced for more frequent checkpoints
    )
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, 0.0, args.checkpoint_dir)
    logging.info("Training completed!")

if __name__ == "__main__":
    main() 