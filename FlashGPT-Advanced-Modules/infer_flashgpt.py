import os
import sys
import torch
import argparse
import logging
from pathlib import Path
from transformers import AutoTokenizer
from modules import (
    GPTConfig, GPTLMHeadModel, MPSMixedPrecision, optimize_for_apple_silicon,
    OptimizedDataset, MPSDataLoader, train_model_optimized, prompt_model_optimized,
    CalculatorTool, TreeOfThought, AlgorithmicReasoner
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_device(use_cuda=False):
    """Get the best available device for PyTorch."""
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
    return device

def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model configuration")
    
    config = checkpoint['config']
    model = GPTLMHeadModel(config)
    
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint does not contain model state dictionary")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def main():
    parser = argparse.ArgumentParser(description="Inference with FlashGPT model")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint or checkpoint directory")
    
    # Optional arguments
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="Name or path to tokenizer")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    # Set up device
    device = get_device(args.cuda)
    
    # Load tokenizer
    try:
        logging.info(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        return
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        # Find the latest checkpoint in the directory
        checkpoints = list(checkpoint_path.glob("checkpoint-*.pt"))
        if not checkpoints:
            logging.error(f"No checkpoints found in {checkpoint_path}")
            return
        checkpoint_path = max(checkpoints, key=os.path.getctime)
        logging.info(f"Using latest checkpoint: {checkpoint_path}")
    
    try:
        model, config = load_model_from_checkpoint(checkpoint_path, device)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return
    
    # Interactive prompting loop
    logging.info("\n=== Model Ready for Prompting ===")
    logging.info("Type 'quit', 'exit', or 'q' to end the session")
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter your prompt: ")
            
            # Check for exit command
            if user_input.lower() in ["quit", "exit", "q"]:
                logging.info("Exiting prompt session")
                break
            
            # Skip empty inputs
            if not user_input.strip():
                continue
            
            # Generate response
            logging.info("Generating response...")
            response = prompt_model_optimized(
                model,
                tokenizer,
                user_input,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=True,
                use_calculator_tool=config.use_calculator,
                device=device
            )
            
            print(f"\n=== Model Response ===\n{response}")
            
        except KeyboardInterrupt:
            logging.info("\nInterrupt received. Exiting gracefully...")
            break
        except Exception as e:
            logging.error(f"Error during generation: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}") 