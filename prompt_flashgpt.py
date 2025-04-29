import os
import sys
import torch
import argparse
from transformers import AutoTokenizer
import flashgpt_model

# Check if the main module is importable
try:
    # Try importing the main model module
    from flashgpt_model import GPTConfig, MPSOptimizedEnhancedGPTLMHeadModel, prompt_model_optimized
    print("Successfully imported model components")
except ImportError:
    print("Failed to import the main model module. Make sure it's in your Python path.")
    print("Current system path:", sys.path)
    sys.exit(1)

def get_device():
    """Get the best available device for PyTorch."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

def main():
    parser = argparse.ArgumentParser(description="Prompt a saved GPT model")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to model checkpoint (required)")
    
    # Optional arguments with defaults
    parser.add_argument("--tokenizer", type=str, default="gpt2", 
                        help="Name or path to tokenizer (default: gpt2)")
    parser.add_argument("--max_tokens", type=int, default=100, 
                        help="Maximum tokens to generate (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Generation temperature (default: 0.7)")
    
    args = parser.parse_args()
    
    # Set up device
    device = get_device()
    
    # Load tokenizer
    try:
        print(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        print(f"Tokenizer loaded with vocabulary size: {vocab_size}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    try:
        # Load the checkpoint to inspect its structure
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Try to extract model configuration from checkpoint
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint:
                config = checkpoint['config']
                print("Found model configuration in checkpoint")
            elif 'model_config' in checkpoint:
                config = checkpoint['model_config']
                print("Found model configuration in checkpoint")
            else:
                # Create a default configuration
                print("No configuration found in checkpoint, using default values")
                config = GPTConfig(
                    vocab_size=vocab_size,
                    max_seq_len=256,
                    n_embd=128,
                    n_layer=4,
                    n_head=4,
                    dropout_prob=0.1,
                    alibi=False,
                    use_rope=True,
                    flash_attention=True,
                    n_kv_heads=2,
                    use_gqa=True,
                    use_rwkv=False,
                    use_ssm=False,
                    use_moe=False,
                    num_experts=4,
                    top_k_experts=2,
                    gradient_checkpointing=False,
                    mla_n_latent=8,
                    use_mla=True,
                    reasoning_steps=1,
                    use_reasoning_tracker=False,
                    use_algorithmic_reasoner=False,
                    use_calculator=True,
                )
        else:
            # If checkpoint is not a dict, create default config
            print("Checkpoint is not a dictionary, using default configuration")
            config = GPTConfig(
                vocab_size=vocab_size,
                max_seq_len=256,
                n_embd=128,
                n_layer=4,
                n_head=4,
                dropout_prob=0.1,
                alibi=False,
                use_rope=True,
                flash_attention=True,
                n_kv_heads=2,
                use_gqa=True,
                use_rwkv=False,
                use_ssm=False,
                use_moe=False,
                num_experts=4,
                top_k_experts=2,
                gradient_checkpointing=False,
                mla_n_latent=8,
                use_mla=True,
                reasoning_steps=1,
                use_reasoning_tracker=False,
                use_algorithmic_reasoner=False,
                use_calculator=True,
            )
        
        # Initialize the model
        print("Initializing model with configuration")
        model = MPSOptimizedEnhancedGPTLMHeadModel(config)
        
        # Load the state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model state from checkpoint")
        else:
            # Try loading the checkpoint directly as a state dict
            try:
                model.load_state_dict(checkpoint)
                print("Loaded checkpoint as raw state dict")
            except Exception as e:
                print(f"Error loading model state: {e}")
                return
        
        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()
        print("Model successfully loaded and ready")
        
        # Prompt loop
        print("\n=== Model Ready for Prompting ===")
        print("Type 'quit', 'exit', or 'q' to end the session")
        
        while True:
            try:
                # Get user input
                user_input = input("\nEnter your prompt: ")
                
                # Check for exit command
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Exiting prompt session")
                    break
                
                # Skip empty inputs
                if not user_input.strip():
                    continue
                
                # Generate response
                print("\nGenerating response...")
                response = prompt_model_optimized(
                    model,
                    tokenizer,
                    user_input,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=40,
                    top_p=0.9,
                    do_sample=True,
                    use_calculator_tool=config.use_calculator,
                    device=device
                )
                
                print(f"\n=== Model Response ===\n{response}")
                
            except KeyboardInterrupt:
                print("\nInterrupt received. Exiting gracefully...")
                break
            except Exception as e:
                print(f"Error during generation: {e}")
    
    except Exception as e:
        print(f"Error initializing model: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"Unexpected error: {e}")