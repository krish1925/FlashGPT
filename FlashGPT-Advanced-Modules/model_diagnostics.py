import torch
import os
import logging
from pathlib import Path
from transformers import AutoTokenizer
from modules import (
    GPTConfig, GPTLMHeadModel, OptimizedDataset, MPSDataLoader,
    prompt_model_optimized
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/diagnostics.log'),
        logging.StreamHandler()
    ]
)

def get_device():
    """Get the appropriate device for running diagnostics."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def check_tokenizer_model_alignment(tokenizer, model_config):
    """Verify that the tokenizer and model vocabulary sizes match."""
    tokenizer_vocab_size = tokenizer.vocab_size
    model_vocab_size = model_config.vocab_size
    
    logging.info(f"Tokenizer vocabulary size: {tokenizer_vocab_size}")
    logging.info(f"Model vocabulary size: {model_vocab_size}")
    
    if tokenizer_vocab_size != model_vocab_size:
        logging.warning(f"MISMATCH: Tokenizer vocab size ({tokenizer_vocab_size}) "
                       f"doesn't match model vocab size ({model_vocab_size})")
        return False
    else:
        logging.info("✓ Tokenizer and model vocabulary sizes match")
        return True

def test_tokenizer_roundtrip(tokenizer):
    """Test tokenizer encode-decode roundtrip."""
    test_texts = [
        "Explain quantum computing in simple terms.",
        "The quick brown fox jumps over the lazy dog.",
        "In 1921, Einstein received the Nobel Prize in Physics."
    ]
    
    logging.info("Testing tokenizer roundtrip:")
    
    for test_text in test_texts:
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        logging.info(f"Original: {test_text}")
        logging.info(f"Encoded: {encoded[:10]}... (length: {len(encoded)})")
        logging.info(f"Decoded: {decoded}")
        
        if test_text not in decoded:
            logging.warning(f"Roundtrip failed: original text not preserved")
        else:
            logging.info("✓ Roundtrip successful")
        
        logging.info("-" * 50)

def inspect_dataset_samples(dataset, tokenizer, num_samples=5):
    """Inspect processed samples from the training dataset."""
    logging.info(f"Inspecting {num_samples} dataset samples:")
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        logging.info(f"Sample {i+1}:")
        logging.info(f"Input IDs: {sample['input_ids'][:10]}... (length: {len(sample['input_ids'])})")
        
        decoded = tokenizer.decode(sample['input_ids'])
        logging.info(f"Decoded: {decoded[:100]}...")
        
        if 'labels' in sample:
            logging.info(f"Labels: {sample['labels'][:10]}... (length: {len(sample['labels'])})")
            
            # Check if labels are properly shifted from input_ids
            if len(sample['input_ids']) == len(sample['labels']):
                matches = sum(1 for i, j in zip(sample['input_ids'][:-1], sample['labels'][1:]) if i == j)
                percent_match = matches / (len(sample['input_ids']) - 1) * 100
                logging.info(f"Label shifting check: {percent_match:.2f}% match (should be close to 100%)")
        
        if 'attention_mask' in sample:
            logging.info(f"Attention mask: {sample['attention_mask'][:10]}... (sum: {sum(sample['attention_mask'])})")
        
        logging.info("-" * 50)

def prompt_model_simple(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, device=None):
    """Simplified model prompting function."""
    if device is None:
        device = model.device if hasattr(model, 'device') else torch.device('cpu')
    
    model.eval()
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with simple parameters
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=(temperature > 0.1),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode and return only the new tokens
            generated_text = tokenizer.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            logging.error(f"Error during generation: {e}")
            # Fallback to manual generation if the built-in generate method fails
            return manual_token_generation(model, tokenizer, prompt, max_new_tokens, temperature, device)

def manual_token_generation(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, device=None):
    """Manual token-by-token generation as a fallback."""
    if device is None:
        device = model.device if hasattr(model, 'device') else torch.device('cpu')
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = input_ids.shape[1]
    
    # Generate tokens one by one
    with torch.no_grad():
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Forward pass
            try:
                outputs = model(generated)
                next_token_logits = outputs['logits'][:, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Sample from the distribution
                if temperature > 0.1:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs[0], 1).unsqueeze(0)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                
                # Append to the sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Stop if we generate an EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
            except Exception as e:
                logging.error(f"Error in manual generation step: {e}")
                break
    
    # Decode only the generated part
    if generated.shape[1] > input_length:
        generated_text = tokenizer.decode(generated[0, input_length:], skip_special_tokens=True)
        return generated_text
    else:
        return "[No output generated]"

def compute_perplexity(model, tokenizer, texts, device=None):
    """Compute perplexity on a list of text samples."""
    if device is None:
        device = model.device if hasattr(model, 'device') else torch.device('cpu')
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            try:
                encodings = tokenizer(text, return_tensors="pt")
                encodings = {k: v.to(device) for k, v in encodings.items()}
                labels = encodings["input_ids"].clone()
                
                outputs = model(
                    input_ids=encodings["input_ids"],
                    attention_mask=encodings["attention_mask"],
                    labels=labels
                )
                
                loss = outputs["loss"].item()
                tokens = labels.numel()
                
                total_loss += loss * tokens
                total_tokens += tokens
            except Exception as e:
                logging.error(f"Error computing perplexity: {e}")
    
    if total_tokens > 0:
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        return perplexity.item()
    else:
        return float('inf')

def register_attention_hooks(model):
    """Register hooks to inspect attention patterns."""
    attention_outputs = []
    
    def hook_fn(module, input, output):
        attention_outputs.append(output.detach())
    
    hooks = []
    for name, module in model.named_modules():
        if "attn" in name:
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    return attention_outputs, hooks

def run_diagnostics(model_path=None):
    """Run comprehensive model diagnostics."""
    # Set up device
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Create model config
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_kv_heads=12,
        dropout_prob=0.1,
        flash_attention=False,
        use_gqa=False
    )
    
    # Create or load model
    if model_path and os.path.exists(model_path):
        logging.info(f"Loading model from {model_path}")
        model = GPTLMHeadModel(config)
        model.load_state_dict(torch.load(model_path))
    else:
        logging.info("Creating new model with default initialization")
        model = GPTLMHeadModel(config)
    
    model.to(device)
    
    # Run tokenizer-model alignment check
    alignment_ok = check_tokenizer_model_alignment(tokenizer, config)
    
    # Test tokenizer roundtrip
    test_tokenizer_roundtrip(tokenizer)
    
    # Create a small test dataset for inspection
    logging.info("Creating small test dataset...")
    test_data = [
        {"text": "Quantum computing is an exciting field that combines physics and computer science."},
        {"text": "The quick brown fox jumps over the lazy dog. This pangram contains all letters of the English alphabet."},
        {"text": "Machine learning models require careful tuning to achieve optimal performance."}
    ]
    
    test_dataset = OptimizedDataset(
        test_data,
        tokenizer=tokenizer,
        max_length=512
    )
    
    # Inspect dataset samples
    inspect_dataset_samples(test_dataset, tokenizer)
    
    # Test generation
    logging.info("Testing text generation:")
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the main components of a transformer model?"
    ]
    
    for prompt in test_prompts:
        logging.info(f"Prompt: {prompt}")
        
        # Try simplified generation
        try:
            logging.info("Using simplified generation:")
            result = prompt_model_simple(model, tokenizer, prompt, device=device)
            logging.info(f"Generated: {result}")
        except Exception as e:
            logging.error(f"Simplified generation failed: {e}")
        
        # Try original generation
        try:
            logging.info("Using original generation:")
            result = prompt_model_optimized(model, tokenizer, prompt, device=device)
            logging.info(f"Generated: {result}")
        except Exception as e:
            logging.error(f"Original generation failed: {e}")
        
        logging.info("-" * 50)
    
    # Compute perplexity
    logging.info("Computing perplexity on test data...")
    eval_texts = [item["text"] for item in test_data]
    try:
        perplexity = compute_perplexity(model, tokenizer, eval_texts, device)
        logging.info(f"Perplexity: {perplexity:.4f}")
    except Exception as e:
        logging.error(f"Perplexity calculation failed: {e}")
    
    # Register and test attention hooks
    logging.info("Testing attention pattern inspection...")
    try:
        attention_outputs, hooks = register_attention_hooks(model)
        
        # Run a forward pass to collect attention patterns
        inputs = tokenizer("This is a test sentence.", return_tensors="pt").to(device)
        model(**inputs)
        
        logging.info(f"Collected {len(attention_outputs)} attention patterns")
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
    except Exception as e:
        logging.error(f"Attention pattern inspection failed: {e}")
    
    logging.info("Diagnostics completed.")

if __name__ == "__main__":
    Path('logs').mkdir(parents=True, exist_ok=True)
    
    # Check for latest checkpoint
    checkpoint_path = None
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        if checkpoint_files:
            checkpoint_path = str(max(checkpoint_files, key=os.path.getmtime))
    
    run_diagnostics(checkpoint_path) 