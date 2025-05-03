import torch
import torch.nn as nn
import time
import os
import gc
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm.auto import tqdm
from .optimization import MPSMixedPrecision
from .models import GPTLMHeadModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging

def train_model_optimized(model: GPTLMHeadModel, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                         epochs: int, device: torch.device, tokenizer_name_or_path: str,
                         accumulation_steps: int = 1, scheduler=None,
                         mixed_precision: bool = True, max_grad_norm: float = 1.0,
                         checkpoint_path: str = "checkpoints/model.pt",
                         log_interval: int = 30, eval_interval: int = 500):
    """Optimized training loop for Apple Silicon GPUs."""
    # Setup mixed precision
    mp_handler = MPSMixedPrecision(enabled=mixed_precision)
    
    # Ensure checkpoint directory exists
    Path(os.path.dirname(checkpoint_path)).mkdir(parents=True, exist_ok=True)
    
    # Training metrics
    best_loss = float('inf')
    total_steps = len(train_loader) * epochs // accumulation_steps
    global_step = 0
    nan_counter = 0  # Track NaN losses for early stopping
    warmup_steps = 1000
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    print(f"\n===== Starting Training on {device} =====")
    print(f"Total steps: {total_steps}, Mixed precision: {mixed_precision}")
    
    # Memory tracking
    if torch.backends.mps.is_available():
        print(f"Initial MPS memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
    
    # Progress tracking
    progress_bar = tqdm(range(total_steps), desc="Training")
    
    # Learning rate warmup
    warmup_steps = min(1000, total_steps // 10)  # 10% of total steps or 1000, whichever is smaller
    
    # Pass tokenizer name/path instead of the object
    prompt_thread = run_regular_prompting(model, tokenizer_name_or_path, device)
    
    # Start checkpointing thread (every 600 seconds)
    # Keep interval_sec=600 as per user request, overriding default
    checkpoint_thread = save_checkpoint_periodically_optimized(model, checkpoint_path, interval_sec=600)
    
    # Training loop
    model.train()
    model.to(device)
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        # Reset gradients at the start of epoch
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Apply learning rate warmup
            if global_step < warmup_steps:
                lr_scale = float(global_step + 1) / warmup_steps
                for idx, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = base_lrs[idx] * lr_scale
            
            # Compute steps in accumulation loop
            with mp_handler as ctx:
                # Forward pass with mixed precision
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels, 
                    use_cache=False
                )
                loss = outputs['loss']
                
                # Check for NaN loss and handle
                if torch.isnan(loss).any():
                    print("NaN loss detected!")
                    print(f"Batch statistics: min={input_ids.min().item()}, max={input_ids.max().item()}")
                    nan_counter += 1
                    if nan_counter > 5:
                        print("Too many NaN losses, stopping training")
                        return
                    continue
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                
                # Backward pass
                loss = mp_handler.scale_loss(loss, optimizer)
                loss.backward()
                
                # Step optimizer if accumulation steps completed
                if (i + 1) % accumulation_steps == 0:
                    mp_handler.step(optimizer, loss, max_grad_norm, model)
                    optimizer.zero_grad()
                    
                    # Update scheduler if provided
                    if scheduler is not None:
                        scheduler.step()
                    
                    # Update progress
                    global_step += 1
                    progress_bar.update(1)
                    
                    # Log metrics
                    if global_step % log_interval == 0:
                        elapsed = time.time() - start_time
                        # Corrected loss logging to use the scaled value before accumulation reset
                        current_loss = loss.item() * accumulation_steps
                        print(f"Step {global_step}/{total_steps} | Loss: {current_loss:.4f} | "
                              f"Time: {elapsed:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Update epoch loss
            # Ensure loss is accumulated correctly even if loop exits before accumulation step
            epoch_loss += loss.item() * accumulation_steps if (i + 1) % accumulation_steps == 0 else loss.item() * ((i + 1) % accumulation_steps)
            
        # End of epoch
        # Average loss calculation needs adjustment if last batch wasn't a full accumulation cycle
        # Using average loss across accumulated steps provides a better epoch loss metric
        avg_epoch_loss = epoch_loss / len(train_loader) # Note: This might slightly misrepresent if accumulation causes uneven batch processing per epoch loss calculation
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # Save best model to a different file
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(os.path.dirname(checkpoint_path), "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            
    # End of training
    print(f"Training completed in {time.time() - start_time:.2f}s")
    return model

def save_checkpoint_periodically_optimized(model: nn.Module, save_path: str, interval_sec: int = 120):
    """Save model checkpoint periodically with optimized MPS handling."""
    import threading
    import time
    
    def save_func():
        while True:
            try:
                # Save model state
                torch.save(model.state_dict(), save_path)
                print(f"Checkpoint saved to {save_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
            time.sleep(interval_sec)
    
    # Start save thread
    save_thread = threading.Thread(target=save_func, daemon=True)
    save_thread.start()
    return save_thread

def prompt_model_optimized(model: GPTLMHeadModel, tokenizer, prompt: str, max_new_tokens: int = 100,
                         temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9,
                         do_sample: bool = True,
                         device: torch.device = None) -> str:
    """Generate text from a prompt using the model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure pad token is set on the passed tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Generate text
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Removed calculator tool logic as it wasn't used based on error logs and simplifies focus
    # if use_calculator_tool and "calculator" in prompt.lower():
    #     try:
    #         from .tools import CalculatorTool
    #         calculator = CalculatorTool()
    #         result = calculator.evaluate(prompt)
    #         if result is not None:
    #             generated_text += f"\nCalculator result: {result}"
    #     except Exception as e:
    #         print(f"Error using calculator tool: {e}")
    
    return generated_text 

def run_regular_prompting(model: GPTLMHeadModel, tokenizer_name_or_path: str, device: torch.device, interval_sec: int = 20):
    """Run model prompting at regular intervals using a fresh tokenizer in the thread."""
    import threading
    import time
    import logging
    import os
    
    # Configure logger
    prompting_logger = logging.getLogger('prompting')
    if not prompting_logger.handlers:
        # Create handlers
        console_handler = logging.StreamHandler()
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'prompting.log'))
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(log_format)
        file_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        prompting_logger.addHandler(console_handler)
        prompting_logger.addHandler(file_handler)
        prompting_logger.setLevel(logging.INFO)

    def prompt_func():
        # Initialize tokenizer inside the thread to avoid fork issues
        try:
            thread_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            # Force pad and eos tokens to be identical and use a common token
            special_token = "<|endoftext|>"  # Common special token in many models
            thread_tokenizer.pad_token = special_token
            thread_tokenizer.eos_token = special_token
            thread_tokenizer.pad_token_id = thread_tokenizer.convert_tokens_to_ids(special_token)
            thread_tokenizer.eos_token_id = thread_tokenizer.convert_tokens_to_ids(special_token)
            prompting_logger.info("Prompting thread tokenizer initialized.")
        except Exception as e:
            prompting_logger.error(f"Failed to initialize tokenizer in prompting thread: {e}")
            print(f"ERROR: Failed to initialize tokenizer in prompting thread: {e}")
            return  # Stop the thread if tokenizer fails

        while True:
            try:
                # Sleep first to allow model to train a bit before first prompt
                time.sleep(interval_sec)
                
                # Use a consistent prompt to track model progress
                prompt = "Explain quantum computing in simple terms."
                
                # Basic safe approach that doesn't rely on generate()
                # First encode the text
                input_tokens = thread_tokenizer.encode(prompt, return_tensors="pt").to(device)
                input_length = input_tokens.shape[1]  # Length of prompt in tokens
                
                # Manual generation loop instead of using generate()
                with torch.no_grad():
                    # Put model in eval mode
                    model.eval()
                    
                    # Start with input sequence
                    generated = input_tokens
                    max_len = 100  # Maximum tokens to generate
                    
                    # Track generated token histories for repetition penalty
                    token_history = []
                    repetition_penalty = 1.2
                    temperature = 0.8  # Temperature for sampling
                    top_k = 50
                    top_p = 0.95
                    
                    # Generate one token at a time
                    for _ in range(max_len):
                        # Forward pass (only need logits output)
                        try:
                            outputs = model(generated)
                            # Access logits as dictionary entry, not as an attribute
                            next_token_logits = outputs['logits'][:, -1, :]
                            
                            # Apply temperature
                            next_token_logits = next_token_logits / temperature
                            
                            # Apply repetition penalty to previously generated tokens
                            if token_history:
                                for token in token_history:
                                    next_token_logits[0, token] /= repetition_penalty
                            
                            # Apply top-p (nucleus) sampling
                            # Simplify the top-p implementation to avoid dimension mismatches
                            probs = torch.softmax(next_token_logits, dim=-1)
                            
                            # Apply repetition penalty
                            if token_history:
                                for token in token_history:
                                    probs[0, token] /= repetition_penalty
                            
                            # Re-normalize
                            probs = probs / probs.sum()
                            
                            # Get top-k tokens and their probabilities
                            top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
                            
                            # Keep only tokens with cumulative probability < top_p
                            cumulative_probs = torch.cumsum(top_k_probs, dim=-1)
                            tokens_to_keep = cumulative_probs < top_p
                            
                            # If no tokens meet the criteria, keep at least one token
                            if tokens_to_keep.sum() == 0:
                                tokens_to_keep[0, 0] = True
                            
                            # Filter top_k_indices and top_k_probs
                            top_k_probs = top_k_probs.masked_fill(~tokens_to_keep, 0)
                            
                            # Sample from filtered distribution
                            sampled_index = torch.multinomial(top_k_probs[0], 1)
                            next_token = top_k_indices[0, sampled_index]
                            
                            # Add to history
                            token_history.append(next_token.item())
                            
                            # Reshape for concatenation
                            next_token = next_token.unsqueeze(0)
                            
                            # Append to generated sequence
                            generated = torch.cat((generated, next_token), dim=-1)
                            
                            # Stop if we generate EOS token or reach max length
                            if next_token.item() == thread_tokenizer.eos_token_id:
                                break
                        except Exception as gen_error:
                            prompting_logger.error(f"Error in token generation: {gen_error}")
                            break
                            
                    # Decode only the newly generated tokens (exclude the prompt)
                    generated_text = ""
                    if generated.shape[1] > input_length:
                        generated_text = thread_tokenizer.decode(generated[0, input_length:], skip_special_tokens=True)
                    
                    # Log the result
                    timestamp = time.strftime('%H:%M:%S')
                    prompting_logger.info(f"=== Model Prompt ({timestamp}) ===\nPrompt: {prompt}\nResponse: {generated_text if generated_text else '[No output generated]'}\n{'='*50}")
                    print(f"\n=== Model Prompt ({timestamp}) ===\nPrompt: {prompt}\nResponse: {generated_text if generated_text else '[No output generated]'}\n{'='*50}")
                
            except Exception as e:
                error_msg = f"Error during prompting: {e}"
                prompting_logger.error(error_msg)
                print(error_msg)
    
    # Start prompt thread
    prompt_thread = threading.Thread(target=prompt_func, daemon=True)
    prompt_thread.start()
    return prompt_thread 