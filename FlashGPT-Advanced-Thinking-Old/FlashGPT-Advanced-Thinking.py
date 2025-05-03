import os
import time
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset # Using datasets library for example data
from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint
import sympy as sp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase # For tokenization




# MPS related stuff ###################################################
# --- Configuration for MPS Optimization ---

def optimize_for_apple_silicon():
    """Configure PyTorch for optimal performance on Apple Silicon."""
    import torch
    import os
    
    # Set environment variables for MPS optimization
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Determine device and settings
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
        # Print device properties 
        print("Apple Silicon MPS device detected")
        print(f"Available memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
        
        # Set autocast and benchmark flags
        # torch.backends.mps.enable_cudnn_benchmark = True  # Optimize performance
    else:
        device = torch.device("cpu")
        print("MPS device not found, using CPU")
    
    # Return optimized device
    return device

# --- Mixed-Precision Training Helper ---

class MPSMixedPrecision:
    """Helper for mixed precision training on MPS"""
    def __init__(self, enabled=True, dtype=torch.float16):
        self.enabled = enabled and torch.backends.mps.is_available()
        self.dtype = dtype
        self.scaler = torch.amp.GradScaler(enabled=self.enabled)
    
    def __enter__(self):
        if self.enabled:
            return torch.autocast(device_type="mps", dtype=self.dtype)
        else:
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyContext()
    
    def __exit__(self, *args):
        pass
    
    def scale_loss(self, loss, optimizer):
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer, loss=None, clip_grad=None, model=None):
        if self.enabled:
            if clip_grad is not None and model is not None:
                # Unscale before clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if clip_grad is not None and model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

# --- Optimized DataLoader for MPS ---

class MPSDataLoader:
    """DataLoader optimized for MPS with prefetching and pinning workarounds"""
    def __init__(self, dataset, batch_size, num_workers, shuffle=True):
        import torch
        from torch.utils.data import DataLoader
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        # For MPS, we need custom handling since pin_memory isn't supported
        if torch.backends.mps.is_available():
            # Create a standard dataloader without pin_memory
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
                prefetch_factor=3 if self.num_workers > 0 else None  # Increase prefetch factor
            )
        else:
            # For other systems, use standard pinned memory
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False,
                prefetch_factor=2 if self.num_workers > 0 else None
            )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)

# --- Memory-Optimized Dataset ---

class OptimizedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length, preload=False):
        import torch
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preload = preload
        self.preloaded_items = None
        
        # Preload a subset for faster access if requested
        if self.preload:
            print("Preloading dataset subset...")
            self.preloaded_items = []
            for idx in range(min(10000, len(self.dataset))):
                self.preloaded_items.append(self._process_item(idx))
            print(f"Preloaded {len(self.preloaded_items)} items")
            
    def _process_item(self, idx):
        text = self.dataset[idx]['text'] if 'text' in self.dataset[idx] else self.dataset[idx]['content']
        encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, 
                                  padding="max_length", return_tensors="pt")
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.preload and idx < len(self.preloaded_items):
            return self.preloaded_items[idx]
        return self._process_item(idx)

# --- Enhanced Training Loop with MPS Optimizations ---

def train_model_optimized(model, train_loader, optimizer, epochs, device, 
                         accumulation_steps=1, scheduler=None, 
                         mixed_precision=True, max_grad_norm=1.0,
                         checkpoint_path="checkpoints/model.pt",
                         log_interval=10, eval_interval=500):
    """Optimized training loop for Apple Silicon GPUs."""
    import torch
    import time
    import os
    import gc
    from tqdm.auto import tqdm
    from pathlib import Path
    
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
    base_lrs     = [pg['lr'] for pg in optimizer.param_groups]
   
    
    print(f"\n===== Starting Training on {device} =====")
    print(f"Total steps: {total_steps}, Mixed precision: {mixed_precision}")
    
    # Memory tracking
    if torch.backends.mps.is_available():
        print(f"Initial MPS memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
    
    # Progress tracking
    progress_bar = tqdm(range(total_steps), desc="Training")
    
    # Learning rate warmup (add this!)
    warmup_steps = min(1000, total_steps // 10)  # 10% of total steps or 1000, whichever is smaller
    
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
            # **Apply warmup scaling here, using base_lrs**:
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
                    
                    # Increment NaN counter and check for early stopping
                    nan_counter += 1
                    if nan_counter >= 5:  # Stop after 5 consecutive NaN losses
                        print("Too many NaN losses in a row. Early stopping.")
                        raise ValueError("Training unstable - too many NaN losses")
                    
                    # Skip this batch and continue
                    continue
                else:
                    # Reset NaN counter on successful batch
                    nan_counter = 0
                                
                # Scale loss for gradient accumulation
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps
                
                # Scale loss for mixed precision
                scaled_loss = mp_handler.scale_loss(loss, optimizer)
                
            # Backward pass
            scaled_loss.backward()
            
            # Update weights (after accumulation steps)
            if (i + 1) % accumulation_steps == 0 or (i + 1 == len(train_loader)):
                # Apply gradient clipping and optimizer step with MP handler
                mp_handler.step(
                    optimizer=optimizer,
                    clip_grad=max_grad_norm,
                    model=model
                )
                optimizer.zero_grad()
                
                # Step LR scheduler if provided
                if scheduler is not None:
                    scheduler.step()
                
                # Update progress
                progress_bar.update(1)
                global_step += 1
                
                # Log metrics
                if global_step % log_interval == 0:
                    # Calculate throughput
                    elapsed = time.time() - start_time
                    samples_per_sec = (global_step * train_loader.batch_size * accumulation_steps) / elapsed
                    
                    # Get current learning rate
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # Memory info (MPS specific)
                    mem_str = ""
                    if torch.backends.mps.is_available():
                        mem_allocated = torch.mps.current_allocated_memory() / 1e9
                        mem_str = f", Memory: {mem_allocated:.2f} GB"
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        "Epoch": epoch + 1, 
                        "Loss": loss.item() * accumulation_steps,
                        "LR": f"{current_lr:.2e}",
                        "Samples/sec": f"{samples_per_sec:.2f}"
                    })
                    
                    # Log to logger
                    logger.info(f"Epoch: {epoch+1}, Step: {global_step}/{total_steps}, "
                               f"Loss: {loss.item() * accumulation_steps:.4f}, "
                               f"LR: {current_lr:.2e}, Samples/sec: {samples_per_sec:.2f}{mem_str}")
                    
                # Save checkpoint periodically
                if global_step % eval_interval == 0:
                    # Save model
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                    
                    # Optional: Run garbage collection
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            
            # Accumulate loss for epoch statistics
            epoch_loss += loss.item() * accumulation_steps
            
            # Periodic memory cleanup during epoch
            if i % 100 == 0 and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN detected in {name}")
                if param.abs().max() > 100:
                    print(f"Large value detected in {name}: {param.abs().max().item()}")
        
        # End of epoch reporting
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} finished in {epoch_time:.2f}s. Avg Loss: {avg_epoch_loss:.4f}")
        logger.info(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Save end-of-epoch checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path.replace(".pt", f"_epoch_{epoch+1}.pt"))
    
    progress_bar.close()
    total_time = time.time() - start_time
    print(f"\n===== Training Complete =====")
    print(f"Total time: {total_time:.2f}s, Final loss: {loss.item():.4f}")
    logger.info(f"Training complete! Total time: {total_time:.2f}s, Final loss: {loss.item():.4f}")
    
    # Final memory report
    if torch.backends.mps.is_available():
        print(f"Final MPS memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
    
    return global_step, avg_epoch_loss

# --- Model Modifications for MPS ---

class MPSOptimizedEnhancedGPTLMHeadModel(nn.Module):
    """Optimized GPT model with MPS-specific enhancements"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpt_lm_head = GPTLMHeadModel(config)
        
        # Optional enhancement modules
        self.use_mla = config.use_mla
        if self.use_mla:
            self.mla = MLA(config.n_embd, config.mla_n_latent, config.n_head,
                           dropout_prob=config.dropout_prob, thinking_steps=1)

        self.use_reasoning_tracker = config.use_reasoning_tracker
        if self.use_reasoning_tracker:
            self.reasoning_tracker = ReasoningTracker(config.n_embd, reasoning_steps=config.reasoning_steps)

        self.use_algorithmic_reasoner = config.use_algorithmic_reasoner
        if self.use_algorithmic_reasoner:
            self.algorithmic_reasoner = AlgorithmicReasoner(config.n_embd,
                                                           num_registers=config.algorithmic_reasoner_registers,
                                                           max_steps=10)

        self.use_calculator = config.use_calculator
        if self.use_calculator:
            self.calculator = CalculatorTool()
    
    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False, position_ids=None):
        # MPS-optimized forward pass with explicit device handling
        
        # Memory check and cleanup
        if torch.backends.mps.is_available() and hasattr(torch.mps, 'empty_cache') and not use_cache:
            # Only clear cache if we're not using KV caching (which would be cleared)
            torch.mps.empty_cache()
        
        # Get base transformer hidden states
        hidden_states = self.gpt_lm_head.transformer(
            input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            position_ids=position_ids
        )

        # Apply MLA enhancement (optional)
        if self.use_mla and not use_cache:
            hidden_states = self.mla(hidden_states)

        # Apply Reasoning Tracker (optional)
        reasoning_output = None
        if self.use_reasoning_tracker and not use_cache:
            refined_states, _, confidence = self.reasoning_tracker(hidden_states)
            hidden_states = refined_states
            reasoning_output = {"reasoning_confidence": confidence}

        # Apply Algorithmic Reasoner (optional)
        algo_output = None
        if self.use_algorithmic_reasoner and not use_cache:
            final_registers, _ = self.algorithmic_reasoner(hidden_states)
            algo_output = {"algorithmic_registers": final_registers}

        # Final LM Head Projection
        logits = F.linear(hidden_states, self.gpt_lm_head.transformer.wte.weight)

        # Calculate Loss
        loss = None
        if labels is not None:
            # Make sure batch sizes match
            if logits.size(0) != labels.size(0):
                print(f"Warning: Batch size mismatch - logits: {logits.size(0)}, labels: {labels.size(0)}")
                min_batch = min(logits.size(0), labels.size(0))
                logits = logits[:min_batch]
                labels = labels[:min_batch]
            
            # Ensure sequence lengths match before shifting
            min_seq_len = min(logits.size(1), labels.size(1))
            logits = logits[:, :min_seq_len, :]
            labels = labels[:, :min_seq_len]
            
            # Now perform the shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute loss with adjusted tensors
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            try:
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            except ValueError as e:
                print(f"Error in loss calculation: {e}")
                # Provide a fallback loss to prevent training crash
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Combine outputs
        output_dict = {"loss": loss, "logits": logits, "hidden_states": hidden_states}
        if reasoning_output:
            output_dict.update(reasoning_output)
        if algo_output:
            output_dict.update(algo_output)

        return output_dict
        
    def generate(self, input_ids, attention_mask=None, max_new_tokens=50, temperature=0.8, 
                top_k=50, top_p=0.9, eos_token_id=None, do_sample=True):
        """Optimized generate method with proper MPS memory management"""
        # Clear MPS cache before generation
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Delegate to the base model's generate method with MPS optimizations
        return self.gpt_lm_head.transformer.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            do_sample=do_sample
        )
    
    def run_calculator(self, expression):
        """Interface to the calculator tool."""
        if not self.use_calculator:
            return "Calculator tool is disabled."
        result_dict = self.calculator.calculate(expression)
        return result_dict.get("result", f"Error: {result_dict.get('error', 'Unknown calculator error')}")

    def run_tot_search(self, prompt, tokenizer, generation_length=50):
        """Interface to run Tree of Thought search (if enabled and implemented)."""
        if not hasattr(self, 'tree_of_thought'):
             # Initialize ToT here if needed, requires tokenizer
             # Check if ToT is conceptually enabled in config, even if not used in fwd
             if hasattr(self.config, 'use_tree_of_thought') and self.config.use_tree_of_thought:
                  self.tree_of_thought = TreeOfThought(self, tokenizer)
             else:
                  return "Tree of Thought is not configured for this model."

        if not isinstance(self.tree_of_thought, TreeOfThought):
             return "Tree of Thought component not available or not initialized correctly."

        logger.info(f"Starting Tree of Thought search for prompt: '{prompt[:50]}...'")
        return self.tree_of_thought.search(prompt, generation_length=generation_length)

    def reset_reasoning_states(self):
        """Resets states of internal reasoning modules if they have state."""
        if self.use_reasoning_tracker and hasattr(self.reasoning_tracker, 'reset_state'):
            self.reasoning_tracker.reset_state()
        
        # Reset RWKV states if present in the model's transformer blocks
        for block in self.gpt_lm_head.transformer.blocks:
            if hasattr(block, 'rwkv_attn') and hasattr(block.rwkv_attn, 'reset_state'):
                block.rwkv_attn.reset_state()
        
        # Reset KV caches in all attention blocks
        for block in self.gpt_lm_head.transformer.blocks:
            if hasattr(block.attn, 'kv_cache') and block.attn.kv_cache is not None:
                block.attn.kv_cache.reset()
        
        logger.info("Reasoning module states reset.")

# --- Optimized Prompting Function ---

def prompt_model_optimized(
    model, tokenizer, prompt_text, max_new_tokens=100,
    temperature=0.7, top_k=40, top_p=0.9, do_sample=True,
    use_calculator_tool=True, calculator_trigger="[CALC:", calculator_end="]",
    device=None
):
    """Optimized prompting function with MPS-specific enhancements."""
    import torch
    import time
    
    # Set device if not provided
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    # Clear MPS cache before generation
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Record generation start time
    start_time = time.time()
    current_text = prompt_text
    generated_sequence = ""
    all_generated_ids = tokenizer.encode(current_text, return_tensors="pt").to(device)
    
    print(f"\n--- Prompting ---")
    print(f"Initial Prompt: {current_text}")
    
    active_generation_length = 0
    while active_generation_length < max_new_tokens:
        # Prepare inputs
        input_ids = all_generated_ids
        attention_mask = torch.ones_like(input_ids)
        
        # Determine generation step length
        generate_step_length = 1 if use_calculator_tool else max_new_tokens - active_generation_length
        
        # Generation step
        with torch.no_grad():
            # Generate optimally with or without tool usage
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=generate_step_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=do_sample,
            )
        
        # Extract only the newly generated token IDs
        new_token_ids = output_ids[0, input_ids.shape[1]:]
        if len(new_token_ids) == 0:
            break  # Generation stopped (e.g., EOS)
        
        # Decode the new token(s)
        new_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        generated_sequence += new_text
        current_text += new_text
        all_generated_ids = output_ids  # Update the full sequence for the next step
        
        active_generation_length += len(new_token_ids)
        
        # --- Tool Check: Calculator ---
        if use_calculator_tool and hasattr(model, 'use_calculator') and model.use_calculator and calculator_trigger in generated_sequence:
            start_idx = generated_sequence.rfind(calculator_trigger)
            if start_idx != -1:
                end_idx = generated_sequence.find(calculator_end, start_idx)
                if end_idx != -1:
                    expression = generated_sequence[start_idx + len(calculator_trigger):end_idx].strip()
                    print(f"[Tool Call Detected] Expression: {expression}")
                    
                    # Call calculator tool
                    calc_result = model.run_calculator(expression)
                    print(f"[Tool Result] Output: {calc_result}")
                    
                    # Inject result back into the context
                    generated_sequence = generated_sequence[:start_idx] + f" {calc_result} " + generated_sequence[end_idx + len(calculator_end):]
                    current_text = prompt_text + generated_sequence  # Rebuild current_text
                    
                    # Re-tokenize the updated text for the next generation step
                    all_generated_ids = tokenizer.encode(current_text, return_tensors="pt").to(device)
                    print(f"[Context Updated] New context tail: ...{current_text[-100:]}")
                    
                    # Clear MPS cache after tool usage
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
        
        # Check for EOS in the generated part
        if tokenizer.eos_token_id in new_token_ids:
            print("[EOS Detected]")
            break
        
        # Periodic memory cleanup during long generations
        if active_generation_length % 10 == 0 and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    # Generation timing
    generation_time = time.time() - start_time
    tokens_per_second = active_generation_length / generation_time if generation_time > 0 else 0
    
    print(f"\n--- Generation Complete ---")
    print(f"Generated {active_generation_length} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/sec)")
    
    # Return only the generated part, excluding the initial prompt
    final_response = tokenizer.decode(all_generated_ids[0, len(tokenizer.encode(prompt_text)):], skip_special_tokens=True)
    print(f"Final Response: {final_response}")
    
    # Final memory cleanup
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return final_response

# --- Add an optimized checkpoint saver ---

def save_checkpoint_periodically_optimized(model, save_path, interval_sec=120):
    """Periodically saves model state dict with MPS optimizations."""
    import os
    import threading
    import time
    import torch
    from pathlib import Path
    
    # Create directory if it doesn't exist
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    
    def save_func():
        while True:
            # Sleep first to give model time to initialize
            time.sleep(interval_sec)
            
            try:
                # Optional: Move to CPU before saving to avoid MPS issues
                if torch.backends.mps.is_available():
                    # Create temp copy of state dict on CPU
                    cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    torch.save(cpu_state_dict, save_path)
                    del cpu_state_dict  # Clean up memory
                    torch.mps.empty_cache()
                else:
                    # Standard save
                    torch.save(model.state_dict(), save_path)
                    
                logger.info(f"Checkpoint saved at {save_path}")
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")
    
    # Start daemon thread
    t = threading.Thread(target=save_func, daemon=True)
    t.start()
    logger.info(f"Started automatic checkpoint saving every {interval_sec} seconds")
    return t  # Return thread in case we need to manage it later

####################################################




# --- Configuration ---

# Setup logging
logging.basicConfig(
    filename='training_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Setting up device and reproducibility
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")

# Move model to device
torch.manual_seed(42)
if torch.cuda.is_available():
    # Note: Deterministic operations can sometimes affect performance.
    # Consider disabling if speed is critical and exact reproducibility isn't paramount.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False # Set False for determinism
    pass # Keeping defaults for now

# --- GPT Configuration Class ---

class GPTConfig:
    """Configuration class for the GPT model architecture."""
    def __init__(self, vocab_size: int, max_seq_len: int, n_embd: int, n_layer: int, n_head: int,
                 dropout_prob: float = 0.1, alibi: bool = False, use_rope: bool = True,
                 flash_attention: bool = True, n_kv_heads: int = None, use_gqa: bool = False,
                 use_rwkv: bool = False, use_ssm: bool = False,
                 use_moe: bool = False, num_experts: int = 8, top_k_experts: int = 2,
                 gradient_checkpointing: bool = False,
                 # Added for reasoning modules
                 mla_n_latent: int = 16, use_mla: bool = True,
                 reasoning_steps: int = 3, use_reasoning_tracker: bool = True,
                 algorithmic_reasoner_registers: int = 4, use_algorithmic_reasoner: bool = False,
                 # Added for tools
                 use_calculator: bool = True):

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_heads = n_kv_heads if (n_kv_heads is not None and use_gqa) else n_head
        self.dropout_prob = dropout_prob

        # Architectural choices
        self.alibi = alibi
        self.use_rope = use_rope and not alibi # RoPE and ALiBi are usually alternatives
        self.flash_attention = flash_attention and hasattr(F, 'scaled_dot_product_attention') # Check if available
        self.use_gqa = use_gqa and (n_kv_heads is not None) and (n_head % n_kv_heads == 0)
        self.use_rwkv = use_rwkv
        self.use_ssm = use_ssm
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.gradient_checkpointing = gradient_checkpointing

        # Enhanced features
        self.mla_n_latent = mla_n_latent
        self.use_mla = use_mla
        self.reasoning_steps = reasoning_steps
        self.use_reasoning_tracker = use_reasoning_tracker
        self.algorithmic_reasoner_registers = algorithmic_reasoner_registers
        self.use_algorithmic_reasoner = use_algorithmic_reasoner
        self.use_calculator = use_calculator

        # Validation
        if self.use_gqa:
            assert self.n_head % self.n_kv_heads == 0, "n_head must be divisible by n_kv_heads for GQA"
        if self.alibi and self.use_rope:
            logger.warning("Both ALiBi and RoPE are enabled. RoPE will be disabled.")
            self.use_rope = False
        if not self.flash_attention and hasattr(F, 'scaled_dot_product_attention'):
             logger.info("Flash Attention is available but disabled in config.")
        elif self.flash_attention and not hasattr(F, 'scaled_dot_product_attention'):
             logger.warning("Flash Attention requested but F.scaled_dot_product_attention not found. Disabling.")
             self.flash_attention = False


# --- Memory Optimization Techniques ---

# Helper function: Chunked matrix multiplication
def chunked_matmul(a: torch.Tensor, b: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    """
    Computes batched matrix multiplication between tensors 'a' and 'b' in smaller chunks
    along the sequence dimension to reduce peak memory usage.

    Args:
        a: Tensor of shape (B, n_heads, T_a, head_dim)
        b: Tensor of shape (B, n_heads, head_dim, T_b)
        chunk_size: Number of tokens (T_a dimension) to process in one chunk.

    Returns:
        A tensor of shape (B, n_heads, T_a, T_b)
    """
    B, n_heads, T_a, head_dim_a = a.shape
    _, _, head_dim_b, T_b = b.shape
    assert head_dim_a == head_dim_b, "Head dimensions must match"

    output_chunks = []
    for start in range(0, T_a, chunk_size):
        end = min(start + chunk_size, T_a)
        chunk_result = torch.matmul(a[:, :, start:end, :], b)  # (B, n_heads, chunk_size, T_b)
        output_chunks.append(chunk_result)
    return torch.cat(output_chunks, dim=2)

# KV Cache for efficient inference
class KVCache:
    """Stores past Key and Value tensors for efficient autoregressive decoding."""
    def __init__(self, max_batch_size: int, max_seq_len: int, n_kv_heads: int, head_dim: int, dtype=torch.float32, device: torch.device = None):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads # Use n_kv_heads for GQA compatibility
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Initialize empty cache tensors (on CPU initially, moved on first update)
        self.k_cache = torch.zeros(max_batch_size, n_kv_heads, max_seq_len, head_dim, dtype=dtype)
        self.v_cache = torch.zeros(max_batch_size, n_kv_heads, max_seq_len, head_dim, dtype=dtype)
        self.current_seq_len = 0
        self.initialized_device = False

    def _initialize_device(self, input_tensor: torch.Tensor):
        """Move cache to the same device as the input tensor on first use."""
        if not self.initialized_device:
            self.device = input_tensor.device
            self.k_cache = self.k_cache.to(self.device)
            self.v_cache = self.v_cache.to(self.device)
            self.initialized_device = True
            logger.info(f"KV cache initialized on device {self.device}, shape: {self.k_cache.shape}")

    def update(self, k: torch.Tensor, v: torch.Tensor, position: int = None):
        """
        Update the KV cache with new key-value pairs.

        Args:
            k: Key tensor of shape (B, n_kv_heads, T_new, head_dim)
            v: Value tensor of shape (B, n_kv_heads, T_new, head_dim)
            position: Start position index to update (defaults to appending).
                      Used for overwriting or specific insertion points.
        """
        self._initialize_device(k)
        B, n_kv_heads, T_new, head_dim = k.shape
        assert B <= self.max_batch_size, f"Batch size {B} exceeds max cache size {self.max_batch_size}"
        assert n_kv_heads == self.n_kv_heads, f"KV heads mismatch: got {n_kv_heads}, expected {self.n_kv_heads}"
        assert head_dim == self.head_dim, f"Head dimension mismatch: got {head_dim}, expected {self.head_dim}"

        if position is None:
            # Append to the end
            start_pos = self.current_seq_len
            end_pos = start_pos + T_new
        else:
            # Update at a specific position
            start_pos = position
            end_pos = start_pos + T_new
            # If this is a position we've already filled, log it:
            if start_pos < self.current_seq_len:
                logger.debug(f"KV Cache: Overwriting existing cache at position {start_pos} with {T_new} new tokens")

        # Check for cache overflow
        if end_pos > self.max_seq_len:
            # Implement a proper eviction strategy
            logger.warning(f"KV Cache overflow: max_seq_len={self.max_seq_len}, needed={end_pos}. Implementing sliding window.")
            
            # Calculate how many tokens we need to discard
            to_discard = end_pos - self.max_seq_len
            
            # Shift existing cache content left, discarding oldest tokens
            self.k_cache[:B, :, :-to_discard] = self.k_cache[:B, :, to_discard:].clone()
            self.v_cache[:B, :, :-to_discard] = self.v_cache[:B, :, to_discard:].clone()
            
            # Adjust positions for insertion
            start_pos = self.max_seq_len - T_new
            end_pos = self.max_seq_len
            
            # Update current_seq_len to reflect full cache
            self.current_seq_len = self.max_seq_len
        
        # Validate positions to avoid out-of-bounds
        if start_pos < 0 or end_pos > self.max_seq_len:
            raise ValueError(f"Invalid cache positions: start={start_pos}, end={end_pos}, max={self.max_seq_len}")
            
        # Update the cache
        self.k_cache[:B, :, start_pos:end_pos] = k
        self.v_cache[:B, :, start_pos:end_pos] = v

        # Update current_seq_len if we extended the sequence
        if position is None or end_pos > self.current_seq_len:
             self.current_seq_len = max(self.current_seq_len, end_pos)
             logger.debug(f"KV Cache: Updated sequence length to {self.current_seq_len}")

    def get(self, current_batch_size: int):
        """
        Retrieve the current key-value pairs from the cache up to current_seq_len.

        Args:
            current_batch_size: The actual batch size being processed (<= max_batch_size).

        Returns:
            Tuple of (k_cache, v_cache) for the current sequence length and batch size.
            Shape: (B, n_kv_heads, current_seq_len, head_dim)
        """
        if not self.initialized_device or self.current_seq_len == 0:
             # Return empty tensors if cache hasn't been used yet
             return (torch.zeros(current_batch_size, self.n_kv_heads, 0, self.head_dim, dtype=self.dtype, device=self.device),
                     torch.zeros(current_batch_size, self.n_kv_heads, 0, self.head_dim, dtype=self.dtype, device=self.device))
        
        # Validate batch size
        if current_batch_size > self.max_batch_size:
            raise ValueError(f"Requested batch size {current_batch_size} exceeds max cache size {self.max_batch_size}")
            
        # Return the filled portion of the cache
        return (self.k_cache[:current_batch_size, :, :self.current_seq_len],
                self.v_cache[:current_batch_size, :, :self.current_seq_len])

    def reset(self, batch_size: int = None):
        """Reset the cache, optionally resizing batch dimension if needed."""
        if self.initialized_device:
            if batch_size is not None and batch_size != self.k_cache.shape[0]:
                 # Reinitialize if batch size changes
                 logger.info(f"Resetting KV Cache and resizing batch dim from {self.max_batch_size} to {batch_size}")
                 self.max_batch_size = batch_size
                 self.k_cache = torch.zeros(self.max_batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim, 
                                           dtype=self.dtype, device=self.device)
                 self.v_cache = torch.zeros(self.max_batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim, 
                                           dtype=self.dtype, device=self.device)
            else:
                # Just clear the existing cache
                self.k_cache.zero_()
                self.v_cache.zero_()
                logger.debug(f"KV Cache: Reset cache with same dimensions")
        self.current_seq_len = 0

# --- Core Model Architecture Components ---

# RMSNorm (Root Mean Square Layer Normalization)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5): # Adjusted default epsilon
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        eps = max(self.eps, 1e-5)
        # Compute Root Mean Square along the feature dimension
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x) # Cast to float for norm, then back
        return output * self.weight

# SwiGLU Feed-Forward Block
class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, expansion_factor: float = 8/3, dropout_prob: float = 0.0):
        super().__init__()
        intermediate_dim = int(expansion_factor * hidden_dim)
        multiple_of = 256
        intermediate_dim = multiple_of * ((intermediate_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add checks for NaN and clipping for stability
        w1_out = self.w1(x)
        w3_out = self.w3(x)
        
        # Clip activations to prevent overflow
        w1_out = torch.clamp(w1_out, min=-20.0, max=20.0)
        w3_out = torch.clamp(w3_out, min=-20.0, max=20.0)
        
        swish_gate = F.silu(w1_out)
        gated = swish_gate * w3_out
        
        # Clip again before final projection
        gated = torch.clamp(gated, min=-50.0, max=50.0)
        output = self.w2(gated)
        
        return output
# ALiBi Positional Bias (Attention with Linear Biases)
def build_alibi_tensor(n_heads: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Builds the ALiBi tensor for biasing attention scores."""
    def get_slopes(n: int) -> list[float]:
        # Replicate the slope calculation from original ALiBi paper/implementations
        try:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            base = torch.tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=torch.float32)
            powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.float32)
            slopes = torch.pow(base, powers)

            if closest_power_of_2 != n: # Handle cases where n is not a power of 2
                extra_base = torch.tensor(2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=torch.float32)
                extra_powers = torch.arange(1, 1 + 2 * (n - closest_power_of_2), 2, dtype=torch.float32)
                slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

            return slopes.tolist() # Return as list
        except Exception as e:
            # Fallback if calculation fails
            logger.error(f"Error calculating ALiBi slopes: {e}. Using simpler power-of-2 calculation.")
            start = 2 ** (-8.0 / n)
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

    slopes = torch.tensor(get_slopes(n_heads), device=device, dtype=dtype)
    # Create relative distances (causal mask style)
    relative_position = torch.arange(seq_len, device=device)[:, None] - torch.arange(seq_len, device=device)[None, :]
    # ALiBi requires negative relative positions for the bias calculation
    alibi = slopes.unsqueeze(1).unsqueeze(2) * relative_position.unsqueeze(0) # Shape: (n_heads, seq_len, seq_len)
    # Add batch dimension placeholder (broadcasts)
    return alibi.unsqueeze(0) # Shape: (1, n_heads, seq_len, seq_len)


# --- Modern Positional Embeddings ---

# Rotary Position Embeddings (RoPE)
class RotaryPositionEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000, device: torch.device = None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.device = device # Store device for cache creation

        # Calculate inverse frequencies (theta_i)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=self.device).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # Outer product: (seq_len, dim/2)
        
        # Store frequencies directly (no duplication needed)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False) # Shape: (seq_len, dim/2)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False) # Shape: (seq_len, dim/2)

    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Applies rotary embeddings to the input tensor."""
        # Get dimensions
        *others, seq_len, head_dim = x.shape
        dim_half = head_dim // 2
        
        # Reshape x to separate real/imaginary parts
        x_reshaped = x.reshape(*others, seq_len, dim_half, 2)
        x_real, x_imag = x_reshaped.unbind(-1)  # (..., seq_len, dim_half) each
        
        # Apply rotation using complex number multiplication formula
        rotated_real = x_real * cos - x_imag * sin  # (..., seq_len, dim_half)
        rotated_imag = x_real * sin + x_imag * cos  # (..., seq_len, dim_half)
        
        # Combine back and reshape to original form
        rotated_x = torch.stack((rotated_real, rotated_imag), dim=-1)
        return rotated_x.reshape(*others, seq_len, head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies RoPE to Query and Key tensors.

        Args:
            q: Query tensor (..., seq_len, head_dim)
            k: Key tensor (..., seq_len, head_dim)
            seq_len: The current sequence length of q and k.

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        # Check if cache needs rebuilding (e.g., for longer sequences than cached)
        if seq_len > self.max_seq_len:
            logger.warning(f"Sequence length {seq_len} > RoPE cache {self.max_seq_len}. Rebuilding cache.")
            self._build_cache(seq_len)

        # Retrieve cached cos/sin values for the current sequence length
        # Slicing ensures we only use the relevant part of the cache
        cos = self.cos_cached[:seq_len] # Shape: (seq_len, dim)
        sin = self.sin_cached[:seq_len] # Shape: (seq_len, dim)

        # Apply rotary embeddings to q and k
        # Unsqueeze cos/sin to match tensor dimensions (e.g., B, n_heads, T, dim -> B, 1, T, dim)
        # Assumes q/k have shape (B, n_heads, T, head_dim) or (B, T, head_dim)
        cos = cos.unsqueeze(0) # Add batch/head dim placeholder
        sin = sin.unsqueeze(0) # Add batch/head dim placeholder
        if q.ndim == 4: # Handle (B, n_heads, T, head_dim)
             cos = cos.unsqueeze(1)
             sin = sin.unsqueeze(1)

        q_embed = self._apply_rotary_emb(q, cos, sin)
        k_embed = self._apply_rotary_emb(k, cos, sin)

        return q_embed, k_embed


# --- Advanced Attention Mechanisms ---

# Base Attention Class (to share common logic)
class BaseAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        self.dropout_prob = config.dropout_prob
        self.use_flash_attention = config.flash_attention
        self.use_rope = config.use_rope
        self.use_alibi = config.alibi
        self.max_seq_len = config.max_seq_len

        if self.use_rope:
            # Initialize RoPE on the correct device if possible
            rope_device = device if torch.cuda.is_available() else torch.device('cpu')
            self.rope = RotaryPositionEmbeddings(self.head_dim, self.max_seq_len, device=rope_device)

        # KV cache needs n_kv_heads
        self.n_kv_heads = config.n_kv_heads if config.use_gqa else config.n_head
        self.kv_cache: KVCache | None = None # Initialize later if use_cache=True

        # Output projection and dropout common to all attention types
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout_prob)
        self.resid_dropout = nn.Dropout(self.dropout_prob) # Dropout after residual connection

    def _init_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """Initializes the KV cache if needed."""
        if self.kv_cache is None:
            self.kv_cache = KVCache(batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim, dtype=dtype, device=device)
        else:
            # Reset if batch size changes, or just reset content
            self.kv_cache.reset(batch_size=batch_size)

    def _process_kv_cache(self, k: torch.Tensor, v: torch.Tensor, use_cache: bool, position_ids: torch.LongTensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Updates and retrieves from KV cache."""
        if use_cache:
            B, _, T_new, _ = k.shape
            self._init_kv_cache(B, k.dtype, k.device)

            # Determine the position for cache update
            cache_position = None
            if position_ids is not None:
                if position_ids.shape[1] == 1:  # Single token generation
                    cache_position = position_ids[0, 0].item()
                else:  # Multi-token processing (e.g., initial prompt)
                    # We'll just append sequentially in this case
                    pass
            
            # Update cache with new keys/values
            self.kv_cache.update(k, v, position=cache_position)
            
            # Get the full sequence from cache
            k_full, v_full = self.kv_cache.get(current_batch_size=B)
            return k_full, v_full
        else:
            return k, v
    def _manual_attention(self, q, k, v, attention_mask, alibi_bias, is_causal):
        """Manual attention computation as fallback with improved numerical stability."""
        B, n_heads, T_q, head_dim = q.shape
        T_k = k.shape[-2]
        
        # Save original dtype for later conversion
        input_dtype = q.dtype
        
        # Convert to float32 for better numerical stability
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        
        # Scale factor
        scale_factor = 1.0 / math.sqrt(head_dim)
        
        # Compute attention scores with careful handling of numerical ranges
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor # (B, n_heads, T_q, T_k)
        
        # Add basic check for NaN scores
        if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
            print(f"NaN/Inf in attention scores after q@k: min={attn_scores.min().item() if not torch.isnan(attn_scores.min()) else 'nan'}, "
                f"max={attn_scores.max().item() if not torch.isnan(attn_scores.max()) else 'nan'}")
            attn_scores = torch.nan_to_num(attn_scores, nan=0.0, posinf=1.0, neginf=-1.0) 
        
        # Apply ALiBi bias if needed
        if self.use_alibi and alibi_bias is not None:
            # Ensure bias matches score dimensions (T_q, T_k)
            if alibi_bias.size(-2) >= T_q and alibi_bias.size(-1) >= T_k:
                alibi_bias = alibi_bias[:, :, :T_q, :T_k]
                # Ensure alibi bias is the same dtype as attention scores
                alibi_bias = alibi_bias.to(torch.float32)
                attn_scores = attn_scores + alibi_bias
            else:
                logger.warning(f"ALiBi bias shape {alibi_bias.shape} incompatible with attention scores {attn_scores.shape}. Skipping ALiBi.")
        
        # Apply causal mask if needed for sequences longer than 1 token
        if is_causal and T_q > 1:
            # Create causal mask dynamically
            causal_mask = torch.ones(T_q, T_k, device=q.device, dtype=torch.bool).tril(diagonal=0)
            attn_scores = attn_scores.masked_fill(~causal_mask, -10000.0)  # Use finite value instead of -inf
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Handle different mask shapes
            if attention_mask.dim() == 2:  # (B, T_k)
                expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_k)
                
                # Ensure mask has enough positions
                if expanded_mask.size(-1) < T_k:
                    # Pad with ones (allow attention)
                    padded_mask = torch.ones(B, 1, 1, T_k, device=attention_mask.device)
                    padded_mask[:, :, :, :expanded_mask.size(-1)] = expanded_mask
                    expanded_mask = padded_mask
                elif expanded_mask.size(-1) > T_k:
                    # Truncate to needed length
                    expanded_mask = expanded_mask[:, :, :, :T_k]
                    
                attn_scores = attn_scores.masked_fill(expanded_mask == 0, -10000.0)  # Use finite value
            elif attention_mask.dim() == 4:  # Already (B, 1, 1, T_k) or similar
                # Ensure compatible shapes
                if attention_mask.size(-1) != T_k:
                    logger.warning(f"Attention mask shape {attention_mask.shape} incompatible with key length {T_k}.")
                    # Handle this case based on your needs
                    if attention_mask.size(-1) < T_k:
                        # Pad
                        padded_mask = torch.ones(B, attention_mask.size(1), attention_mask.size(2), T_k, 
                                            device=attention_mask.device)
                        padded_mask[:, :, :, :attention_mask.size(-1)] = attention_mask
                        attention_mask = padded_mask
                    else:
                        # Truncate
                        attention_mask = attention_mask[:, :, :, :T_k]
                
                attn_scores = attn_scores.masked_fill(attention_mask == 0, -10000.0)  # Use finite value
        
        # Use stable softmax with careful handling
        max_scores = torch.max(attn_scores, dim=-1, keepdim=True)[0]
        attn_scores = attn_scores - max_scores  # Subtract max for numerical stability
        
        # Add safety clipping to prevent overflow
        attn_scores = torch.clamp(attn_scores, min=-15.0, max=15.0)
        
        # Compute softmax
        attn_probs = torch.exp(attn_scores)
        attn_sum = torch.sum(attn_probs, dim=-1, keepdim=True)
        
        # Avoid division by zero
        attn_sum = torch.clamp(attn_sum, min=1e-6)
        
        # Calculate normalized weights
        attn_weights = attn_probs / attn_sum
        
        # Apply dropout
        attn_weights = self.attn_dropout(attn_weights)
        
        # Check for NaN in weights
        if torch.isnan(attn_weights).any():
            print("NaN detected in attention weights after softmax")
            attn_weights = torch.nan_to_num(attn_weights, nan=1.0/T_k)  # Replace NaNs with uniform attention
        
        # Compute output
        attn_output = torch.matmul(attn_weights, v)
        
        # Final safety check
        if torch.isnan(attn_output).any():
            print("NaN detected in attention output")
            attn_output = torch.nan_to_num(attn_output, nan=0.0)
        
        # Convert back to original dtype
        attn_output = attn_output.to(input_dtype)
        
        return attn_output
    def _compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    attention_mask: torch.Tensor | None, alibi_bias: torch.Tensor | None) -> torch.Tensor:
        """Computes attention output using FlashAttention or manual implementation."""
        B, n_heads, T_q, head_dim = q.shape
        T_k = k.shape[-2]

        # Determine if causal masking is needed
        is_causal = T_q == T_k # Simple check: causal if query/key seq lengths match
        
        # For generation with KV cache: special case where query length is 1
        is_generation = T_q == 1 and T_k > 1

        # Add robust stability check and fix - Handle NaN/Inf in inputs
        if torch.isnan(q).any() or torch.isinf(q).any():
            print(f"NaN/Inf in attention q: min={q.min().item() if not torch.isnan(q.min()) else 'nan'}, "
                f"max={q.max().item() if not torch.isnan(q.max()) else 'nan'}")
            # Replace NaN/Inf with zeros and clamp to reasonable range
            q = torch.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0)
            q = torch.clamp(q, min=-1.0, max=1.0)  # Add reasonable bounds
            
        if torch.isnan(k).any() or torch.isinf(k).any():
            print(f"NaN/Inf in attention k: min={k.min().item() if not torch.isnan(k.min()) else 'nan'}, "
                f"max={k.max().item() if not torch.isnan(k.max()) else 'nan'}")
            k = torch.nan_to_num(k, nan=0.0, posinf=1.0, neginf=-1.0)
            k = torch.clamp(k, min=-1.0, max=1.0)
            
        if torch.isnan(v).any() or torch.isinf(v).any():
            print(f"NaN/Inf in attention v: min={v.min().item() if not torch.isnan(v.min()) else 'nan'}, "
                f"max={v.max().item() if not torch.isnan(v.max()) else 'nan'}")
            v = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
            v = torch.clamp(v, min=-1.0, max=1.0)

        # Use float32 for attention calculation for better numerical stability
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        
        if self.use_flash_attention and not self.use_alibi and attention_mask is None:
            # Transpose for flash attention
            q = q.transpose(1, 2) # (B, T_q, n_heads, head_dim)
            k = k.transpose(1, 2) # (B, T_k, n_kv_heads, head_dim)
            v = v.transpose(1, 2) # (B, T_k, n_kv_heads, head_dim)

            # During generation with KV cache, we don't need causal masking
            # as the query is only one token that should attend to all previous tokens
            if is_generation:
                is_causal = False
                
            try:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None, 
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=is_causal,
                    scale=1.0 / math.sqrt(head_dim)  # Explicitly set scale
                )
            except RuntimeError as e:
                # Fallback to manual attention when flash attention fails
                logger.warning(f"Flash attention failed with shapes Q:{q.shape}, K:{k.shape}. Using manual attention.")
                # Go to manual implementation
                q = q.transpose(1, 2)  # Back to (B, n_heads, T_q, head_dim)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                return self._manual_attention(q, k, v, attention_mask, alibi_bias, is_causal)

            # --- Insert Fix for mismatched sequence lengths ---
            T_q = q.shape[1]
            T_k = k.shape[1]
            if T_q != T_k:
                if T_q < T_k:
                    k = k[:, -T_q:, :, :]
                    v = v[:, -T_q:, :, :]
                else:
                    raise ValueError(f"Query length (T_q={T_q}) is longer than Key length (T_k={T_k}).")
            # --- End Fix ---

            attn_output = attn_output.transpose(1, 2) # (B, n_heads, T_q, head_dim)
            
            # Convert back to original dtype before returning
            attn_output = attn_output.type_as(input_dtype)
            return attn_output
        else:
            # Manual attention calculation
            return self._manual_attention(q, k, v, attention_mask, alibi_bias, is_causal)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the forward method")


# MultiHead Self Attention (Standard)
class MultiHeadSelfAttention(BaseAttention):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        assert not config.use_gqa, "MHA should not be used when GQA is enabled"
        assert config.n_embd % config.n_head == 0

        # Projections for Q, K, V
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
                alibi_bias: torch.Tensor | None = None, use_cache: bool = False,
                position_ids: torch.LongTensor | None = None) -> torch.Tensor:
        B, T, C = x.size()

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled (before caching)
        current_seq_len = k.shape[-2] # Sequence length of the *new* tokens
        if self.use_rope:
            # Pass the actual sequence length for RoPE calculation
            q, k = self.rope(q, k, current_seq_len)

        # Handle KV Caching
        k, v = self._process_kv_cache(k, v, use_cache, position_ids)

        # Compute attention
        attn_output = self._compute_attention(q, k, v, attention_mask, alibi_bias)

        # Combine heads and project output
        # after transpose, shape is (B, T_out, n_head, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, -1, C)   # let PyTorch infer the correct sequence length

        output = self.resid_dropout(self.out_proj(attn_output))
        return output


# Grouped Query Attention (GQA)
class GroupedQueryAttention(BaseAttention):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        assert config.use_gqa, "GQA must be enabled in config"
        assert config.n_head % config.n_kv_heads == 0
        self.n_kv_heads = config.n_kv_heads
        self.num_query_groups = config.n_head // self.n_kv_heads

        # Projections: Q maps to n_head, K/V map to n_kv_heads
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False) # n_head * head_dim
        self.k_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeats KV heads to match query heads: (B, n_kv, T, head_dim) -> (B, n_q, T, head_dim)"""
        B, n_kv, T, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x.unsqueeze(2) # (B, n_kv, 1, T, head_dim)
            .expand(B, n_kv, n_rep, T, head_dim) # (B, n_kv, n_rep, T, head_dim)
            .reshape(B, n_kv * n_rep, T, head_dim) # (B, n_q, T, head_dim)
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
                alibi_bias: torch.Tensor | None = None, use_cache: bool = False,
                position_ids: torch.LongTensor | None = None) -> torch.Tensor:
        B, T, C = x.size()

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape Q: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Reshape K/V: (B, T, C_kv) -> (B, n_kv_head, T, head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled (before caching)
        current_seq_len = k.shape[-2] # Sequence length of the *new* tokens
        if self.use_rope:
            q, k = self.rope(q, k, current_seq_len)

        # Handle KV Caching (operates on the n_kv_heads dimension)
        k, v = self._process_kv_cache(k, v, use_cache, position_ids)
        # Note: k, v retrieved from cache will have shape (B, n_kv_heads, T_full, head_dim)

        # Repeat K/V heads to match query heads for attention calculation
        k_rep = self._repeat_kv(k, self.num_query_groups) # (B, n_head, T_full, head_dim)
        v_rep = self._repeat_kv(v, self.num_query_groups) # (B, n_head, T_full, head_dim)

        # Compute attention using repeated K/V
        attn_output = self._compute_attention(q, k_rep, v_rep, attention_mask, alibi_bias)

        # Combine heads and project output
        # after transpose, shape is (B, T_out, n_head, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, -1, C)   # let PyTorch infer the correct sequence length

        output = self.resid_dropout(self.out_proj(attn_output))
        return output


# RWKV-style Linear Attention (Simplified)
class RWKVAttention(nn.Module):
    """Simplified RWKV-style time-mixing block (operates per-channel)."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd

        # Learnable time decay and first token bias (per channel)
        # Initialize decay close to zero for stability
        self.time_decay = nn.Parameter(torch.ones(self.n_embd) * -5.0)
        self.time_first = nn.Parameter(torch.ones(self.n_embd) * 0.1)

        # Projections (simplified compared to full RWKV block)
        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.output = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # State buffers (initialized on first forward)
        self.register_buffer('state_a', torch.Tensor(), persistent=False) # Numerator state
        self.register_buffer('state_b', torch.Tensor(), persistent=False) # Denominator state (exp(k))
        self.register_buffer('state_p', torch.Tensor(), persistent=False) # Max K state for numeric stability
        self.state_initialized = False

    def _init_state(self, B: int, dtype: torch.dtype, device: torch.device):
        self.state_a = torch.zeros(B, self.n_embd, dtype=dtype, device=device)
        self.state_b = torch.zeros(B, self.n_embd, dtype=dtype, device=device)
        # Initialize p to -inf for correct max calculation on first step
        self.state_p = torch.full((B, self.n_embd), -float('inf'), dtype=dtype, device=device)
        self.state_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Initialize state if needed or if batch size/device changes
        if not self.state_initialized or self.state_a.shape[0] != B or self.state_a.device != x.device:
            self._init_state(B, x.dtype, x.device)

        # Project inputs
        k = self.key(x)   # (B, T, C)
        v = self.value(x) # (B, T, C)
        r = torch.sigmoid(self.receptance(x)) # (B, T, C) - Receptance gate

        # Time parameters (ensure correct shape and type)
        # Clamp decay to avoid instability
        time_decay = -torch.exp(torch.clamp(self.time_decay.float(), max=5.0)).type_as(x) # (C,) -> (1, 1, C)
        time_first = self.time_first.type_as(x) # (C,) -> (1, 1, C)

        outputs = []
        # Use local variables for state to avoid modifying buffers directly in loop
        state_a, state_b, state_p = self.state_a, self.state_b, self.state_p

        # Recurrent calculation over time steps
        for t in range(T):
            kt, vt, rt = k[:, t], v[:, t], r[:, t] # (B, C)

            # WKV calculation (stable version using max trick)
            max_p = torch.maximum(state_p, kt) # Max(prev_max, current_k)
            exp_kt_minus_p = torch.exp(kt - max_p)
            exp_prev_a = torch.exp(state_a + state_p - max_p)

            wkv = (exp_kt_minus_p * vt + exp_prev_a * state_b) / \
                  (exp_kt_minus_p + exp_prev_a * torch.exp(state_a)) # Denominator needs adjustment? Review RWKV details.
                  # Simpler: Use the denominator state `state_b` directly? Let's try the paper's formulation
            # Re-evaluating based on common implementations:
            wkv_num = (exp_kt_minus_p * vt) + (exp_prev_a * state_b)
            wkv_den = exp_kt_minus_p + exp_prev_a # Simpler denominator often used

            # Update states for next step
            # Apply time_first only for the very first token (t=0) effectively
            decay_factor = time_decay + (time_first if t == 0 else 0)

            state_a = decay_factor + state_a # Update log-decay state
            state_b = (torch.exp(state_a) * state_b) + (exp_kt_minus_p * vt) # Update numerator state
            state_p = max_p # Update max_k state

            # Apply receptance gate
            outputs.append(rt * wkv_num / wkv_den) # Apply gating to the result

        # Update persistent state buffers after loop
        self.state_a, self.state_b, self.state_p = state_a, state_b, state_p

        output = torch.stack(outputs, dim=1) # (B, T, C)
        return self.output(output)

    def reset_state(self):
        """Reset the recurrent state."""
        self.state_initialized = False
        # Clear tensor data if they exist
        self.state_a = torch.Tensor()
        self.state_b = torch.Tensor()
        self.state_p = torch.Tensor()


# --- State Space Model (Mamba-style Selective SSM) ---

class SelectiveSSM(nn.Module):
    """Simplified Selective State Space Model inspired by Mamba."""
    def __init__(self, hidden_dim: int, ssm_state_dim: int = 16, ssm_expand_factor: int = 2,
                 dt_rank: str | int = 'auto', dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ssm_state_dim = ssm_state_dim
        self.expand_factor = ssm_expand_factor
        self.expanded_dim = hidden_dim * ssm_expand_factor

        # Input projections (x -> B, C, dt)
        self.in_proj = nn.Linear(hidden_dim, 2 * self.expanded_dim + self.expanded_dim, bias=False) # Project to z, x, dt

        # Selective parameters (data-dependent)
        self.x_proj_weight = nn.Parameter(torch.randn(self.expanded_dim, self.expanded_dim))

        # Time step parameterization
        if dt_rank == 'auto':
            dt_rank = math.ceil(self.expanded_dim / 16)
        self.dt_rank = dt_rank
        self.dt_proj = nn.Linear(self.expanded_dim, dt_rank + 2 * self.ssm_state_dim, bias=True) # Project to dt, B, C components

        # Initialize dt bias for stability
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.bias, dt_scale)
        else: # random
             nn.init.uniform_(self.dt_proj.bias, -dt_scale, dt_scale)


        # State space parameters (A, B, C, D) - A is diagonal, B/C are dense
        self.A_log = nn.Parameter(torch.log(torch.arange(1, ssm_state_dim + 1, dtype=torch.float32)).repeat(self.expanded_dim, 1))
        self.B = nn.Parameter(torch.randn(self.expanded_dim, dt_rank, ssm_state_dim)) # Used with dt_proj output
        self.C = nn.Parameter(torch.randn(self.expanded_dim, self.ssm_state_dim))
        self.D = nn.Parameter(torch.ones(self.expanded_dim)) # Direct feedthrough

        # Output projection
        self.out_proj = nn.Linear(self.expanded_dim, hidden_dim, bias=False)

        # Store min/max dt
        self.dt_min = dt_min
        self.dt_max = dt_max

    def _selective_scan(self, u, delta, A, B, C, D):
        """Performs the selective scan operation (simplified parallel version)."""
        # u: (B, T, E) input
        # delta: (B, T, E) time step
        # A: (E, N) diagonal A matrix (log space)
        # B: (B, T, E, N) time-varying B matrix
        # C: (B, T, E, N) time-varying C matrix
        # D: (E) skip connection D
        B, T, E = u.shape
        N = A.shape[-1] # State dim

        # Discretize A and B using Zero-Order Hold (ZOH)
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)) # (B, T, E, N)
        delta_B_u = delta.unsqueeze(-1) * B * u.unsqueeze(-1) # (B, T, E, N)

        # Approximate parallel scan using cumulative product/sum
        # This is a simplification; true parallel scan is more complex.
        # Using a simple cumulative sum for demonstration; replace with optimized scan if needed.
        h = torch.zeros(B, E, N, device=u.device, dtype=u.dtype) # Initial state
        ys = []
        for t in range(T):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            y = torch.einsum('ben,ben->be', h, C[:, t]) # (B, E)
            ys.append(y)

        y = torch.stack(ys, dim=1) # (B, T, E)
        y = y + u * D.unsqueeze(0).unsqueeze(0) # Add skip connection
        return y


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        E = self.expanded_dim
        N = self.ssm_state_dim

        # --- Input Projections ---
        # Project x to z (gate), x (main), and components for dt, B, C
        z_x_dtbc = self.in_proj(x) # (B, T, 2*E + E)
        z, x_dtbc = z_x_dtbc.split([E, E + E], dim=-1) # z:(B,T,E), x_dtbc:(B,T,2E)

        # --- Compute Selective Parameters (dt, B, C) ---
        # Project x_dtbc to get raw values for dt, B, C parameters
        # Note: In full Mamba, x influences dt, B, C directly. Here simplified.
        dt_B_C_params = self.dt_proj(x) # Project x to get params (B, T, dt_rank + 2*N)
        dt_raw, B_params, C_params = dt_B_C_params.split([self.dt_rank, N, N], dim=-1) # dt:(B,T,rank), B:(B,T,N), C:(B,T,N)

        # Calculate dt (time step) with activation and constraints
        delta = F.softplus(dt_raw) # Ensure positivity
        delta = torch.clamp(delta * (self.dt_max - self.dt_min) + self.dt_min, min=self.dt_min, max=self.dt_max) # Clamp to range
        delta = delta.unsqueeze(-1).expand(-1, -1, E) # Expand delta to match expanded dim (B, T, E)


        # Calculate time-varying B and C (simplified)
        # B_tv = B_params.unsqueeze(2).expand(-1, -1, E, -1) # (B, T, E, N) - needs adjustment
        # C_tv = C_params.unsqueeze(2).expand(-1, -1, E, -1) # (B, T, E, N) - needs adjustment
        # Proper calculation involves dt_rank intermediate projection - simplified here:
        # Let's assume B and C are projected from x per time step
        B_tv = torch.einsum('btd,edn->betn', x, self.B) # Example projection (B, T, E, N)
        C_tv = self.C.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1) # Non-time varying C for simplicity (B, T, E, N)

        # --- State Space Calculation ---
        # Get diagonal A matrix (fixed)
        A = -torch.exp(self.A_log.float()) # (E, N)

        # Apply SSM scan
        # Input u to scan is element-wise product of x and gate z
        u = x * F.silu(z) # Use main projection x and gate z (B, T, E)
        y = self._selective_scan(u, delta, A, B_tv, C_tv, self.D)

        # --- Output Projection ---
        output = self.out_proj(y) # (B, T, D)
        return output


# --- Mixture of Experts Layer ---

class SparseMoE(nn.Module):
    """Sparse Mixture of Experts layer routes tokens to a subset of experts."""
    def __init__(self, hidden_dim: int, num_experts: int = 8, top_k: int = 2,
                 capacity_factor: float = 1.25, noisy_gating: bool = True, router_bias: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.noisy_gating = noisy_gating

        # Router network (learns which experts to send tokens to)
        self.router = nn.Linear(hidden_dim, num_experts, bias=router_bias)
        # Optional noise for gating during training
        self.noise_layer = nn.Linear(hidden_dim, num_experts, bias=router_bias) if noisy_gating else None

        # Expert networks (using SwiGLU as the FFN)
        self.experts = nn.ModuleList([
            SwiGLU(hidden_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D) # Flatten: (B*T, D)
        num_tokens = x_flat.shape[0]

        # --- Routing ---
        router_logits = self.router(x_flat) # (num_tokens, num_experts)

        # Add noise during training for better load balancing (optional)
        if self.noisy_gating and self.training and self.noise_layer is not None:
             noise_logits = self.noise_layer(x_flat)
             noise = torch.randn_like(router_logits) * F.softplus(noise_logits)
             router_logits = router_logits + noise

        # Get top-k experts and their weights (softmax over top-k)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).type_as(x) # (num_tokens, top_k)

        # --- Dispatch Tokens ---
        # Calculate expert capacity
        # Each token goes to top_k experts, scale by capacity factor
        tokens_per_expert = int(num_tokens * self.top_k / self.num_experts * self.capacity_factor)
        tokens_per_expert = max(tokens_per_expert, 1) # Ensure minimum capacity

        # Create dispatch mask and combine weights
        # Combine weights assigns the softmax score to the chosen expert index
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts) # (num_tokens, top_k, num_experts)
        combine_weights = (routing_weights.unsqueeze(-1) * expert_mask).sum(dim=1) # (num_tokens, num_experts) - assigns weights to chosen experts

        # Calculate load per expert (how many tokens want to go to each expert)
        dispatch_mask = combine_weights.bool() # (num_tokens, num_experts)
        expert_load = dispatch_mask.sum(dim=0) # (num_experts,)

        # --- Expert Computation with Capacity ---
        final_output = torch.zeros_like(x_flat)
        # Use permutation to handle capacity efficiently (similar to Fairseq/Megatron-LM)
        # Sort tokens based on which expert they are assigned to (by router weights)
        route_prob_max, route_idx_max = combine_weights.max(dim=1) # Get highest probability assignment
        sorted_indices = route_idx_max.argsort() # Sort tokens by their primary expert choice

        # Reorder data based on sorted indices
        x_sorted = x_flat[sorted_indices]
        combine_weights_sorted = combine_weights[sorted_indices]

        # Process experts sequentially, handling capacity
        current_pos = 0
        for i in range(self.num_experts):
            # Find tokens assigned to this expert in the sorted list
            num_assigned = expert_load[i].item()
            expert_tokens = x_sorted[current_pos : current_pos + num_assigned]

            # Enforce capacity
            if expert_tokens.shape[0] > tokens_per_expert:
                # If overloaded, keep only the top tokens_per_expert based on routing weights
                expert_probs = combine_weights_sorted[current_pos : current_pos + num_assigned, i]
                keep_indices = expert_probs.topk(tokens_per_expert, dim=0).indices
                expert_tokens = expert_tokens[keep_indices]
                num_processed = tokens_per_expert
            else:
                 num_processed = expert_tokens.shape[0]


            if num_processed > 0:
                 # Compute expert output
                 expert_output = self.experts[i](expert_tokens)

                 # Get the corresponding weights for the processed tokens
                 processed_weights = combine_weights_sorted[current_pos : current_pos + num_assigned][keep_indices if num_processed == tokens_per_expert else slice(None), i]

                 # Apply weights and add to final output (use scatter_add for correct placement)
                 # We need the original indices of the processed tokens
                 original_indices = sorted_indices[current_pos : current_pos + num_assigned][keep_indices if num_processed == tokens_per_expert else slice(None)]
                 final_output.scatter_add_(0, original_indices.unsqueeze(1).expand(-1, D), expert_output * processed_weights.unsqueeze(1))


            current_pos += num_assigned

        # Reshape back to original shape
        return final_output.reshape(B, T, D)


# --- Advanced Reasoning Components ---

# Reasoning Tracker (Simple GRU State)
class ReasoningTracker(nn.Module):
    """Maintains a reasoning state using a GRU over token representations."""
    def __init__(self, hidden_dim: int, num_layers: int = 1, reasoning_steps: int = 1): # Steps might be implicit in application
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reasoning_steps = reasoning_steps # How many times to pass through tracker
        self.state_tracker = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # Optional projection/confidence head
        self.confidence_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor, initial_state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = hidden_states.size()

        # Initialize reasoning state if not provided
        if initial_state is None:
            h_0 = torch.zeros(self.state_tracker.num_layers, B, self.hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            h_0 = initial_state

        current_states = hidden_states
        final_reasoning_state = h_0

        # Iteratively refine representations (optional, controlled by reasoning_steps)
        for _ in range(self.reasoning_steps):
             output_states, final_reasoning_state = self.state_tracker(current_states, final_reasoning_state)
             # You could add the output back to the input for iterative refinement:
             # current_states = hidden_states + output_states # Example refinement
             current_states = output_states # Or just use the output


        # Use the final state of the *last token* as the overall reasoning confidence/summary
        final_token_state = final_reasoning_state[-1] # Get last layer's state (B, C)
        confidence = torch.sigmoid(self.confidence_predictor(final_token_state)) # (B, 1)

        # Return the final token representations and the final hidden state of the GRU
        return current_states, final_reasoning_state, confidence

    def reset_state(self):
         # GRU state is implicitly reset on each forward pass if initial_state=None
         pass


# Tree of Thought Reasoning (Conceptual Placeholder)
class TreeOfThought:
    """
    Conceptual class for Tree of Thought reasoning.
    Actual implementation requires complex integration with generation loop,
    value functions, and search strategies (e.g., Beam Search, MCTS).
    This is a simplified placeholder demonstrating the idea.
    """
    def __init__(self, model, tokenizer: PreTrainedTokenizerBase, num_branches: int = 3, max_depth: int = 3, beam_size: int = 2):
        self.model = model # The main language model
        self.tokenizer = tokenizer
        self.num_branches = num_branches
        self.max_depth = max_depth
        self.beam_size = beam_size
        # A simple value head (could be trained separately or use model logits)
        self.value_head = nn.Linear(model.config.n_embd, 1).to(model.gpt_lm_head.transformer.wte.weight.device)

    @torch.no_grad()
    def _evaluate_state(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
        """Evaluates the 'value' of a given thought state (sequence)."""
        # Use the model's hidden state before the LM head
        outputs = self.model.gpt_lm_head.transformer(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        # Use the representation of the last token
        last_token_hidden_state = outputs[:, -1, :]
        value = self.value_head(last_token_hidden_state).mean().item() # Average over batch if B > 1
        return value

    @torch.no_grad()
    def _expand_node(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, temperature: float = 0.7) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        """Generates candidate next steps (branches) from a node."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs["logits"][:, -1, :] / temperature # Get logits for the next token

        # Ensure logits are float32 for stable sampling
        probs = F.softmax(logits.float(), dim=-1)

        # Sample multiple next tokens to create branches
        # Use multinomial sampling - might need adjustments for beam search compatibility
        next_token_candidates = torch.multinomial(probs, num_samples=self.num_branches, replacement=True) # (B, num_branches)

        branches = []
        for i in range(self.num_branches):
            next_token = next_token_candidates[:, i:i+1] # (B, 1)
            branch_ids = torch.cat([input_ids, next_token], dim=1)
            branch_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

            # Evaluate the value of the new state
            value = self._evaluate_state(branch_ids, branch_mask)
            branches.append((branch_ids, branch_mask, value))

        return branches

    @torch.no_grad()
    def search(self, initial_prompt: str, generation_length: int = 50) -> str:
        """Performs ToT search to generate a response."""
        self.model.eval()
        inputs = self.tokenizer(initial_prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Beam search state: list of (ids, mask, accumulated_value) tuples
        beam = [(input_ids, attention_mask, self._evaluate_state(input_ids, attention_mask))]

        for depth in range(min(self.max_depth, generation_length)):
            all_candidates = []
            for node_ids, node_mask, node_value in beam:
                # Prevent generating beyond max length
                if node_ids.shape[1] >= self.model.config.max_seq_len or node_ids.shape[1] >= input_ids.shape[1] + generation_length:
                    all_candidates.append((node_ids, node_mask, node_value)) # Keep finished sequences
                    continue

                branches = self._expand_node(node_ids, node_mask)
                for branch_ids, branch_mask, branch_value in branches:
                     # Simple cumulative value; more sophisticated combination possible
                    all_candidates.append((branch_ids, branch_mask, node_value + branch_value))

            # Prune the beam
            # Sort candidates by value (higher is better)
            all_candidates.sort(key=lambda x: x[2], reverse=True)
            beam = all_candidates[:self.beam_size]

            # Check for stopping condition (e.g., all beams finished or generated EOS)
            if all(self.tokenizer.eos_token_id in b[0][0] for b in beam if b[0].shape[1] > input_ids.shape[1]):
                 break

        # Select the best path from the final beam
        best_ids, _, best_value = max(beam, key=lambda x: x[2])
        logger.info(f"ToT Search finished. Best path value: {best_value:.4f}")

        # Decode the result
        # Remove the initial prompt part if needed
        result_ids = best_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(result_ids, skip_special_tokens=True)


# Algorithmic Reasoner (Conceptual Neural Register Machine)
class AlgorithmicReasoner(nn.Module):
    """
    A conceptual module for performing step-by-step algorithmic reasoning
    using neural registers and learned operations.
    """
    def __init__(self, hidden_dim: int, num_registers: int = 4, max_steps: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_registers = num_registers
        self.max_steps = max_steps

        # Learnable initial state for registers
        self.register_init = nn.Parameter(torch.randn(1, num_registers, hidden_dim))

        # Control network: decides operation and registers based on input/current state
        # Input: current hidden state + flattened registers
        controller_input_dim = hidden_dim + num_registers * hidden_dim
        self.controller = nn.Linear(controller_input_dim, hidden_dim * 3 + num_registers * 2 + 1) # Ops + RegSelect + Halt

        # Simple learned operations (can be made more complex)
        self.op_transform = nn.Linear(hidden_dim, hidden_dim) # General transformation


    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Performs algorithmic steps based on the input hidden states.

        Args:
            hidden_states: Input tensor (B, T, C). Typically uses the final token's state.

        Returns:
            Tuple of (final_registers_state, list_of_intermediate_register_states)
        """
        B, T, C = hidden_states.size()
        # Use the last token's representation as the 'program' or initial instruction
        program_input = hidden_states[:, -1, :] # (B, C)

        # Initialize registers
        registers = self.register_init.expand(B, -1, -1).clone() # (B, num_reg, C)
        all_register_states = [registers.clone()]
        halted = torch.zeros(B, 1, device=hidden_states.device, dtype=torch.bool)

        for step in range(self.max_steps):
            if halted.all():
                break

            # Prepare controller input
            controller_input = torch.cat([program_input, registers.view(B, -1)], dim=1) # (B, C + num_reg*C)
            control_signals = self.controller(controller_input) # (B, ops+regs+halt)

            # Parse control signals
            op_signal_dim = self.hidden_dim * 3 # Example: 3 operations
            reg_signal_dim = self.num_registers * 2
            op_signals, reg_signals, halt_signal = control_signals.split(
                [op_signal_dim, reg_signal_dim, 1], dim=1
            )

            # --- Register Selection ---
            reg_a_scores, reg_b_scores = reg_signals.view(B, 2, self.num_registers).unbind(1) # (B, num_reg) each
            reg_a_idx = torch.argmax(reg_a_scores, dim=1) # (B,)
            reg_b_idx = torch.argmax(reg_b_scores, dim=1) # (B,)

            # Get register values (gather based on indices)
            # Need batch indexing: registers[batch_idx, register_idx]
            batch_indices = torch.arange(B, device=registers.device)
            reg_a = registers[batch_indices, reg_a_idx] # (B, C)
            reg_b = registers[batch_indices, reg_b_idx] # (B, C)

            # --- Operation Execution (Example: simple gated update) ---
            # Signals could control mixing coefficients or select specific ops
            gate1, gate2, gate3 = op_signals.chunk(3, dim=1) # (B, C) each
            gate1, gate2, gate3 = torch.sigmoid(gate1), torch.sigmoid(gate2), torch.tanh(gate3) # Activations

            # Example operation: result = g1*reg_a + g2*reg_b + g3*transform(reg_a)
            result = gate1 * reg_a + gate2 * reg_b + gate3 * self.op_transform(reg_a) # (B, C)

            # --- Update Register ---
            # Update register 'a' with the result (can be made more flexible)
            # Use scatter_ for in-place update is tricky with autograd, clone and index assignment is safer
            new_registers = registers.clone()
            new_registers[batch_indices, reg_a_idx] = result
            registers = new_registers

            all_register_states.append(registers.clone())

            # --- Halt Condition ---
            halt_prob = torch.sigmoid(halt_signal) # (B, 1)
            halt_decision = (halt_prob > 0.5)
            halted = halted | halt_decision # Update halted status

        # Return the final state of the registers
        return registers, all_register_states


# Calculator Tool (using SymPy)
class CalculatorTool:
    """Provides symbolic math evaluation using SymPy."""
    def __init__(self):
        pass # No initialization needed

    def calculate(self, expression: str) -> dict[str, str | None]:
        """
        Evaluates a mathematical expression using SymPy.

        Args:
            expression: A string containing the mathematical expression.

        Returns:
            A dict with 'result' (string representation) or 'error'.
        """
        try:
            # Sanitize input slightly (basic checks)
            if not isinstance(expression, str) or not expression.strip():
                 return {"error": "Invalid input: Expression must be a non-empty string."}
            # Avoid overly complex or potentially harmful expressions (simple length check)
            if len(expression) > 200:
                 return {"error": "Expression too long."}

            # Attempt to parse and evaluate
            # Use sympify with locals to restrict available functions (optional security)
            # local_dict = {"sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp, "sin": sp.sin, ...}
            # expr = sp.sympify(expression, locals=local_dict)
            expr = sp.sympify(expression)

            # Evaluate numerically if possible, otherwise simplify
            try:
                # Attempt numerical evaluation first
                evaluated_result = expr.evalf()
                # Check if the result is a number
                if isinstance(evaluated_result, sp.Number):
                    result = str(evaluated_result)
                else:
                    # If not purely numeric, simplify the symbolic expression
                    simplified_result = sp.simplify(expr)
                    result = str(simplified_result)
            except (TypeError, ValueError):
                # Fallback to symbolic simplification if evalf fails
                simplified_result = sp.simplify(expr)
                result = str(simplified_result)

            # Limit output length
            if len(result) > 500:
                 return {"error": "Result too long."}

            return {"result": result}

        except (sp.SympifyError, TypeError, SyntaxError, ValueError) as e:
            return {"error": f"Calculation error: {type(e).__name__}"}
        except Exception as e:
            logger.error(f"Unexpected calculator error: {e} for expression '{expression}'")
            return {"error": "An unexpected error occurred during calculation."}


# --- Transformer Block and Model Architecture ---

# Transformer Block
class TransformerBlock(nn.Module):
    """A single block of the Transformer model."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.attn_norm = RMSNorm(config.n_embd)

        # Choose attention mechanism
        if config.use_gqa:
            self.attn = GroupedQueryAttention(config)
        else:
            self.attn = MultiHeadSelfAttention(config) # Default MHA

        # Optional RWKV block
        self.use_rwkv = config.use_rwkv
        if self.use_rwkv:
             self.rwkv_norm = RMSNorm(config.n_embd) # Normalize input to RWKV
             self.rwkv_attn = RWKVAttention(config)

        # Optional SSM block
        self.use_ssm = config.use_ssm
        if self.use_ssm:
             self.ssm_norm = RMSNorm(config.n_embd) # Normalize input to SSM
             self.ssm = SelectiveSSM(config.n_embd, ssm_state_dim=config.n_embd//8) # Example state dim

        # FFN block (choose between standard SwiGLU and MoE)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.use_moe = config.use_moe
        if self.use_moe:
            self.mlp = SparseMoE(config.n_embd, num_experts=config.num_experts, top_k=config.top_k_experts)
        else:
            self.mlp = SwiGLU(config.n_embd, dropout_prob=config.dropout_prob) # Pass dropout here if desired

        self.dropout = nn.Dropout(config.dropout_prob) # Applied after residual connections

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
                alibi_bias: torch.Tensor | None = None, use_cache: bool = False,
                position_ids: torch.LongTensor | None = None) -> torch.Tensor:

        # --- Attention Path ---
        residual = x
        h = self.attn_norm(x)
        attn_output = self.attn(h, attention_mask=attention_mask, alibi_bias=alibi_bias,
                                use_cache=use_cache, position_ids=position_ids)
        # Apply dropout after attention projection (inside attn module)
        x = residual + attn_output # First residual connection

        # --- Optional RWKV Path ---
        if self.use_rwkv:
            residual = x
            h = self.rwkv_norm(x)
            rwkv_output = self.rwkv_attn(h)
            x = residual + self.dropout(rwkv_output) # Add dropout after RWKV output

        # --- Optional SSM Path ---
        if self.use_ssm:
             residual = x
             h = self.ssm_norm(x)
             ssm_output = self.ssm(h)
             x = residual + self.dropout(ssm_output) # Add dropout after SSM output


        # --- FFN Path ---
        residual = x
        h = self.ffn_norm(x)
        ffn_output = self.mlp(h)
        x = residual + self.dropout(ffn_output) # Second residual connection + dropout

        return x

    def _forward_with_checkpointing(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                                    alibi_bias: torch.Tensor | None, use_cache: bool,
                                    position_ids: torch.LongTensor | None) -> torch.Tensor:
        # Define functions for checkpointing each major computation step
        def run_attn(current_x, norm_layer, attn_layer, mask, bias, cache, pos_ids):
            normed = norm_layer(current_x)
            return attn_layer(normed, attention_mask=mask, alibi_bias=bias, use_cache=cache, position_ids=pos_ids)

        def run_rwkv(current_x, norm_layer, rwkv_layer):
             normed = norm_layer(current_x)
             return rwkv_layer(normed)

        def run_ssm(current_x, norm_layer, ssm_layer):
             normed = norm_layer(current_x)
             return ssm_layer(normed)

        def run_ffn(current_x, norm_layer, ffn_layer, dropout_layer):
            normed = norm_layer(current_x)
            output = ffn_layer(normed)
            return dropout_layer(output) # Apply dropout within checkpoint if needed

        # Attention path
        residual = x
        # Checkpointing requires all inputs to require grad or be explicitly marked non-requiring
        attn_output = checkpoint(run_attn, x, self.attn_norm, self.attn, attention_mask, alibi_bias, use_cache, position_ids, use_reentrant=False)
        x = residual + attn_output

        # RWKV path
        if self.use_rwkv:
             residual = x
             rwkv_output = checkpoint(run_rwkv, x, self.rwkv_norm, self.rwkv_attn, use_reentrant=False)
             x = residual + self.dropout(rwkv_output) # Dropout outside checkpoint

        # SSM path
        if self.use_ssm:
             residual = x
             ssm_output = checkpoint(run_ssm, x, self.ssm_norm, self.ssm, use_reentrant=False)
             x = residual + self.dropout(ssm_output) # Dropout outside checkpoint

        # FFN path
        residual = x
        ffn_output = checkpoint(run_ffn, x, self.ffn_norm, self.mlp, self.dropout, use_reentrant=False)
        x = residual + ffn_output # Dropout applied inside run_ffn for checkpointing

        return x

# Core GPT Model (Stack of Transformer Blocks)
class GPTModel(nn.Module):
    """The core GPT model consisting of embedding, transformer blocks, and final normalization."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Input Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Positional embeddings are handled by RoPE or ALiBi inside attention
        self.drop = nn.Dropout(config.dropout_prob)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
             TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final Normalization
        self.norm_f = RMSNorm(config.n_embd)

        # Initialize weights
        self.apply(self._init_weights)
        # Log parameter count
        logger.info(f"Model parameter count: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            # Careful initialization with lower standard deviation
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Smaller initialization scale for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
            # Initialize normalization layers with ones
            if hasattr(module, 'weight'):
                torch.nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        # Special handling for attention projections (Q, K, V) to ensure stability
        if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
            with torch.no_grad():
                # Smaller initialization for attention projections
                for proj in [module.q_proj, module.k_proj, module.v_proj]:
                    torch.nn.init.normal_(proj.weight, mean=0.0, std=0.01)

    def detect_anomaly(name, tensor):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"NaN or Inf detected in {name}: min={tensor.min().item()}, max={tensor.max().item()}")
            return True
        return False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                use_cache: bool = False, position_ids: torch.LongTensor | None = None) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.config.max_seq_len:
             logger.warning(f"Input sequence length ({T}) exceeds model max ({self.config.max_seq_len}). Truncating.")
             input_ids = input_ids[:, :self.config.max_seq_len]
             if position_ids is not None:
                  position_ids = position_ids[:, :self.config.max_seq_len]
             T = self.config.max_seq_len

        # 1. Token Embeddings
        token_embeddings = self.wte(input_ids) # (B, T, C)
        x = self.drop(token_embeddings)

        # 2. Build ALiBi bias if needed (once per forward pass)
        alibi_bias = None
        if self.config.alibi:
            # Build bias appropriate for the current sequence length T
            alibi_bias = build_alibi_tensor(self.config.n_head, T, device=x.device, dtype=x.dtype)

        # 3. Process through Transformer Blocks
        for i, block in enumerate(self.blocks):
             # Apply gradient checkpointing if enabled and training
             if self.config.gradient_checkpointing and self.training:
                 # Checkpointing needs function closure
                 def create_custom_forward(blk):
                     def custom_forward(_x, _mask, _bias, _cache, _pos_ids):
                         return blk(_x, attention_mask=_mask, alibi_bias=_bias, use_cache=_cache, position_ids=_pos_ids)
                     return custom_forward

                 x = checkpoint(create_custom_forward(block), x, attention_mask, alibi_bias, use_cache, position_ids, use_reentrant=False)
             else:
                 # Normal forward pass
                 x = block(x, attention_mask=attention_mask, alibi_bias=alibi_bias,
                           use_cache=use_cache, position_ids=position_ids)

        # 4. Final Normalization
        x = self.norm_f(x)
        return x # Return hidden states (B, T, C)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
             max_new_tokens: int = 50, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
             eos_token_id: int | None = None, do_sample: bool = True) -> torch.Tensor:
        """Generates sequences autoregressively using KV caching."""
        self.eval() # Set model to evaluation mode
        B, T_prompt = input_ids.shape

        # Reset KV caches in all attention blocks
        for block in self.blocks:
            # GQA/MHA share the same base class structure
            if hasattr(block.attn, 'kv_cache') and block.attn.kv_cache is not None:
                block.attn.kv_cache.reset(batch_size=B)
            # Reset RWKV state if present
            if hasattr(block, 'rwkv_attn') and hasattr(block.rwkv_attn, 'reset_state'):
                block.rwkv_attn.reset_state()

        generated_ids = input_ids.clone()
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.clone()

        with torch.no_grad():
            # --- Process Prompt Phase ---
            # Calculate position_ids for the prompt
            prompt_position_ids = torch.arange(0, T_prompt, device=input_ids.device).unsqueeze(0).expand(B, -1)
            
            # Run prompt through the model to fill the KV cache
            _ = self(input_ids, attention_mask=attention_mask, use_cache=True, position_ids=prompt_position_ids)
            # KV cache now holds the state for the prompt

            # --- Generation Phase ---
            for step in range(max_new_tokens):
                # Get the last token ID to predict the next one
                next_token_input_ids = generated_ids[:, -1:] # (B, 1)
                
                # Calculate position_id for the *current* token
                current_position = generated_ids.size(1) - 1
                next_position_ids = torch.tensor([[current_position]], device=input_ids.device, dtype=torch.long)

                try:
                    # Forward pass for the single next token, using KV cache
                    hidden_states = self(next_token_input_ids, 
                                    attention_mask=None,  # Masking handled by cache
                                    use_cache=True, 
                                    position_ids=next_position_ids) # Pass correct position

                    # Get logits for the last generated token
                    # Project hidden state to vocabulary using the embedding matrix's transpose (tied weights)
                    logits = F.linear(hidden_states[:, -1, :], self.wte.weight) # (B, vocab_size)

                    # --- Apply Sampling Strategies ---
                    if do_sample:
                        # Temperature scaling
                        logits = logits / temperature

                        # Top-K filtering
                        if top_k > 0:
                            v, _ = torch.topk(logits, top_k)
                            logits[logits < v[:, [-1]]] = -float('Inf')

                        # Top-P (nucleus) filtering
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            # Remove tokens with cumulative probability above the threshold
                            sorted_indices_to_remove = cumulative_probs > top_p
                            # Shift right to keep the first token above the threshold
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            # Scatter back to original order
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            logits[indices_to_remove] = -float('Inf')

                        # Sample from the filtered distribution
                        probs = F.softmax(logits.float(), dim=-1) # Use float32 for stability
                        next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
                    else:
                        # Greedy decoding
                        next_token = torch.argmax(logits, dim=-1, keepdim=True) # (B, 1)

                    # Append the generated token
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                    # Update attention mask for the new token
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

                    # Check for EOS token
                    if eos_token_id is not None and (next_token == eos_token_id).all():
                        break
                except RuntimeError as e:
                    logger.error(f"Error during generation step {step}: {e}")
                    # If we've generated at least some tokens, return what we have
                    if step > 0:
                        logger.warning(f"Generation stopped early at step {step}. Returning partial result.")
                        break
                    else:
                        # If we couldn't generate anything, re-raise the exception
                        raise

            return generated_ids


# GPT LM Head Model (Adds Language Modeling Head)
class GPTLMHeadModel(nn.Module):
    """GPT Model with a language modeling head on top."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = GPTModel(config)
        # LM head weight is tied to token embedding matrix (transformer.wte)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None, use_cache: bool = False,
                position_ids: torch.LongTensor | None = None) -> dict[str, torch.Tensor | None]:
        """
        Forward pass for language modeling.

        Args:
            input_ids (Tensor): Input token IDs (B, T).
            attention_mask (Tensor, optional): Mask for padding tokens (B, T).
            labels (Tensor, optional): Target token IDs, shifted right (B, T).
            use_cache (bool): Whether to use KV caching (for inference).
            position_ids (LongTensor, optional): Explicit position IDs.

        Returns:
            dict: Contains 'loss' (if labels provided) and 'logits'.
        """
        # 1. Get hidden states from the base transformer model
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask,
                                         use_cache=use_cache, position_ids=position_ids) # (B, T, C)

        # 2. Project hidden states to vocabulary logits
        # Uses the same weight matrix as the token embeddings (weight tying)
        logits = F.linear(hidden_states, self.transformer.wte.weight) # (B, T, vocab_size)

        # 3. Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            # Logits: Ignore last token's prediction (no label) -> (B, T-1, V)
            shift_logits = logits[:, :-1, :].contiguous()
            # Labels: Ignore first token (no prediction for it) -> (B, T-1)
            shift_labels = labels[:, 1:].contiguous()

            # Flatten and compute cross-entropy loss
            # Ignore index -100 (common practice for padding in labels)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss, "logits": logits, "hidden_states": hidden_states} # Return hidden states too


# --- Multihead Latent Attention (MLA) ---

class MLA(nn.Module):
    """
    Multihead Latent Attention (MLA) uses learnable latent tokens to process
    sequence information, potentially enhancing reasoning or summarization.
    Inspired by Perceiver IO / Set Transformer concepts.
    """
    def __init__(self, n_embd: int, n_latent: int, n_head: int, dropout_prob: float = 0.1, thinking_steps: int = 1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_latent = n_latent
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.thinking_steps = thinking_steps # Iterative refinement steps

        # Learnable latent query tokens
        self.latent_queries = nn.Parameter(torch.randn(1, n_latent, n_embd)) # Add batch dim placeholder

        # Projections for cross-attention (Latents attend to Input)
        self.q_proj_latent = nn.Linear(n_embd, n_embd, bias=False) # Query from latents
        self.k_proj_input = nn.Linear(n_embd, n_embd, bias=False) # Key from input sequence
        self.v_proj_input = nn.Linear(n_embd, n_embd, bias=False) # Value from input sequence

        # Projections for self-attention (Latents attend to Latents - optional, for refinement)
        # self.q_proj_self = nn.Linear(n_embd, n_embd, bias=False)
        # self.k_proj_self = nn.Linear(n_embd, n_embd, bias=False)
        # self.v_proj_self = nn.Linear(n_embd, n_embd, bias=False)

        # Output projection and normalization/dropout
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False) # Project aggregated latent back
        self.norm_latent = RMSNorm(n_embd)
        self.norm_input = RMSNorm(n_embd) # Normalize input K/V
        self.dropout = nn.Dropout(dropout_prob)

    def _cross_attention(self, latents: torch.Tensor, input_x: torch.Tensor) -> torch.Tensor:
        """Latent tokens attend to the input sequence."""
        B, T_latent, C = latents.size()
        B, T_input, C_in = input_x.size()
        assert C == C_in

        # Project Q (from latents), K/V (from input_x)
        q = self.q_proj_latent(latents)
        # Normalize input K/V for stability
        input_normed = self.norm_input(input_x)
        k = self.k_proj_input(input_normed)
        v = self.v_proj_input(input_normed)

        # Reshape for multi-head
        q = q.view(B, T_latent, self.n_head, self.head_dim).transpose(1, 2) # (B, H, T_latent, D_h)
        k = k.view(B, T_input, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T_input, D_h)
        v = v.view(B, T_input, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T_input, D_h)

        # Compute attention scores (Latent Q x Input K)
        # Use float32 for stability
        attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, T_latent, T_input)
        attn_weights = F.softmax(attn_scores, dim=-1).type_as(latents)
        attn_weights = self.dropout(attn_weights)

        # Compute output (Attn Weights x Input V)
        attn_output = torch.matmul(attn_weights, v) # (B, H, T_latent, D_h)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_latent, C)
        return attn_output # Updated latent representation

    # Optional: Self-attention among latent tokens
    # def _self_attention(self, latents: torch.Tensor) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input sequence using latent attention.

        Args:
            x: Input tensor (B, T, C)

        Returns:
            Tensor: Enhanced input tensor with aggregated latent info (B, T, C)
        """
        B, T, C = x.size()

        # Expand latent queries for the batch
        current_latents = self.latent_queries.expand(B, -1, -1) # (B, n_latent, C)

        # Iterative refinement loop
        for _ in range(self.thinking_steps):
            # 1. Latents attend to input sequence (Cross-Attention)
            latent_update = self._cross_attention(current_latents, x)
            # Add & Norm (residual connection for latents)
            current_latents = self.norm_latent(current_latents + self.dropout(latent_update))

            # 2. Optional: Latents attend to themselves (Self-Attention)
            # latent_self_update = self._self_attention(current_latents)
            # current_latents = self.norm_latent(current_latents + self.dropout(latent_self_update))

        # Aggregate the final latent representations (e.g., mean pooling)
        aggregated_latent = current_latents.mean(dim=1) # (B, C)

        # Project the aggregated info and add it back to the original input sequence
        # This injects the "summary" or "reasoning result" back into each token position
        enhancement = self.out_proj(aggregated_latent) # (B, C)
        enhanced_x = x + enhancement.unsqueeze(1) # Add to each token (broadcast over T)

        return enhanced_x


# --- Enhanced Model Wrapper ---

class EnhancedGPTLMHeadModel(nn.Module):
    """
    Wrapper model integrating the base GPT-LM with advanced components like
    MLA, Reasoning Tracker, Algorithmic Reasoner, and Calculator Tool.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # --- Core Language Model ---
        self.gpt_lm_head = GPTLMHeadModel(config)

        # --- Optional Enhancement Modules ---
        self.use_mla = config.use_mla
        if self.use_mla:
            self.mla = MLA(config.n_embd, config.mla_n_latent, config.n_head,
                           dropout_prob=config.dropout_prob, thinking_steps=1) # 1 step default

        self.use_reasoning_tracker = config.use_reasoning_tracker
        if self.use_reasoning_tracker:
            self.reasoning_tracker = ReasoningTracker(config.n_embd, reasoning_steps=config.reasoning_steps)

        self.use_algorithmic_reasoner = config.use_algorithmic_reasoner
        if self.use_algorithmic_reasoner:
            self.algorithmic_reasoner = AlgorithmicReasoner(config.n_embd,
                                                           num_registers=config.algorithmic_reasoner_registers,
                                                           max_steps=10) # Example max steps

        # --- Tools ---
        self.use_calculator = config.use_calculator
        if self.use_calculator:
            self.calculator = CalculatorTool()


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None, use_cache: bool = False, position_ids: torch.LongTensor | None = None) -> dict[str, torch.Tensor | None]:
        # MPS-optimized forward pass with explicit device handling
        
        # Memory check and cleanup
        if torch.backends.mps.is_available() and hasattr(torch.mps, 'empty_cache') and not use_cache:
            # Only clear cache if we're not using KV caching (which would be cleared)
            torch.mps.empty_cache()
        
        # Get base transformer hidden states
        hidden_states = self.gpt_lm_head.transformer(
            input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            position_ids=position_ids
        )

        # Apply MLA enhancement (optional)
        if self.use_mla and not use_cache:
            hidden_states = self.mla(hidden_states)

        # Apply Reasoning Tracker (optional)
        reasoning_output = None
        if self.use_reasoning_tracker and not use_cache:
            refined_states, _, confidence = self.reasoning_tracker(hidden_states)
            hidden_states = refined_states
            reasoning_output = {"reasoning_confidence": confidence}

        # Apply Algorithmic Reasoner (optional)
        algo_output = None
        if self.use_algorithmic_reasoner and not use_cache:
            final_registers, _ = self.algorithmic_reasoner(hidden_states)
            algo_output = {"algorithmic_registers": final_registers}

        # Final LM Head Projection
        logits = F.linear(hidden_states, self.gpt_lm_head.transformer.wte.weight)

        # Calculate Loss
        loss = None
        if labels is not None:
            # Make sure batch sizes match
            if logits.size(0) != labels.size(0):
                print(f"Warning: Batch size mismatch - logits: {logits.size(0)}, labels: {labels.size(0)}")
                min_batch = min(logits.size(0), labels.size(0))
                logits = logits[:min_batch]
                labels = labels[:min_batch]
            
            # Ensure sequence lengths match before shifting
            min_seq_len = min(logits.size(1), labels.size(1))
            logits = logits[:, :min_seq_len, :]
            labels = labels[:, :min_seq_len]
            
            # Now perform the shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute loss with adjusted tensors
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            try:
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            except ValueError as e:
                print(f"Error in loss calculation: {e}")
                # Provide a fallback loss to prevent training crash
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Combine outputs
        output_dict = {"loss": loss, "logits": logits, "hidden_states": hidden_states}
        if reasoning_output:
            output_dict.update(reasoning_output)
        if algo_output:
            output_dict.update(algo_output)

        return output_dict

    # Delegate generate call to the underlying GPTModel's generate method
    # which handles KV caching correctly.
    def generate(self, *args, **kwargs):
        # Ensure generation uses the base model's efficient path
        return self.gpt_lm_head.transformer.generate(*args, **kwargs)

    # --- Methods for Tool Use and Advanced Reasoning ---

    def run_calculator(self, expression: str) -> str:
        """Interface to the calculator tool."""
        if not self.use_calculator:
            return "Calculator tool is disabled."
        result_dict = self.calculator.calculate(expression)
        return result_dict.get("result", f"Error: {result_dict.get('error', 'Unknown calculator error')}")

    def run_tot_search(self, prompt: str, tokenizer: PreTrainedTokenizerBase, generation_length: int = 50) -> str:
        """Interface to run Tree of Thought search (if enabled and implemented)."""
        if not hasattr(self, 'tree_of_thought'):
             # Initialize ToT here if needed, requires tokenizer
             # Check if ToT is conceptually enabled in config, even if not used in fwd
             if self.config.use_tree_of_thought: # Placeholder config flag needed
                  self.tree_of_thought = TreeOfThought(self, tokenizer) # Pass self (model) and tokenizer
             else:
                  return "Tree of Thought is not configured for this model."

        if not isinstance(self.tree_of_thought, TreeOfThought):
             return "Tree of Thought component not available or not initialized correctly."

        logger.info(f"Starting Tree of Thought search for prompt: '{prompt[:50]}...'")
        return self.tree_of_thought.search(prompt, generation_length=generation_length)

    def reset_reasoning_states(self):
        """Resets states of internal reasoning modules if they have state."""
        if self.use_reasoning_tracker and hasattr(self.reasoning_tracker, 'reset_state'):
            self.reasoning_tracker.reset_state()
        # Add resets for other stateful reasoning modules if necessary
        logger.info("Reasoning module states reset.")


# --- Training Loop & Utilities ---

# Simple Dataset for demonstration
class SimpleDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Simple tokenization: encode all texts
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        # Create labels by shifting input_ids
        self.labels = self.encodings["input_ids"].clone()
        # Set padding tokens in labels to ignore_index for CrossEntropyLoss
        self.labels[self.labels == tokenizer.pad_token_id] = -100

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def train_model(model: EnhancedGPTLMHeadModel, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                epochs: int, device: torch.device, accumulation_steps: int = 1):
    """Basic training loop with progress bar and logging."""
    if torch.backends.mps.is_available():
        # Check available memory
        print(f"Memory before batch: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
    model.train()
    model.to(device)
    total_steps = len(train_loader) * epochs // accumulation_steps
    progress_bar = tqdm(range(total_steps), desc="Training")
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train() # Ensure model is in training mode each epoch
        optimizer.zero_grad() # Reset gradients at the start of epoch

        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
            loss = outputs['loss']

            # Scale loss for gradient accumulation
            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            # Backward pass
            loss.backward()

            # Optimizer step (every accumulation_steps)
            if (i + 1) % accumulation_steps == 0:
                 # Gradient clipping (optional but recommended)
                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                 optimizer.step()
                 optimizer.zero_grad() # Clear gradients after step
                 progress_bar.update(1)
                 global_step += 1
                 progress_bar.set_postfix({"Epoch": epoch + 1, "Loss": loss.item() * accumulation_steps}) # Log unscaled loss

            epoch_loss += loss.item() * accumulation_steps # Accumulate unscaled loss

            # Log step loss periodically
            if global_step % 100 == 0 and global_step > 0: # Log every 100 steps
                 logger.info(f"Epoch: {epoch+1}, Step: {global_step}/{total_steps}, Loss: {loss.item() * accumulation_steps:.4f}")


        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")

    progress_bar.close()
    print('Training complete!')


# --- Prompting Function ---

def prompt_model(
    model: EnhancedGPTLMHeadModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    do_sample: bool = True,
    use_calculator_tool: bool = True,
    calculator_trigger: str = "[CALC:",
    calculator_end: str = "]"
) -> str:
    """
    Handles prompting, generation, and optional tool use (Calculator).

    Args:
        model: The EnhancedGPTLMHeadModel instance.
        tokenizer: The tokenizer.
        prompt_text: The input prompt string.
        max_new_tokens: Max tokens to generate.
        temperature, top_k, top_p, do_sample: Generation parameters.
        use_calculator_tool: Whether to enable calculator calls.
        calculator_trigger: String that triggers the calculator.
        calculator_end: String that ends the calculator expression.

    Returns:
        The generated response string.
    """
    model.eval()
    model.to(device)

    current_text = prompt_text
    generated_sequence = ""
    all_generated_ids = tokenizer.encode(current_text, return_tensors="pt").to(device)

    print(f"\n--- Prompting ---")
    print(f"Initial Prompt: {current_text}")

    active_generation_length = 0
    while active_generation_length < max_new_tokens:

        # Prepare inputs for the core generate function
        input_ids = all_generated_ids
        attention_mask = torch.ones_like(input_ids)

        # Use model's internal generate function
        # Only generate *one* token at a time if we need to check for tools
        generate_step_length = 1 if use_calculator_tool else max_new_tokens - active_generation_length

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=generate_step_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
        )

        # Extract *only* the newly generated token IDs
        new_token_ids = output_ids[0, input_ids.shape[1]:]
        if len(new_token_ids) == 0:
            break # Generation stopped (e.g., EOS)

        # Decode the new token(s)
        new_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        generated_sequence += new_text
        current_text += new_text
        all_generated_ids = output_ids # Update the full sequence for the next step

        active_generation_length += len(new_token_ids)

        # --- Tool Check: Calculator ---
        if use_calculator_tool and model.use_calculator and calculator_trigger in generated_sequence:
            start_idx = generated_sequence.rfind(calculator_trigger)
            if start_idx != -1:
                end_idx = generated_sequence.find(calculator_end, start_idx)
                if end_idx != -1:
                    expression = generated_sequence[start_idx + len(calculator_trigger):end_idx].strip()
                    print(f"[Tool Call Detected] Expression: {expression}")

                    # Call calculator tool
                    calc_result = model.run_calculator(expression)
                    print(f"[Tool Result] Output: {calc_result}")

                    # Inject result back into the context
                    # Replace the trigger and expression with the result
                    generated_sequence = generated_sequence[:start_idx] + f" {calc_result} " + generated_sequence[end_idx + len(calculator_end):]
                    current_text = prompt_text + generated_sequence # Rebuild current_text

                    # Re-tokenize the updated text for the next generation step
                    all_generated_ids = tokenizer.encode(current_text, return_tensors="pt").to(device)
                    print(f"[Context Updated] New context tail: ...{current_text[-100:]}")


        # Check for EOS in the generated part
        if tokenizer.eos_token_id in new_token_ids:
            print("[EOS Detected]")
            break

    print(f"\n--- Generation Complete ---")
    # Return only the generated part, excluding the initial prompt
    final_response = tokenizer.decode(all_generated_ids[0, len(tokenizer.encode(prompt_text)):], skip_special_tokens=True)
    print(f"Final Response: {final_response}")
    return final_response



# # --- Main Execution Block ---
# if __name__ == '__main__':
#     # --- Configuration ---
#     VOCAB_SIZE = 10000
#     MAX_SEQ_LEN = 512  # Increase context window 
#     N_EMBD = 256       # Increase embedding dimension
#     N_LAYER = 6        # Add more layers for depth
#     N_HEAD = 8 
#     N_KV_HEADS = 2
#     USE_GQA = True
#     USE_ROPE = True
#     USE_FLASH = True
#     USE_MOE = False
#     USE_MLA = True
#     USE_REASONING = False
#     USE_CALCULATOR = True
#     GRADIENT_CHECKPOINTING = True
    

#     # --- Tokenizer Loading ---
#     tokenizer_name = "gpt2"
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#             print(f"Set PAD token to EOS token: {tokenizer.pad_token}")
#         VOCAB_SIZE = tokenizer.vocab_size
#     except Exception as e:
#         print(f"Error loading tokenizer '{tokenizer_name}': {e}")
#         exit()

#     # --- Model Configuration ---
#     config = GPTConfig(
#         vocab_size=VOCAB_SIZE,
#         max_seq_len=MAX_SEQ_LEN,
#         n_embd=N_EMBD,
#         n_layer=N_LAYER,
#         n_head=N_HEAD,
#         dropout_prob=0.1,
#         alibi=False, use_rope=USE_ROPE,
#         flash_attention=USE_FLASH,
#         n_kv_heads=N_KV_HEADS if USE_GQA else None,
#         use_gqa=USE_GQA,
#         use_rwkv=False, use_ssm=False,
#         use_moe=USE_MOE,
#         num_experts=4, top_k_experts=2,
#         gradient_checkpointing=GRADIENT_CHECKPOINTING,
#         mla_n_latent=8, use_mla=USE_MLA,
#         reasoning_steps=1, use_reasoning_tracker=USE_REASONING,
#         use_algorithmic_reasoner=False,
#         use_calculator=USE_CALCULATOR,
#     )
#     logger.info(f"Model Config: {config.__dict__}")

#     # --- Model Initialization ---
#     model = EnhancedGPTLMHeadModel(config)
#     model.to(device)

#     # --- Load Checkpoint if exists ---
#     checkpoint_path = "checkpoints/latest_checkpoint.pth"
#     if os.path.exists(checkpoint_path):
#         print(f"Loading checkpoint from {checkpoint_path}")
#         model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#     else:
#         print("No checkpoint found, training from scratch.")

#     # --- Start saving checkpoints every 120 seconds ---
#     save_checkpoint_periodically(model, checkpoint_path, interval_sec=120)

#     # --- Load Multiple Real Datasets ---
#     from datasets import load_dataset, concatenate_datasets

#     datasets_to_load = [
#         ("tiny_stories",     "train"),                 # correct name on HF Hub
#         ("wikitext",         "wikitext-103-raw-v1"),   # specify config
#         ("bookcorpus",       "train")
#     ]

#     loaded_datasets = []
#     for name, config_or_split in datasets_to_load:
#         try:
#             # if it’s wikitext, pass the config; otherwise HuggingFace treats
#             # the second arg as split
#             if name == "wikitext":
#                 ds = load_dataset(name, config_or_split, split="train")
#             else:
#                 ds = load_dataset(name, split=config_or_split)
#             loaded_datasets.append(ds)
#             print(f"Loaded {name} ({config_or_split}) → {len(ds)} samples")
#         except Exception as e:
#             print(f"Error loading {name}: {e}")
        
#     if not loaded_datasets:
#         raise RuntimeError("Failed to load any datasets!")

#     # --- Merge and Shuffle All Datasets ---
#     combined_dataset = concatenate_datasets(loaded_datasets)
#     combined_dataset = combined_dataset.shuffle(seed=42)  # Shuffle once

#     print(f"Combined dataset has {len(combined_dataset)} examples.")

#     # --- Prepare Dataset for Next Token Prediction ---
#     class RealDataset(Dataset):
#         def __init__(self, dataset, tokenizer, max_length):
#             self.dataset = dataset
#             self.tokenizer = tokenizer
#             self.max_length = max_length

#         def __len__(self):
#             return len(self.dataset)

#         def __getitem__(self, idx):
#             text = self.dataset[idx]['text'] if 'text' in self.dataset[idx] else self.dataset[idx]['content']
#             encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
#             input_ids = encodings['input_ids'].squeeze(0)
#             attention_mask = encodings['attention_mask'].squeeze(0)
#             labels = input_ids.clone()
#             labels[input_ids == tokenizer.pad_token_id] = -100
#             return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

#     dataset_cache_path = "tokenized_dataset.pt"

#     if os.path.exists(dataset_cache_path):
#         print(f"Loading tokenized dataset from {dataset_cache_path}")
#         train_dataset = torch.load(dataset_cache_path, weights_only = False)
#     else:
#         print("Tokenizing dataset...")
#         train_dataset = RealDataset(combined_dataset, tokenizer, MAX_SEQ_LEN)
#         torch.save(train_dataset, dataset_cache_path)
#         print(f"Saved tokenized dataset to {dataset_cache_path}")

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=2,
#         shuffle=True,
#         num_workers=2,
#         pin_memory=True,
#         prefetch_factor=2  # Prefetch 2 batches per worker
#     )


#     # --- Training Setup ---
#     # Use better learning rate and training settings
#     epochs = 5
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=5e-5,                # Lower learning rate for stability
#         weight_decay=0.01,
#         betas=(0.9, 0.95)       # Better momentum parameters
#     )

#     # Add learning rate scheduler
#     from torch.optim.lr_scheduler import CosineAnnealingLR
#     scheduler = CosineAnnealingLR(
#         optimizer,
#         T_max=len(train_loader) * epochs,
#         eta_min=1e-6
#     )
    

#     print("\n--- Starting Training (Real Mixed Data) ---")
#     train_model(model, train_loader, optimizer, epochs=epochs, device=device, accumulation_steps=1)

#     # --- Save Model ---
#     model_save_path = 'enhanced_gpt_model_real_mixed.pth'
#     torch.save(model.state_dict(), model_save_path)
#     logger.info(f"Model state dict saved to {model_save_path}")
#     print(f"Model saved to {model_save_path}")

#     # --- Prompting Example ---
#     print("\n--- Starting Prompting Demo ---")
#     prompt = "Once upon a time, in a world filled with machines,"
#     generated_text = prompt_model(
#         model,
#         tokenizer,
#         prompt,
#         max_new_tokens=60,
#         temperature=0.7,
#         top_k=50,
#         top_p=0.95,
#         do_sample=True,
#         use_calculator_tool=config.use_calculator
#     )

#     print("\n--- Demo Finished ---")



# --- Main Execution Block with MPS Optimization ---
import argparse
import os
import torch
from transformers import AutoTokenizer

# --- Main Execution Block with MPS Optimization and Command Line Arguments ---
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or run a GPT-style model with MPS optimizations')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'prompt', 'both'],
                        help='Operation mode: train, prompt, or both')
    parser.add_argument('--model_path', type=str, default='checkpoints/latest_checkpoint.pth',
                        help='Path to load/save model checkpoint')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--seq_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--embd_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_kv_head', type=int, default=2, help='Number of KV heads for GQA')
    parser.add_argument('--prompt', type=str, default="Once upon a time, in a world filled with machines,",
                        help='Prompt text for generation')
    parser.add_argument('--max_tokens', type=int, default=60, help='Maximum tokens to generate')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--use_gqa', action='store_true', help='Use Grouped Query Attention')
    parser.add_argument('--use_mla', action='store_true', help='Use Multihead Latent Attention')
    parser.add_argument('--use_calculator', action='store_true', help='Enable calculator tool')
    
    args = parser.parse_args()

    # --- Setup MPS Optimized Environment ---
    device = optimize_for_apple_silicon()
    checkpoint_path = args.model_path

    # --- Configuration ---
    VOCAB_SIZE = 10000
    MAX_SEQ_LEN = args.seq_len
    N_EMBD = args.embd_dim
    N_LAYER = args.n_layer
    N_HEAD = args.n_head
    N_KV_HEADS = args.n_kv_head
    USE_GQA = args.use_gqa
    USE_ROPE = True
    USE_FLASH = True
    USE_MOE = False
    USE_MLA = args.use_mla
    USE_REASONING = False
    USE_CALCULATOR = args.use_calculator
    GRADIENT_CHECKPOINTING = True
    
    # Enable mixed precision
    USE_MIXED_PRECISION = args.mixed_precision
    
    # Batch size and optimization settings
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS
    NUM_WORKERS = 4  # Data loader workers

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.001
    MAX_GRAD_NORM = 0.1
    
    # --- Tokenizer Loading ---
    tokenizer_name = "gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set PAD token to EOS token: {tokenizer.pad_token}")
        VOCAB_SIZE = tokenizer.vocab_size
    except Exception as e:
        print(f"Error loading tokenizer '{tokenizer_name}': {e}")
        exit()

    # --- Model Configuration ---
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        dropout_prob=0.1,
        alibi=False, use_rope=USE_ROPE,
        flash_attention=USE_FLASH,
        n_kv_heads=N_KV_HEADS if USE_GQA else None,
        use_gqa=USE_GQA,
        use_rwkv=False, use_ssm=False,
        use_moe=USE_MOE,
        num_experts=4, top_k_experts=2,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        mla_n_latent=8, use_mla=USE_MLA,
        reasoning_steps=1, use_reasoning_tracker=USE_REASONING,
        use_algorithmic_reasoner=False,
        use_calculator=USE_CALCULATOR,
    )
    logger.info(f"Model Config: {config.__dict__}")

    # --- Model Initialization (Use MPS-optimized model) ---
    model = MPSOptimizedEnhancedGPTLMHeadModel(config)
    model.to(device)

    # --- Load checkpoint if it exists ---
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            ckpt_dict = torch.load(checkpoint_path, map_location=device)
            # If it's a dictionary with model_state_dict key
            if isinstance(ckpt_dict, dict) and 'model_state_dict' in ckpt_dict:
                model.load_state_dict(ckpt_dict['model_state_dict'])
                print(f"Loaded checkpoint from epoch {ckpt_dict.get('epoch', 'unknown')}")
            else:
                # If it's just a state_dict
                model.load_state_dict(ckpt_dict)
                print("Loaded raw state_dict.")
        except Exception as e:
            print(f"Warning: failed to load checkpoint: {e}")
    
    # --- Training Mode ---
    if args.mode in ['train', 'both']:
        print("\n=== Starting MPS-Optimized Training ===")
        
        # --- Load Dataset ---
        from datasets import load_dataset, concatenate_datasets

        datasets_to_load = [
            ("tiny_stories", "train"),
            ("wikitext", "wikitext-103-raw-v1")
        ]

        loaded_datasets = []
        for name, config_or_split in datasets_to_load:
            try:
                if name == "wikitext":
                    ds = load_dataset(name, config_or_split, split="train")
                else:
                    ds = load_dataset(name, split=config_or_split)
                loaded_datasets.append(ds)
                print(f"Loaded {name} ({config_or_split}) → {len(ds)} samples")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        
        if not loaded_datasets:
            raise RuntimeError("Failed to load any datasets!")

        # --- Merge and Shuffle All Datasets ---
        combined_dataset = concatenate_datasets(loaded_datasets)
        combined_dataset = combined_dataset.filter(lambda ex: ex.get('text', '').strip() != "")
        combined_dataset = combined_dataset.shuffle(seed=42)

        print(f"Combined dataset has {len(combined_dataset)} examples.")
        
        # --- Create Optimized Dataset ---
        dataset_cache_path = "tokenized_dataset.pt"
        
        if os.path.exists(dataset_cache_path):
            print(f"Loading tokenized dataset from {dataset_cache_path}")
            train_dataset = torch.load(dataset_cache_path, weights_only=False)
        else:
            print("Tokenizing dataset...")
            train_dataset = OptimizedDataset(combined_dataset, tokenizer, MAX_SEQ_LEN, preload=False)
            torch.save(train_dataset, dataset_cache_path)
            print(f"Saved tokenized dataset to {dataset_cache_path}")
        
        # --- Create MPS-optimized DataLoader ---
        train_loader = MPSDataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True
        )
        
        # --- Training Setup ---
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Add learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * args.epochs,
            eta_min=1e-6
        )
        
        # --- Start periodic checkpoint saving ---
        save_thread = save_checkpoint_periodically_optimized(model, checkpoint_path, interval_sec=120)
        
        # --- Train model ---
        train_model_optimized(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epochs=args.epochs,
            device=device,
            accumulation_steps=ACCUMULATION_STEPS,
            scheduler=scheduler,
            mixed_precision=USE_MIXED_PRECISION,
            max_grad_norm=MAX_GRAD_NORM,
            checkpoint_path=checkpoint_path,
            log_interval=10,
            eval_interval=500
        )
        
        # --- Save final model ---
        final_model_path = checkpoint_path.replace('.pth', '_final.pt')
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, final_model_path)
        
        print(f"Training complete! Final model saved to {final_model_path}")
    
    # --- Prompt Mode ---
    if args.mode in ['prompt', 'both']:
        print("\n--- Testing Model with Prompt ---")
        
        # Make sure model is in eval mode
        model.eval()
        
        generated_text = prompt_model_optimized(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            use_calculator_tool=USE_CALCULATOR,
            device=device
        )
        
        print("\n=== Final Generated Text ===")
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {generated_text}")