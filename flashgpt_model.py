
# flashgpt_model.py
# This module contains the GPT model definitions, MPS optimization utilities,
# training functions, and other components for the FlashGPT project.

import os
import time
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset # Using datasets library for example data (dependency needed if OptimizedDataset uses it)
from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint
import sympy as sp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase # For tokenization
import threading
from pathlib import Path
import gc # For garbage collection in training loop

# --- Configuration ---

# Setup logging
# Note: It's often better to configure logging in the main script,
# but keeping it here as it was in the original code.
logging.basicConfig(
    filename='training_log.txt', # Consider making this configurable
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) # Use __name__ for module logger

# Setting up device (will be determined dynamically or passed as argument)
# We'll define functions that determine the device later.
device = None # Placeholder, will be set by optimize_for_apple_silicon or passed in functions


# MPS related stuff ###################################################
# --- Configuration for MPS Optimization ---

def optimize_for_apple_silicon():
    """Configure PyTorch for optimal performance on Apple Silicon."""
    global device # Modify the global device variable
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
        try:
             # Note: current_allocated_memory might be 0 initially
             print(f"Initial allocated memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
        except Exception as e:
             print(f"Could not get MPS memory info: {e}")

        # Set autocast and benchmark flags (benchmark might not be needed/available for MPS)
        # torch.backends.mps.enable_cudnn_benchmark = True # No cudnn on MPS
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("MPS/CUDA device not found, using CPU")

    # Move model to device (should happen after model initialization)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        # torch.backends.cudnn.deterministic = True # Optional for reproducibility
        # torch.backends.cudnn.benchmark = False # Optional for reproducibility
        pass # Keep defaults for now

    # Return optimized device
    return device

# --- Mixed-Precision Training Helper ---

class MPSMixedPrecision:
    """Helper for mixed precision training on MPS"""
    def __init__(self, enabled=True, dtype=torch.float16):
        # Ensure enabled is only True if MPS is available
        self.enabled = enabled and torch.backends.mps.is_available()
        self.dtype = dtype
        # Initialize scaler only if enabled
        self.scaler = torch.amp.GradScaler(enabled=self.enabled) if self.enabled else None
        if self.enabled:
            print(f"MPS Mixed Precision enabled with dtype {self.dtype}")
        else:
            print("MPS Mixed Precision disabled (MPS not available or not enabled)")


    def __enter__(self):
        if self.enabled:
            # Use autocast context manager for MPS
            return torch.autocast(device_type="mps", dtype=self.dtype, enabled=True)
        else:
            # Return a dummy context manager if not enabled
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyContext()

    def __exit__(self, *args):
        # Exit autocast context implicitly
        pass

    def scale_loss(self, loss, optimizer):
        if self.enabled and self.scaler:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer, loss=None, clip_grad=None, model=None):
        if self.enabled and self.scaler:
            if clip_grad is not None and model is not None:
                # Unscale before clipping gradients
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard optimizer step if not using mixed precision
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
        self.num_workers = num_workers if num_workers > 0 else 0 # Ensure num_workers isn't negative
        self.shuffle = shuffle

        # For MPS, we need custom handling since pin_memory isn't supported directly
        if torch.backends.mps.is_available():
            print(f"Configuring MPS DataLoader (num_workers={self.num_workers})")
            # Create a standard dataloader without pin_memory
            # persistent_workers=True requires num_workers > 0
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
                prefetch_factor=3 if self.num_workers > 0 else None,  # Increase prefetch factor
                pin_memory=False # Explicitly False for MPS clarity
            )
        else:
            print(f"Configuring standard DataLoader (pin_memory=True, num_workers={self.num_workers})")
            # For other systems, use standard pinned memory if CUDA available
            pin_memory_enabled = torch.cuda.is_available()
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=pin_memory_enabled,
                persistent_workers=True if self.num_workers > 0 else False,
                prefetch_factor=2 if self.num_workers > 0 else None
            )

    def __iter__(self):
        # Return the iterator of the underlying DataLoader
        return iter(self.dataloader)

    def __len__(self):
        # Return the length of the underlying DataLoader
        return len(self.dataloader)

# --- Memory-Optimized Dataset ---

class OptimizedDataset(torch.utils.data.Dataset):
    """Basic Dataset wrapper for tokenization (can be optimized further)."""
    def __init__(self, dataset, tokenizer, max_length, preload=False):
        import torch
        self.dataset = dataset # Expects a list/iterable of dictionaries with 'text' or 'content'
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preload = preload
        self.preloaded_items = []

        # Preload a subset for faster access if requested
        if self.preload:
            print("Preloading dataset subset...")
            num_to_preload = min(10000, len(self.dataset)) # Limit preloading
            for idx in tqdm(range(num_to_preload), desc="Preloading"):
                try:
                    self.preloaded_items.append(self._process_item(idx))
                except Exception as e:
                    print(f"Warning: Skipping item {idx} during preload due to error: {e}")
            print(f"Preloaded {len(self.preloaded_items)} items")

    def _process_item(self, idx):
        # Handle potential variations in dataset structure
        item = self.dataset[idx]
        text = item.get('text', item.get('content', '')) # Safely get text
        if not text:
             logger.warning(f"Empty text found at index {idx}. Skipping.")
             # Return placeholder or raise error? Returning empty tensors might cause issues.
             # Let's return tensors with pad tokens.
             input_ids = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
             attention_mask = torch.zeros((self.max_length,), dtype=torch.long)
             labels = torch.full((self.max_length,), -100, dtype=torch.long) # Ignore index for loss
             return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


        encodings = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                  padding="max_length", return_tensors="pt")
        input_ids = encodings['input_ids'].squeeze(0) # Remove batch dim
        attention_mask = encodings['attention_mask'].squeeze(0) # Remove batch dim
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.preload and idx < len(self.preloaded_items):
            return self.preloaded_items[idx]
        # Process on-the-fly if not preloaded or index is out of preload range
        return self._process_item(idx)


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
                 use_calculator: bool = True,
                 # Added for ToT (conceptual)
                 use_tree_of_thought: bool = False): # Added config flag for ToT

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
        self.use_tree_of_thought = use_tree_of_thought # Store ToT config flag

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

# Helper function: Chunked matrix multiplication (Included as it was in the original code)
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
        if not self.initialized_device or self.device != input_tensor.device:
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

        # Basic validation
        if B > self.max_batch_size:
            logger.error(f"KV Cache Error: Batch size {B} exceeds max cache size {self.max_batch_size}. Resizing needed or error.")
            # Option 1: Raise error
            raise ValueError(f"Batch size {B} exceeds max cache size {self.max_batch_size}")
            # Option 2: Attempt resize (could lead to OOM)
            # self.reset(batch_size=B) # This would clear existing cache! Not ideal mid-generation.

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
                logger.debug(f"KV Cache: Overwriting cache at pos {start_pos} with {T_new} tokens")

        # Check for cache overflow and apply sliding window
        if end_pos > self.max_seq_len:
            # Calculate how many tokens to discard from the beginning
            to_discard = end_pos - self.max_seq_len
            logger.warning(f"KV Cache overflow: max_seq_len={self.max_seq_len}, needed={end_pos}. Discarding {to_discard} oldest tokens.")

            # Shift existing cache content left
            self.k_cache = torch.roll(self.k_cache, shifts=(-to_discard,), dims=2)
            self.v_cache = torch.roll(self.v_cache, shifts=(-to_discard,), dims=2)

            # Adjust positions for insertion at the *new* end of the shifted cache
            start_pos = self.max_seq_len - T_new
            end_pos = self.max_seq_len

            # Update current_seq_len to reflect full cache
            self.current_seq_len = self.max_seq_len
        else:
             # Only update current_seq_len if appending or extending beyond current length
             if position is None or end_pos > self.current_seq_len:
                  self.current_seq_len = max(self.current_seq_len, end_pos)


        # Validate final positions before writing
        if not (0 <= start_pos < end_pos <= self.max_seq_len):
             raise ValueError(f"KV Cache Error: Invalid cache write indices: start={start_pos}, end={end_pos}, max={self.max_seq_len}")


        # Update the cache slice
        try:
             self.k_cache[:B, :, start_pos:end_pos] = k
             self.v_cache[:B, :, start_pos:end_pos] = v
             logger.debug(f"KV Cache: Updated pos {start_pos}:{end_pos}. New seq_len={self.current_seq_len}")
        except IndexError as e:
             logger.error(f"KV Cache Error during write: {e}. Shapes - k_cache: {self.k_cache.shape}, k: {k.shape}, B: {B}, start: {start_pos}, end: {end_pos}")
             raise

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

        # Validate batch size against cache size
        if current_batch_size > self.max_batch_size:
            logger.error(f"KV Cache Get Error: Requested batch size {current_batch_size} > cache max {self.max_batch_size}")
            raise ValueError(f"Requested batch size {current_batch_size} exceeds max cache size {self.max_batch_size}")

        # Return the relevant slice of the cache
        return (self.k_cache[:current_batch_size, :, :self.current_seq_len],
                self.v_cache[:current_batch_size, :, :self.current_seq_len])

    def reset(self, batch_size: int = None):
        """Reset the cache, optionally resizing batch dimension if needed."""
        new_batch_size = batch_size if batch_size is not None else self.max_batch_size

        if self.initialized_device and new_batch_size != self.k_cache.shape[0]:
             # Reinitialize if batch size changes (potentially expensive)
             logger.info(f"Resetting KV Cache and resizing batch dim from {self.max_batch_size} to {new_batch_size}")
             self.max_batch_size = new_batch_size
             self.k_cache = torch.zeros(self.max_batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim,
                                       dtype=self.dtype, device=self.device)
             self.v_cache = torch.zeros(self.max_batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim,
                                       dtype=self.dtype, device=self.device)
        elif self.initialized_device:
            # Just clear the existing cache contents
            self.k_cache.zero_()
            self.v_cache.zero_()
            logger.debug(f"KV Cache: Reset content (batch size {self.max_batch_size} unchanged)")
        # Else: not initialized yet, reset does nothing until first update

        self.current_seq_len = 0


# --- Core Model Architecture Components ---

# RMSNorm (Root Mean Square Layer Normalization)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6): # Use common 1e-6 epsilon
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Compute Root Mean Square along the feature dimension
        # Use float32 for stability during calculation
        return x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x).type_as(x) # Cast back to original input type
        return output * self.weight

# SwiGLU Feed-Forward Block
class SwiGLU(nn.Module):
    """ SwiGLU FFN Layer """
    def __init__(self, hidden_dim: int, expansion_factor: float = 8/3, dropout_prob: float = 0.0):
        super().__init__()
        # Calculate intermediate dimension, often making it multiple of 256
        intermediate_dim = int(expansion_factor * hidden_dim)
        multiple_of = 256
        intermediate_dim = multiple_of * ((intermediate_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False) # Gated component
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout_prob) # Added dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply SiLU activation to the first projection
        swish_gate = F.silu(self.w1(x))
        # Element-wise multiply with the third projection (gate)
        gated_output = swish_gate * self.w3(x)
        # Apply final projection
        output = self.w2(gated_output)
        # Apply dropout
        output = self.dropout(output)
        return output

# ALiBi Positional Bias (Attention with Linear Biases)
def build_alibi_tensor(n_heads: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Builds the ALiBi tensor for biasing attention scores."""
    def get_slopes(n: int) -> list[float]:
        # Standard ALiBi slope calculation (handles powers of 2 and non-powers of 2)
        try:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            base = torch.tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=torch.float32)
            powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.float32)
            slopes = torch.pow(base, powers)

            if closest_power_of_2 != n: # Handle cases where n is not a power of 2
                extra_base = torch.tensor(2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=torch.float32)
                # Corrected extra_powers calculation for non-power-of-2 heads
                extra_powers = torch.arange(1, 1 + 2 * (n - closest_power_of_2), 2, dtype=torch.float32)
                slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

            return slopes.tolist() # Return as list
        except Exception as e:
            # Fallback if calculation fails (simple geometric progression)
            logger.error(f"Error calculating ALiBi slopes: {e}. Using simpler power-of-2 calculation.")
            start = 2 ** (-8.0 / n) # Example fallback start value
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

    slopes = torch.tensor(get_slopes(n_heads), device=device, dtype=dtype) # (n_heads,)
    # Create relative distances (causal mask style)
    relative_position = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1) # (seq_len, seq_len)
    # ALiBi uses negative relative position * slope
    # Result shape needs to be (1, n_heads, seq_len, seq_len) for broadcasting
    alibi = slopes.unsqueeze(1).unsqueeze(2) * relative_position.abs().mul(-1).unsqueeze(0)
    return alibi.unsqueeze(0) # Add batch dimension


# --- Modern Positional Embeddings ---

# Rotary Position Embeddings (RoPE)
class RotaryPositionEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000, device: torch.device = None):
        super().__init__()
        self.dim = dim
        self.max_seq_len_cached = 0 # Track cache size
        self.base = base
        self.device = device # Store target device

        # Precompute inverse frequencies (theta_i) - moved to _build_cache
        # self.register_buffer("inv_freq", None, persistent=False)
        # self.register_buffer("cos_cached", None, persistent=False)
        # self.register_buffer("sin_cached", None, persistent=False)
        self._build_cache(max_seq_len) # Initial cache build

    def _build_cache(self, seq_len: int):
        # Only rebuild if needed seq_len > cached length or device changes
        current_device = self.device if self.device else torch.device('cpu') # Default to CPU if not set
        # if seq_len <= self.max_seq_len_cached and hasattr(self, 'inv_freq') and self.inv_freq.device == current_device:
        #     return # Cache is sufficient

        logger.info(f"Building RoPE cache for seq_len {seq_len} on device {current_device}")
        self.max_seq_len_cached = seq_len
        self.device = current_device # Update stored device

        # Calculate inverse frequencies (theta_i)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=self.device).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # Outer product: (seq_len, dim/2)

        # Different from paper ordering? -> Corrected einsum and emb generation
        # emb = torch.cat((freqs, freqs), dim=-1) # Shape: (seq_len, dim)
        # self.register_buffer("cos_cached", emb.cos(), persistent=False)
        # self.register_buffer("sin_cached", emb.sin(), persistent=False)

        # Store freqs directly (no duplication needed before applying)
        self.register_buffer("cos_cached", freqs.cos().to(inv_freq.dtype), persistent=False) # Shape: (seq_len, dim/2)
        self.register_buffer("sin_cached", freqs.sin().to(inv_freq.dtype), persistent=False) # Shape: (seq_len, dim/2)


    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Applies rotary embeddings to the input tensor x."""
        # x: (..., seq_len, head_dim)
        # cos, sin: (seq_len, head_dim/2)
        d = x.shape[-1]
        cos = cos[:, : d // 2] # Ensure dimensions match if dim is not the RoPE dim
        sin = sin[:, : d // 2]

        # Reshape x to separate real/imaginary parts (pairs of features)
        # x = (x1, x2, x3, x4, ...) -> x_half1 = (x1, x3, ...), x_half2 = (x2, x4, ...)
        x_half1 = x[..., 0 : d : 2]
        x_half2 = x[..., 1 : d : 2]

        # Unsqueeze cos/sin to match batch/head dimensions if necessary
        # Example: if x is (B, H, T, D), cos/sin need to be (1, 1, T, D/2) for broadcasting
        while cos.ndim < x.ndim -1:
             cos = cos.unsqueeze(0)
             sin = sin.unsqueeze(0)


        # Apply rotation using complex number multiplication formula:
        # rotated_x1 = x1 * cos - x2 * sin
        # rotated_x2 = x1 * sin + x2 * cos
        rotated_x_half1 = x_half1 * cos - x_half2 * sin
        rotated_x_half2 = x_half1 * sin + x_half2 * cos

        # Combine back: create tensor of shape (..., D) where pairs are interleaved
        rotated_x = torch.empty_like(x)
        rotated_x[..., 0 : d : 2] = rotated_x_half1
        rotated_x[..., 1 : d : 2] = rotated_x_half2

        return rotated_x

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
        # Ensure cache is built for the required sequence length and device
        self._build_cache(seq_len)

        # Retrieve cached cos/sin values for the current sequence length
        # Slicing ensures we only use the relevant part of the cache
        cos = self.cos_cached[:seq_len] # Shape: (seq_len, dim/2)
        sin = self.sin_cached[:seq_len] # Shape: (seq_len, dim/2)

        # Apply rotary embeddings to q and k
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
        assert self.head_dim * self.n_head == self.n_embd, "n_embd must be divisible by n_head"
        self.dropout_prob = config.dropout_prob
        self.use_flash_attention = config.flash_attention
        self.use_rope = config.use_rope
        self.use_alibi = config.alibi
        self.max_seq_len = config.max_seq_len

        if self.use_rope:
            # Initialize RoPE - try to infer device, fallback to CPU
            try:
                rope_device = next(self.parameters()).device
            except StopIteration: # No parameters yet, default to CPU or MPS if available
                rope_device = optimize_for_apple_silicon() if torch.backends.mps.is_available() else torch.device('cpu')
            self.rope = RotaryPositionEmbeddings(self.head_dim, self.max_seq_len, device=rope_device)

        # KV cache needs n_kv_heads (number of heads for K and V)
        self.n_kv_heads = config.n_kv_heads if config.use_gqa else config.n_head
        self.kv_cache: KVCache | None = None # Initialize later if use_cache=True

        # Output projection and dropout common to all attention types
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout_prob)
        self.resid_dropout = nn.Dropout(self.dropout_prob) # Dropout after residual connection

    def _init_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """Initializes the KV cache if needed or resizes if batch size changes."""
        if self.kv_cache is None:
             logger.info(f"Initializing KV Cache: B={batch_size}, T={self.max_seq_len}, heads={self.n_kv_heads}, dim={self.head_dim}")
             self.kv_cache = KVCache(batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim, dtype=dtype, device=device)
        elif self.kv_cache.max_batch_size != batch_size or self.kv_cache.device != device:
             logger.warning(f"Re-initializing KV Cache due to change: B={batch_size}(old {self.kv_cache.max_batch_size}), device={device}(old {self.kv_cache.device})")
             # Reset and potentially resize
             self.kv_cache.reset(batch_size=batch_size)
             # Ensure device is correct after reset
             if self.kv_cache.device != device:
                  self.kv_cache = KVCache(batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim, dtype=dtype, device=device)
        else:
             # Reset content for new sequence if batch size/device are same
             self.kv_cache.reset()

    def _process_kv_cache(self, k: torch.Tensor, v: torch.Tensor, use_cache: bool, position_ids: torch.LongTensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Updates and retrieves from KV cache."""
        if not use_cache:
            return k, v

        B, _, T_new, _ = k.shape
        # Initialize cache if this is the first call with use_cache=True
        if self.kv_cache is None:
            self._init_kv_cache(B, k.dtype, k.device)
        # If cache exists, ensure it matches current batch size/device, reset content only
        elif self.kv_cache.max_batch_size != B or self.kv_cache.device != k.device:
             self._init_kv_cache(B, k.dtype, k.device) # Re-init if mismatch
        # else: cache exists and matches, content will be reset by update/get logic implicitly


        # Determine the position for cache update based on position_ids or cache state
        cache_position = None
        if position_ids is not None:
            # Use the minimum position_id as the starting point for the update
            # Assumes position_ids are contiguous for the new tokens
            cache_position = position_ids.min().item()
        else:
             # If no position_ids, assume we are appending to the current cache length
             cache_position = self.kv_cache.current_seq_len

        # Update cache with new keys/values at the determined position
        self.kv_cache.update(k, v, position=cache_position)

        # Get the full sequence from cache up to the new length
        k_full, v_full = self.kv_cache.get(current_batch_size=B)
        return k_full, v_full

    def _manual_attention(self, q, k, v, attention_mask, alibi_bias, is_causal):
        """Manual attention computation as fallback with stability improvements."""
        B, n_heads, T_q, head_dim = q.shape
        T_k = k.shape[-2]

        # Use float32 for score calculation for stability
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)

        # Compute attention scores
        attn_scores = torch.matmul(q_f32, k_f32.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply ALiBi bias if needed (before masks)
        if self.use_alibi and alibi_bias is not None:
            # Ensure bias matches score dimensions (T_q, T_k)
            if alibi_bias.size(-2) >= T_q and alibi_bias.size(-1) >= T_k:
                # Slice or pad bias if necessary
                alibi_bias_sliced = alibi_bias[:, :, :T_q, :T_k]
                attn_scores = attn_scores + alibi_bias_sliced.to(torch.float32) # Add bias
            else:
                logger.warning(f"ALiBi bias shape {alibi_bias.shape} incompatible with attn scores {attn_scores.shape} (Tq={T_q}, Tk={T_k}). Skipping ALiBi.")

        # Apply causal mask if needed (for training or full sequence processing)
        if is_causal: # Typically True when T_q == T_k during training/prompt processing
            # Create causal mask dynamically based on T_q and T_k
            # Mask needs to be False where attention is allowed
            causal_mask = torch.ones(T_q, T_k, device=q.device, dtype=torch.bool).tril(diagonal=0)
            attn_scores = attn_scores.masked_fill(~causal_mask, -float('inf')) # Mask out upper triangle

        # Apply padding attention mask if provided
        if attention_mask is not None:
            # Expected shape (B, T_k) or (B, 1, 1, T_k) or (B, 1, T_q, T_k)
             if attention_mask.dim() == 2: # (B, T_k) -> (B, 1, 1, T_k)
                  mask_expanded = attention_mask.view(B, 1, 1, T_k)
             elif attention_mask.dim() == 3: # Maybe (B, T_q, T_k) -> (B, 1, T_q, T_k) ? Unlikely. Assume (B, 1, T_k)?
                  mask_expanded = attention_mask.unsqueeze(1) # (B, 1, T_q/1?, T_k)
             elif attention_mask.dim() == 4:
                  mask_expanded = attention_mask
             else:
                  raise ValueError(f"Unexpected attention mask shape: {attention_mask.shape}")

             # Ensure mask length matches key length Tk
             if mask_expanded.shape[-1] != T_k:
                 logger.warning(f"Attention mask length {mask_expanded.shape[-1]} != Key length {T_k}. Adjusting mask.")
                 if mask_expanded.shape[-1] > T_k:
                     mask_expanded = mask_expanded[..., :T_k]
                 else: # Pad mask if shorter (assume padding allows attention)
                     pad_width = T_k - mask_expanded.shape[-1]
                     mask_expanded = F.pad(mask_expanded, (0, pad_width), value=1) # Pad with 1 (allow attn)


             # Ensure mask query length matches query length Tq if mask is 4D
             if mask_expanded.dim() == 4 and mask_expanded.shape[-2] != T_q:
                  if mask_expanded.shape[-2] == 1: # Broadastable mask (e.g., from padding only)
                       pass # Okay
                  else:
                       logger.warning(f"Attention mask query length {mask_expanded.shape[-2]} != Query length {T_q}. Adjusting mask.")
                       if mask_expanded.shape[-2] > T_q:
                           mask_expanded = mask_expanded[..., :T_q, :]
                       else: # Pad mask query dimension (highly unusual)
                            pad_height = T_q - mask_expanded.shape[-2]
                            mask_expanded = F.pad(mask_expanded, (0, 0, 0, pad_height), value=1)


             # Apply mask: mask value of 0 means mask out (fill with -inf)
             # Ensure mask is bool for masked_fill
             attn_scores = attn_scores.masked_fill(mask_expanded == 0, -float('inf'))


        # Compute attention weights (softmax)
        # Use float32 for stability
        attn_weights = F.softmax(attn_scores, dim=-1).type_as(q) # Cast back to original type

        # Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)

        # Compute output (weighted sum of values)
        attn_output = torch.matmul(attn_weights, v) # (B, n_heads, T_q, head_dim)

        # Sanity check for NaNs in output
        if torch.isnan(attn_output).any():
            logger.error("NaN detected in manual attention output!")
            # Option: replace NaNs with zeros? Might hide underlying issues.
            # attn_output = torch.nan_to_num(attn_output, nan=0.0)
            raise FloatingPointError("NaN detected in manual attention output")

        return attn_output

    def _compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           attention_mask: torch.Tensor | None, alibi_bias: torch.Tensor | None) -> torch.Tensor:
        """Computes attention output using FlashAttention or manual implementation."""
        B, n_heads, T_q, head_dim = q.shape
        T_k = k.shape[-2]

        # Determine if causal masking is needed based on sequence lengths and context
        # Causal is typically needed when T_q == T_k (processing a full sequence)
        # or when generating the very first token from a prompt (T_q=1, T_k=PromptLength).
        # It's NOT needed when generating subsequent tokens (T_q=1, T_k=PromptLength+PreviousGenerated).
        is_causal = T_q == T_k # Default: assume causal if lengths match
        is_generation_step = T_q == 1 and T_k > 1 # True if generating single token with cache

        # Flash Attention Requirements:
        # - No ALiBi bias
        # - No explicit attention_mask (padding handled separately or assumes none)
        # - Compatible hardware/pytorch version
        # - Dtype often needs to be fp16 or bf16
        can_use_flash = (
            self.use_flash_attention
            and not self.use_alibi # Flash Attn v2 supports ALiBi, but basic F.sdpa might not handle it easily
            and attention_mask is None # F.sdpa can take a mask, but simpler without
            and hasattr(F, 'scaled_dot_product_attention')
        )
        
        # Check for NaN/Inf in inputs before attention
        if __debug__: # Only run checks during development/debugging
             for name, tensor in [('q', q), ('k', k), ('v', v)]:
                  if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                       logger.warning(f"NaN/Inf detected in input '{name}' to attention. Shape: {tensor.shape}")
                       # Consider clamping or replacing NaNs here if it's a recurring issue
                       # tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

        if can_use_flash:
            # Prepare inputs for F.scaled_dot_product_attention if needed (B, T, H, D_h)
            # Q, K, V should have shape (B, H, T, D_h) for our manual path,
            # but F.sdpa prefers (B, T, H, D_h). Let's keep our shape and transpose inside if needed.

            # Note: F.sdpa handles causal masking internally with `is_causal=True`
            # If it's a generation step (T_q=1, T_k > 1), we should NOT use causal masking.
            sdpa_is_causal = is_causal and not is_generation_step

            try:
                 # logger.debug(f"Using Flash Attention (F.sdpa): Tq={T_q}, Tk={T_k}, is_causal={sdpa_is_causal}")
                 # Use dropout only during training
                 dropout_p = self.attn_dropout.p if self.training else 0.0
                 attn_output = F.scaled_dot_product_attention(
                      q, k, v,
                      attn_mask=None, # Explicit mask handled manually or not used with flash
                      dropout_p=dropout_p,
                      is_causal=sdpa_is_causal
                 )
                 return attn_output
            except Exception as e:
                 # Catch potential errors like dtype/device incompatibility or OOM
                 logger.warning(f"Flash Attention (F.sdpa) failed: {e}. Falling back to manual attention. Tq={T_q}, Tk={T_k}, is_causal={sdpa_is_causal}")
                 # Fallthrough to manual attention

        # Manual Attention Calculation (Fallback or if flash cannot be used)
        # logger.debug(f"Using Manual Attention: Tq={T_q}, Tk={T_k}, is_causal={is_causal}, is_gen={is_generation_step}")
        # Pass `is_causal=True` only if processing a full sequence or the very first token generation
        manual_is_causal = is_causal # Use the originally determined causality for manual path
        return self._manual_attention(q, k, v, attention_mask, alibi_bias, manual_is_causal)


    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the forward method")


# MultiHead Self Attention (Standard)
class MultiHeadSelfAttention(BaseAttention):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        if config.use_gqa:
            logger.warning("MHA initialized, but GQA is enabled in config. Ensure this is intended.")
        assert config.n_embd % config.n_head == 0

        # Projections for Q, K, V (all map from n_embd to n_embd)
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
                alibi_bias: torch.Tensor | None = None, use_cache: bool = False,
                position_ids: torch.LongTensor | None = None) -> torch.Tensor:
        B, T, C = x.size() # Input shape: Batch, Sequence Length, Embedding Dim

        # 1. Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape for multi-head: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # K has n_head heads
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # V has n_head heads

        # 3. Apply RoPE if enabled (before caching)
        if self.use_rope:
            # Determine sequence length for RoPE based on input T or cache state
            current_seq_len = T # Length of the new input sequence
            q, k = self.rope(q, k, current_seq_len)

        # 4. Handle KV Caching (operates on k and v)
        # Note: k, v passed here have n_head heads (same as q)
        k_cache, v_cache = self._process_kv_cache(k, v, use_cache, position_ids)
        # k_cache, v_cache might have different T dimension (T_full) if cache is used

        # 5. Compute attention using potentially cached K/V
        # Q is from current input, K/V are from cache (or current input if no cache)
        attn_output = self._compute_attention(q, k_cache, v_cache, attention_mask, alibi_bias)
        # attn_output shape: (B, n_head, T_q, head_dim), where T_q = T (input sequence length)

        # 6. Combine heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C) # (B, T_q, C)
        output = self.resid_dropout(self.out_proj(attn_output)) # Apply output projection and dropout

        return output


# Grouped Query Attention (GQA)
class GroupedQueryAttention(BaseAttention):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        if not config.use_gqa:
             logger.warning("GQA class initialized, but use_gqa=False in config.")
        assert config.n_head % config.n_kv_heads == 0, "n_head must be divisible by n_kv_heads for GQA"
        self.n_kv_heads = config.n_kv_heads
        self.num_query_groups = config.n_head // self.n_kv_heads

        # Projections: Q maps to n_head, K/V map to n_kv_heads
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False) # Output dim C = n_head * head_dim
        self.k_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False) # Output dim C_kv
        self.v_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=False) # Output dim C_kv

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeats KV heads to match query heads: (B, n_kv, T, head_dim) -> (B, n_q, T, head_dim)"""
        B, n_kv, T, head_dim = x.shape
        if n_rep == 1:
            return x
        # Expand and reshape: insert group dim, expand, reshape to flatten kv*rep
        return (
            x.unsqueeze(2) # (B, n_kv, 1, T, head_dim)
            .expand(B, n_kv, n_rep, T, head_dim) # (B, n_kv, n_rep, T, head_dim)
            .reshape(B, n_kv * n_rep, T, head_dim) # (B, n_q, T, head_dim)
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
                alibi_bias: torch.Tensor | None = None, use_cache: bool = False,
                position_ids: torch.LongTensor | None = None) -> torch.Tensor:
        B, T, C = x.size()

        # 1. Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape Q: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Reshape K/V: (B, T, C_kv) -> (B, n_kv_head, T, head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE if enabled (before caching)
        if self.use_rope:
            current_seq_len = T
            q, k = self.rope(q, k, current_seq_len) # Apply RoPE to Q and K (with fewer heads)

        # 4. Handle KV Caching (operates on the n_kv_heads dimension)
        k_cache, v_cache = self._process_kv_cache(k, v, use_cache, position_ids)
        # k_cache, v_cache have shape (B, n_kv_heads, T_full, head_dim)

        # 5. Repeat K/V heads to match query heads for attention calculation
        # Only repeat if cache is used (T dimension might change) or if T_q != T_k
        k_rep = self._repeat_kv(k_cache, self.num_query_groups) # (B, n_head, T_full, head_dim)
        v_rep = self._repeat_kv(v_cache, self.num_query_groups) # (B, n_head, T_full, head_dim)

        # 6. Compute attention using Q and repeated K/V
        # Ensure ALiBi bias matches the query heads (n_head) if used
        attn_output = self._compute_attention(q, k_rep, v_rep, attention_mask, alibi_bias)
        # attn_output shape: (B, n_head, T_q, head_dim)

        # 7. Combine heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C) # (B, T_q, C)
        output = self.resid_dropout(self.out_proj(attn_output))

        return output


# RWKV-style Linear Attention (Simplified Placeholder)
class RWKVAttention(nn.Module):
    """Simplified RWKV-style time-mixing block (operates per-channel). Needs review for correctness."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        logger.warning("RWKVAttention is a simplified placeholder and may not match official implementations.")

        # Learnable time decay and first token bias (per channel)
        # Initialize decay close to zero for stability (large negative value)
        self.time_decay = nn.Parameter(torch.ones(self.n_embd) * -5.0)
        self.time_first = nn.Parameter(torch.randn(self.n_embd) * 0.1) # Small random init

        # Projections (simplified compared to full RWKV block with R, K, V gates)
        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False) # 'R' gate
        self.output = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # State buffers (initialized on first forward)
        # State shape needs to be (Batch, EmbDim) for per-channel recurrence
        self.register_buffer('state_a', torch.Tensor(), persistent=False) # Numerator state part
        self.register_buffer('state_b', torch.Tensor(), persistent=False) # Denominator state part (often just exp(k))
        self.register_buffer('state_p', torch.Tensor(), persistent=False) # Max K state for numeric stability
        self.state_initialized = False

    def _init_state(self, B: int, dtype: torch.dtype, device: torch.device):
        """Initialize state tensors for the batch."""
        # logger.debug(f"Initializing RWKV state for B={B}, Dtype={dtype}, Device={device}")
        self.state_a = torch.zeros(B, self.n_embd, dtype=dtype, device=device)
        self.state_b = torch.zeros(B, self.n_embd, dtype=dtype, device=device)
        # Initialize p to -inf for correct max calculation on first step
        self.state_p = torch.full((B, self.n_embd), -float('inf'), dtype=dtype, device=device)
        self.state_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Initialize state if needed or if batch size/device changes
        if not self.state_initialized or self.state_a.shape[0] != B or self.state_a.device != x.device or self.state_a.dtype != x.dtype:
             self._init_state(B, x.dtype, x.device)

        # --- Get projections ---
        k = self.key(x)   # (B, T, C)
        v = self.value(x) # (B, T, C)
        r = torch.sigmoid(self.receptance(x)) # (B, T, C) - Receptance gate

        # --- Prepare time parameters ---
        # Use float() for exp calculation then cast back
        # Clamp decay: large positive decay (-value) means fast decay, large negative means slow decay
        w = torch.exp(-torch.exp(self.time_decay.float())).type_as(x) # (C,) - Decay factor per step
        u = self.time_first.type_as(x) # (C,) - Bonus for first token

        # --- Recurrent calculation over time steps ---
        outputs = []
        # Use local variables for state to avoid modifying buffers directly in loop
        # Ensure state is contiguous and on the correct device/dtype
        state_a, state_b, state_p = self.state_a, self.state_b, self.state_p

        for t in range(T):
            kt, vt, rt = k[:, t], v[:, t], r[:, t] # (B, C)

            # WKV calculation with numeric stability (using max trick `p`)
            max_p = torch.maximum(state_p, kt + u if t == 0 else kt) # Max(prev_max, current_k + bonus_if_first)
            exp_kt_p = torch.exp((kt + u if t == 0 else kt) - max_p) # exp(k_t - p_t)
            exp_prev_state_p = torch.exp(state_p - max_p + self.time_decay.type_as(x)) # exp(p_{t-1} - p_t -w) where w=-time_decay

            # Calculate wkv = ( numerator / denominator )
            wkv_num = (state_a * exp_prev_state_p) + (exp_kt_p * vt)
            wkv_den = (state_b * exp_prev_state_p) + exp_kt_p

            wkv = wkv_num / torch.clamp(wkv_den, min=1e-8) # Avoid division by zero

            # Update states for the next time step
            state_a = (state_a * w) + (kt * vt) # Update numerator state (simplified, review RWKV details)
            state_b = (state_b * w) + kt        # Update denominator state (simplified)
            state_p = max_p                     # Update max_k state

            # Apply receptance gate `r` to the wkv result
            outputs.append(rt * wkv)

        # --- Update persistent state buffers after loop ---
        # Detach state to prevent graph growth across forward calls if not intended
        self.state_a = state_a.detach()
        self.state_b = state_b.detach()
        self.state_p = state_p.detach()

        output = torch.stack(outputs, dim=1) # (B, T, C)
        return self.output(output) # Final output projection


    def reset_state(self):
        """Reset the recurrent state."""
        self.state_initialized = False
        # Clear tensor data by assigning empty tensors
        self.state_a = torch.Tensor()
        self.state_b = torch.Tensor()
        self.state_p = torch.Tensor()
        # logger.debug("RWKV state reset.")


# --- State Space Model (Mamba-style Selective SSM Placeholder) ---

class SelectiveSSM(nn.Module):
    """Simplified Selective State Space Model inspired by Mamba. Placeholder implementation."""
    def __init__(self, hidden_dim: int, ssm_state_dim: int = 16, ssm_expand_factor: int = 2,
                 dt_rank: str | int = 'auto', dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0):
        super().__init__()
        logger.warning("SelectiveSSM is a simplified placeholder and may not match official Mamba.")
        self.hidden_dim = hidden_dim
        self.ssm_state_dim = ssm_state_dim # N
        self.expand_factor = ssm_expand_factor
        self.expanded_dim = hidden_dim * ssm_expand_factor # E = D * expand

        # Input projections (x -> z, x_for_ssm) + (x -> dt, B, C params)
        self.in_proj = nn.Linear(hidden_dim, 2 * self.expanded_dim, bias=False) # Project to z, x_ssm
        self.dt_proj = nn.Linear(hidden_dim, self.expanded_dim, bias=True) # Project x to get dt parameter
        self.B_proj = nn.Linear(hidden_dim, self.expanded_dim * self.ssm_state_dim, bias=False) # Project x to B params
        self.C_proj = nn.Linear(hidden_dim, self.expanded_dim * self.ssm_state_dim, bias=False) # Project x to C params


        # Initialize dt bias for stability
        dt_init_std = dt_scale # Simplified init
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.bias, dt_init_std)
        else: # random init ~ Uniform or Normal
             nn.init.uniform_(self.dt_proj.bias, -dt_init_std, dt_init_std)


        # State space parameters (A is fixed/simple, B/C are data-dependent, D is learned)
        # A: diagonal matrix, initialized fixedly (common practice)
        A = torch.arange(1, ssm_state_dim + 1, dtype=torch.float32).unsqueeze(0).repeat(self.expanded_dim, 1) # (E, N)
        self.A_log = nn.Parameter(torch.log(A)) # Learn log(A)
        self.A_log._no_weight_decay = True # Often exclude A from weight decay

        # B and C are generated dynamically based on input x
        # D: Direct feedthrough parameter
        self.D = nn.Parameter(torch.ones(self.expanded_dim)) # (E,)
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.expanded_dim, hidden_dim, bias=False)

        # Store min/max dt
        self.dt_min = dt_min
        self.dt_max = dt_max

        # Placeholder for recurrent state during generation
        self.register_buffer("ssm_state", torch.Tensor(), persistent=False)
        self.state_initialized = False


    def _init_state(self, B: int, dtype: torch.dtype, device: torch.device):
        """Initialize state tensors for the batch."""
        # State is typically (B, E, N)
        self.ssm_state = torch.zeros(B, self.expanded_dim, self.ssm_state_dim, dtype=dtype, device=device)
        self.state_initialized = True

    def _selective_scan(self, u, delta, A, B, C, D, state):
        """
        Performs the selective scan operation using a recurrent loop.
        Args:
            u: Input sequence (B, T, E)
            delta: Time step (B, T, E)
            A: State transition matrix (log space) (E, N)
            B: Input matrix (B, T, E, N)
            C: Output matrix (B, T, E, N)
            D: Skip connection (E,)
            state: Recurrent state (B, E, N)
        Returns:
            Output sequence y (B, T, E), next state (B, E, N)
        """
        B, T, E = u.shape
        N = A.shape[-1] # State dim

        # Discretize A (A_bar = exp(delta * A))
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)) # (B, T, E, N)
        # Discretize B (B_bar = delta * B - simplified ZOH)
        delta_B = delta.unsqueeze(-1) * B # (B, T, E, N)

        ys = []
        current_state = state
        for t in range(T):
            # Recurrence: h_t = A_bar_t * h_{t-1} + B_bar_t * u_t
            current_state = delta_A[:, t] * current_state + delta_B[:, t] * u[:, t].unsqueeze(-1) # (B, E, N)
            # Output: y_t = C_t * h_t + D * u_t
            yt = torch.einsum('ben,ben->be', current_state, C[:, t]) # (B, E)
            ys.append(yt)

        y = torch.stack(ys, dim=1) # (B, T, E)
        y = y + u * D.unsqueeze(0).unsqueeze(0) # Add skip connection (B, T, E)

        return y, current_state # Return output sequence and final state

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """
        Forward pass for training or inference.
        `use_cache` is relevant for generation but simplified here.
        """
        B, T, D = x.shape
        E = self.expanded_dim
        N = self.ssm_state_dim

        # Initialize state if needed
        if not self.state_initialized or self.ssm_state.shape[0] != B or self.ssm_state.device != x.device or self.ssm_state.dtype != x.dtype:
            self._init_state(B, x.dtype, x.device)

        # --- Input Projections ---
        z_x_ssm = self.in_proj(x) # (B, T, 2*E)
        z, x_ssm = z_x_ssm.split([E, E], dim=-1) # z:(B,T,E), x_ssm:(B,T,E)

        # --- Compute Selective Parameters (dt, B, C) ---
        delta = self.dt_proj(x) # (B, T, E)
        B_params = self.B_proj(x).view(B, T, E, N) # (B, T, E, N)
        C_params = self.C_proj(x).view(B, T, E, N) # (B, T, E, N)

        # Calculate dt (time step) with activation and constraints
        delta = F.softplus(delta) # Ensure positivity
        delta = torch.clamp(delta, min=self.dt_min, max=self.dt_max) # Clamp to range

        # --- Get Fixed State Parameter A ---
        A = -torch.exp(self.A_log.float()) # Use negative exp for stability (E, N)

        # --- State Space Calculation ---
        # Input u to scan is element-wise product of x_ssm and gated z
        u = x_ssm * F.silu(z) # (B, T, E)

        # Perform selective scan
        y, next_state = self._selective_scan(u, delta, A, B_params, C_params, self.D, self.ssm_state)
        self.ssm_state = next_state.detach() # Update stored state for next step (if generating)

        # --- Output Projection ---
        output = self.out_proj(y) # (B, T, D)
        return output

    def reset_state(self):
         self.state_initialized = False
         self.ssm_state = torch.Tensor()
         # logger.debug("SSM state reset.")


# --- Mixture of Experts Layer ---

class SparseMoE(nn.Module):
    """Sparse Mixture of Experts layer routes tokens to a subset of experts."""
    def __init__(self, hidden_dim: int, num_experts: int = 8, top_k: int = 2,
                 capacity_factor: float = 1.25, noisy_gating: bool = True, router_bias: bool = False,
                 expert_dropout: float = 0.0): # Added dropout for experts
        super().__init__()
        assert hidden_dim > 0 and num_experts > 0 and top_k > 0
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts) # Cannot pick more experts than available
        self.capacity_factor = capacity_factor
        self.noisy_gating = noisy_gating

        # Router network (learns which experts to send tokens to)
        self.router = nn.Linear(hidden_dim, num_experts, bias=router_bias)
        # Optional noise for gating during training
        self.noise_layer = nn.Linear(hidden_dim, num_experts, bias=router_bias) if noisy_gating else None

        # Expert networks (using SwiGLU as the FFN, including dropout within SwiGLU)
        self.experts = nn.ModuleList([
            SwiGLU(hidden_dim, dropout_prob=expert_dropout) for _ in range(num_experts)
        ])

        # Loss for load balancing (auxiliary loss)
        self.load_balancing_loss_coeff = 0.01 # Example coefficient, tune as needed

    def compute_load_balancing_loss(self, router_probs, expert_indices):
        """Calculates load balancing loss based on router probabilities and expert assignments."""
        # router_probs: (num_tokens, num_experts) - probabilities from router softmax
        # expert_indices: (num_tokens, top_k) - indices of selected experts

        num_tokens, num_experts = router_probs.shape
        if num_tokens == 0: return 0.0

        # Calculate fraction of tokens dispatched to each expert (f_i)
        # One-hot encoding of expert assignments for each token's top-k choices
        expert_mask = F.one_hot(expert_indices, num_classes=num_experts) # (num_tokens, top_k, num_experts)
        # Sum over top_k dim to get a mask indicating if an expert was chosen for a token
        token_expert_chosen = expert_mask.sum(dim=1) # (num_tokens, num_experts)
        # Sum over tokens to get count per expert, divide by total tokens
        f_i = token_expert_chosen.sum(dim=0) / num_tokens # (num_experts,)

        # Calculate average router probability for tokens sent to each expert (P_i)
        # Sum router probs for tokens dispatched to each expert, divide by num tokens dispatched
        P_i = (router_probs * token_expert_chosen).sum(dim=0) / torch.clamp(token_expert_chosen.sum(dim=0), min=1e-6) # (num_experts,)

        # Load balancing loss: Coefficient * sum(f_i * P_i) * num_experts
        # Encourages f_i and P_i to be uniform across experts
        loss = self.load_balancing_loss_coeff * torch.sum(f_i * P_i) * num_experts
        return loss


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: # Return output and aux loss
        B, T, D = x.shape
        x_flat = x.reshape(-1, D) # Flatten: (B*T, D)
        num_tokens = x_flat.shape[0]
        if num_tokens == 0:
             # Handle empty input gracefully
             return x, torch.tensor(0.0, device=x.device, dtype=x.dtype)


        # --- Routing ---
        router_logits = self.router(x_flat) # (num_tokens, num_experts)

        # Add noise during training for better load balancing (optional)
        if self.noisy_gating and self.training and self.noise_layer is not None:
             # Calculate noise std dev based on noise_layer output
             noise_std = F.softplus(self.noise_layer(x_flat)) # Ensure positive std dev
             # Generate noise and add to logits
             noise = torch.randn_like(router_logits) * noise_std
             router_logits = router_logits + noise

        # Get probabilities for all experts via softmax
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32) # Use float32 for stability

        # Get top-k experts and their gating scores (probabilities)
        gating_scores, selected_experts = torch.topk(router_probs, self.top_k, dim=-1) # scores: (N, K), indices: (N, K)

        # Normalize scores among the top-k chosen experts
        gating_scores = gating_scores / torch.clamp(gating_scores.sum(dim=-1, keepdim=True), min=1e-6)
        gating_scores = gating_scores.type_as(x) # Cast back to original type

        # --- Load Balancing Loss ---
        # Compute loss before capacity is applied, using original probabilities
        aux_loss = self.compute_load_balancing_loss(router_probs, selected_experts)


        # --- Dispatch Tokens with Capacity ---
        # Create a flattened expert index for each token's choices (token_idx, k_choice) -> expert_idx
        flat_expert_indices = selected_experts.flatten() # (num_tokens * top_k)

        # Create a mask indicating which token goes to which expert (expanded)
        dispatch_mask = F.one_hot(selected_experts, num_classes=self.num_experts) # (N, K, E)

        # Calculate expert capacity: max tokens per expert
        # Capacity should be slightly larger than uniform assignment
        tokens_per_expert_ideal = num_tokens * self.top_k / self.num_experts
        capacity = int(self.capacity_factor * tokens_per_expert_ideal)
        capacity = max(capacity, 1) # Minimum capacity of 1

        # --- Assign tokens to experts respecting capacity (complex part) ---
        # This often involves sorting or complex indexing. Using a simpler scatter-based approach.
        # Create output tensor
        final_output_flat = torch.zeros_like(x_flat)

        # Combine gating scores with the dispatch mask
        # combine_weights: (N, E) - weight of token N for expert E (0 if not top-k)
        combine_weights = (gating_scores.unsqueeze(-1) * dispatch_mask).sum(dim=1)

        # Get load per expert based on assignments (ignoring capacity for now)
        expert_load = dispatch_mask.sum(dim=(0, 1)) # Sum over N and K -> (E,)

        # TODO: Implement capacity handling (e.g., dropping tokens per expert if load > capacity)
        # This is non-trivial. For now, proceeding without strict capacity enforcement, which
        # might lead to memory issues or load imbalance in practice.
        # A proper implementation often uses `torch.scatter_add` or custom kernels.

        # --- Process Experts ---
        # Process each expert independently (can be parallelized)
        expert_outputs = []
        for i in range(self.num_experts):
             # Select tokens assigned to this expert (indices where combine_weights[:, i] > 0)
             # This selection method is inefficient for large scale.
             token_indices_for_expert_i = torch.nonzero(combine_weights[:, i] > 0, as_tuple=True)[0]

             if len(token_indices_for_expert_i) == 0:
                  # No tokens for this expert, skip computation
                  continue

             expert_input_tokens = x_flat[token_indices_for_expert_i]

             # --- Apply Capacity (Simple Truncation - Inefficient/Approximate) ---
             if expert_input_tokens.shape[0] > capacity:
                 logger.warning(f"MoE Expert {i}: Load {expert_input_tokens.shape[0]} exceeds capacity {capacity}. Truncating.")
                 # Simply take the first `capacity` tokens (not ideal, should use scores)
                 expert_input_tokens = expert_input_tokens[:capacity]
                 # Adjust indices and weights accordingly
                 token_indices_for_expert_i = token_indices_for_expert_i[:capacity]


             # Compute expert output
             expert_output = self.experts[i](expert_input_tokens) # (num_tokens_for_expert, D)

             # Get the gating scores for these specific tokens and this expert
             expert_gating_scores = combine_weights[token_indices_for_expert_i, i].unsqueeze(1) # (num_tokens_for_expert, 1)

             # Weight the expert output by its gating score
             weighted_expert_output = expert_output * expert_gating_scores

             # Add the weighted output back to the final output tensor using scatter_add_
             # Need to expand indices to match dimensions
             indices_expanded = token_indices_for_expert_i.unsqueeze(1).expand(-1, D)
             final_output_flat.scatter_add_(0, indices_expanded, weighted_expert_output)


        # Reshape back to original shape
        final_output = final_output_flat.reshape(B, T, D)

        return final_output, aux_loss


# --- Advanced Reasoning Components ---

# Reasoning Tracker (Simple GRU State)
class ReasoningTracker(nn.Module):
    """Maintains a reasoning state using a GRU over token representations."""
    def __init__(self, hidden_dim: int, num_layers: int = 1, reasoning_steps: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Reasoning steps not used directly in GRU, but conceptually how many passes
        self.reasoning_steps = reasoning_steps
        self.state_tracker = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # Optional projection/confidence head
        self.confidence_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor, initial_state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input: hidden_states (B, T, C)
        B, T, C = hidden_states.size()

        # Initialize reasoning state if not provided
        if initial_state is None:
            # GRU expects initial state shape (num_layers, B, hidden_dim)
            h_0 = torch.zeros(self.state_tracker.num_layers, B, self.hidden_dim,
                              device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            h_0 = initial_state

        # Pass sequence through GRU
        # output_states: (B, T, C), final_reasoning_state: (num_layers, B, C)
        output_states, final_reasoning_state = self.state_tracker(hidden_states, h_0)

        # Use the final hidden state of the *last token* from the *last layer* for confidence
        final_token_state = final_reasoning_state[-1] # (B, C)
        confidence = torch.sigmoid(self.confidence_predictor(final_token_state)) # (B, 1)

        # Return the GRU output states (potentially refined representations),
        # the final hidden state of the GRU, and the confidence score.
        return output_states, final_reasoning_state, confidence

    def reset_state(self):
         # GRU state is implicitly reset on each forward pass if initial_state=None is used
         # No explicit reset needed unless state is manually carried over.
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
        # Ensure value head is on the same device as the model
        self.value_head = nn.Linear(model.config.n_embd, 1).to(next(model.parameters()).device)
        logger.info(f"Initialized ToT with beam_size={beam_size}, max_depth={max_depth}, branches={num_branches}")


    @torch.no_grad()
    def _evaluate_state(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
        """Evaluates the 'value' of a given thought state (sequence). Simplified version."""
        if input_ids.numel() == 0: return -float('inf') # Handle empty input
        try:
            # Use the model's forward pass to get hidden states (disable cache for evaluation)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            hidden_states = outputs.get("hidden_states") # Get hidden states if returned
            if hidden_states is None:
                 # Fallback: get hidden states from base transformer if wrapper doesn't return them
                 base_outputs = self.model.gpt_lm_head.transformer(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                 hidden_states = base_outputs

            if hidden_states is None or hidden_states.numel() == 0:
                 logger.warning("ToT Evaluation: Could not get hidden states.")
                 return -1.0 # Default low value

            # Use the representation of the last token
            last_token_hidden_state = hidden_states[:, -1, :] # (B, C)
            value = self.value_head(last_token_hidden_state).mean().item() # Average over batch if B > 1
            return value
        except Exception as e:
            logger.error(f"Error during ToT state evaluation: {e}")
            return -float('inf') # Indicate evaluation failure

    @torch.no_grad()
    def _expand_node(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, temperature: float = 0.7) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        """Generates candidate next steps (branches) from a node."""
        if input_ids.numel() == 0: return []
        try:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.get("logits")
            if logits is None:
                logger.error("ToT Expansion: Could not get logits from model.")
                return []

            # Get logits for the *next* token prediction (at the last position)
            next_token_logits = logits[:, -1, :] # (B, vocab_size)

            # Apply temperature scaling
            if temperature > 0:
                 next_token_logits = next_token_logits / temperature

            # Get probabilities (use float32 for stability)
            probs = F.softmax(next_token_logits.float(), dim=-1)

            # Sample multiple next tokens to create branches
            # Using multinomial sampling. Beam search usually uses top-k logits directly.
            # Ensure num_samples <= vocab_size
            num_samples = min(self.num_branches, probs.shape[-1])
            if num_samples <= 0: return []

            next_token_candidates = torch.multinomial(probs, num_samples=num_samples, replacement=True) # (B, num_branches)

            branches = []
            B = input_ids.shape[0]
            if B > 1: logger.warning("ToT expansion currently only supports B=1") # Simplify for now

            for i in range(num_samples):
                next_token = next_token_candidates[:, i:i+1] # (B, 1)
                # Append new token and update mask
                branch_ids = torch.cat([input_ids, next_token], dim=1)
                # Create attention mask for the new token (assuming it's not padding)
                new_token_mask = torch.ones_like(next_token)
                branch_mask = torch.cat([attention_mask, new_token_mask], dim=1)

                # Evaluate the value of the newly created state
                value = self._evaluate_state(branch_ids, branch_mask)
                branches.append((branch_ids, branch_mask, value))

            return branches
        except Exception as e:
            logger.error(f"Error during ToT node expansion: {e}")
            return []


    @torch.no_grad()
    def search(self, initial_prompt: str, generation_length: int = 50) -> str:
        """Performs ToT search (simplified beam search style) to generate a response."""
        self.model.eval()
        current_device = next(self.model.parameters()).device # Get model's device

        try:
            inputs = self.tokenizer(initial_prompt, return_tensors="pt", padding=False) # No padding needed for single prompt
            input_ids = inputs["input_ids"].to(current_device)
            attention_mask = inputs["attention_mask"].to(current_device)
        except Exception as e:
             logger.error(f"ToT Tokenization Error: {e}")
             return "[ToT Tokenization Error]"

        if input_ids.shape[1] == 0:
             logger.error("ToT Error: Empty prompt after tokenization.")
             return "[ToT Empty Prompt Error]"

        initial_value = self._evaluate_state(input_ids, attention_mask)

        # Beam search state: list of (ids, mask, accumulated_logprob_or_value) tuples
        # Using value here instead of logprob
        beam = [(input_ids, attention_mask, initial_value)]
        max_len = input_ids.shape[1] + generation_length

        for depth in range(generation_length): # Limit generation length
            if not beam: # Stop if beam is empty
                break

            all_candidates = []
            for node_ids, node_mask, node_value in beam:
                # Stop expanding this path if EOS is generated or max length reached
                if node_ids.shape[1] >= max_len or (self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in node_ids[0]):
                    all_candidates.append((node_ids, node_mask, node_value)) # Keep finished sequences
                    continue

                # Expand the current node
                branches = self._expand_node(node_ids, node_mask)
                for branch_ids, branch_mask, branch_value in branches:
                     # Simple cumulative value; more sophisticated combination possible (e.g., average)
                    all_candidates.append((branch_ids, branch_mask, node_value + branch_value))

            if not all_candidates: # Stop if no candidates generated
                 break

            # Prune the beam: Sort candidates by value (higher is better) and keep top `beam_size`
            all_candidates.sort(key=lambda x: x[2], reverse=True)
            beam = all_candidates[:self.beam_size]

            # Early stopping condition: if the top beam element is finished
            if beam and (beam[0][0].shape[1] >= max_len or (self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in beam[0][0][0])):
                  break

        # Select the best path from the final beam
        if not beam:
             logger.error("ToT Search Error: Beam became empty.")
             return "[ToT Search Error: Empty Beam]"

        best_ids, _, best_value = max(beam, key=lambda x: x[2])
        logger.info(f"ToT Search finished. Best path value: {best_value:.4f}, Length: {best_ids.shape[1]}")

        # Decode the result, removing the initial prompt part
        prompt_length = len(self.tokenizer.encode(initial_prompt)) # Re-encode to get exact length
        result_ids = best_ids[0, prompt_length:]
        return self.tokenizer.decode(result_ids, skip_special_tokens=True)


# Algorithmic Reasoner (Conceptual Neural Register Machine Placeholder)
class AlgorithmicReasoner(nn.Module):
    """
    A conceptual module for performing step-by-step algorithmic reasoning
    using neural registers and learned operations. Placeholder implementation.
    """
    def __init__(self, hidden_dim: int, num_registers: int = 4, max_steps: int = 10):
        super().__init__()
        logger.warning("AlgorithmicReasoner is a conceptual placeholder.")
        self.hidden_dim = hidden_dim
        self.num_registers = num_registers
        self.max_steps = max_steps

        # Learnable initial state for registers
        self.register_init = nn.Parameter(torch.randn(1, num_registers, hidden_dim))

        # Control network: decides operation and registers based on input/current state
        # Input: current hidden state + flattened registers
        controller_input_dim = hidden_dim + num_registers * hidden_dim
        # Output: Signals for Op1, Op2, Gate, RegA_Select, RegB_Select, RegDest_Select, HaltProb
        control_output_dim = hidden_dim * 3 + num_registers * 3 + 1 # Example dimensionality
        self.controller = nn.Linear(controller_input_dim, control_output_dim)

        # Simple learned operations (can be made more complex: e.g., attention over registers)
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

            # Prepare controller input (current instruction + current register state)
            controller_input = torch.cat([program_input, registers.view(B, -1)], dim=1) # (B, C + num_reg*C)
            control_signals = self.controller(controller_input) # (B, control_output_dim)

            # --- Parse control signals ---
            op_signal_dim = self.hidden_dim * 3 # Example: 3 signals for operations
            reg_signal_dim = self.num_registers * 3 # Example: Select Reg A, Reg B, Reg Dest
            halt_signal_dim = 1
            op_signals, reg_signals, halt_signal = control_signals.split(
                [op_signal_dim, reg_signal_dim, halt_signal_dim], dim=1
            )

            # --- Register Selection (using argmax for simplicity) ---
            reg_a_scores, reg_b_scores, reg_dest_scores = reg_signals.view(B, 3, self.num_registers).unbind(1) # (B, num_reg) each
            reg_a_idx = torch.argmax(reg_a_scores, dim=1) # (B,)
            reg_b_idx = torch.argmax(reg_b_scores, dim=1) # (B,)
            reg_dest_idx = torch.argmax(reg_dest_scores, dim=1) # (B,)


            # Get register values using batch indexing
            batch_indices = torch.arange(B, device=registers.device)
            try:
                 reg_a = registers[batch_indices, reg_a_idx] # (B, C)
                 reg_b = registers[batch_indices, reg_b_idx] # (B, C)
            except IndexError as e:
                 logger.error(f"Algorithmic Reasoner IndexError: B={B}, idx_a={reg_a_idx}, idx_b={reg_b_idx}, reg_shape={registers.shape}. Error: {e}")
                 # Handle error, maybe return current registers
                 return registers, all_register_states


            # --- Operation Execution (Example: simple gated update) ---
            gate1, gate2, update_val = op_signals.chunk(3, dim=1) # (B, C) each
            gate1, gate2 = torch.sigmoid(gate1), torch.sigmoid(gate2) # Gates (0 to 1)
            update_val = torch.tanh(update_val) # Update value (-1 to 1)

            # Example operation: result = g1*reg_a + g2*reg_b + update*transform(reg_a)
            result = gate1 * reg_a + gate2 * reg_b + update_val * self.op_transform(reg_a) # (B, C)

            # --- Update Destination Register ---
            # Create a copy to modify, then assign back
            new_registers = registers.clone()
            try:
                 new_registers[batch_indices, reg_dest_idx] = result
                 registers = new_registers
            except IndexError as e:
                  logger.error(f"Algorithmic Reasoner IndexError during update: B={B}, idx_dest={reg_dest_idx}, reg_shape={registers.shape}. Error: {e}")
                  return registers, all_register_states


            all_register_states.append(registers.clone())

            # --- Halt Condition ---
            halt_prob = torch.sigmoid(halt_signal) # (B, 1)
            halt_decision = (halt_prob > 0.5) & (~halted) # Decide to halt only if not already halted
            halted = halted | halt_decision # Update overall halted status

        # Return the final state of the registers and the history
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
        logger.debug(f"Calculator attempting to evaluate: '{expression}'")
        try:
            # Basic sanitization
            if not isinstance(expression, str) or not expression.strip():
                 return {"error": "Invalid input: Expression must be a non-empty string."}
            # Limit length to prevent abuse
            if len(expression) > 200:
                 logger.warning(f"Calculator input too long: {len(expression)} chars.")
                 return {"error": "Expression too long (max 200 chars)."}
            # Avoid potentially harmful keywords (very basic check)
            forbidden_keywords = ['import', 'os.', 'sys.', 'eval', 'exec', '__']
            if any(kw in expression for kw in forbidden_keywords):
                 logger.warning(f"Calculator blocked potentially unsafe expression: '{expression}'")
                 return {"error": "Expression contains forbidden keywords."}


            # Attempt to parse and evaluate using sympify
            # Using a limited local dict can enhance security slightly, but sympify itself is powerful.
            # local_dict = {"sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "pi": sp.pi}
            # Use strict=True to prevent automatic symbol creation from typos?
            expr = sp.sympify(expression, strict=False) # strict=False is more lenient

            # Evaluate numerically if possible, otherwise simplify
            evaluated_result = None
            try:
                # Attempt numerical evaluation with N
                evaluated_result = sp.N(expr)
                # Check if the result is reasonably numeric (not symbolic/complex infinity)
                if isinstance(evaluated_result, sp.Number) and evaluated_result.is_real and evaluated_result.is_finite:
                    # Format nicely, avoid excessive precision
                    result = f"{evaluated_result:.4f}" if abs(evaluated_result) < 1e6 else f"{evaluated_result:.4e}"
                else:
                    # If not purely numeric or too large/small, simplify symbolically
                    simplified_result = sp.simplify(expr)
                    result = str(simplified_result)
            except (TypeError, ValueError, NotImplementedError) as e_eval:
                logger.warning(f"Calculator evalf failed for '{expression}': {e_eval}. Falling back to simplify.")
                # Fallback to symbolic simplification if evalf fails
                simplified_result = sp.simplify(expr)
                result = str(simplified_result)

            # Limit output length
            if len(result) > 500:
                 logger.warning(f"Calculator result too long: {len(result)} chars.")
                 return {"error": "Result too long (max 500 chars)."}

            logger.debug(f"Calculator result for '{expression}': '{result}'")
            return {"result": result}

        except (sp.SympifyError, TypeError, SyntaxError, ValueError) as e:
            logger.warning(f"Calculator sympify/evaluation error: {type(e).__name__} for '{expression}'")
            return {"error": f"Calculation error: {type(e).__name__}"}
        except Exception as e:
            logger.error(f"Unexpected calculator error: {e} for expression '{expression}'", exc_info=True)
            return {"error": "An unexpected error occurred during calculation."}


# --- Transformer Block and Model Architecture ---

# Transformer Block
class TransformerBlock(nn.Module):
    """A single block of the Transformer model."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.attn_norm = RMSNorm(config.n_embd)

        # --- Choose Attention Mechanism ---
        if config.use_gqa:
            self.attn = GroupedQueryAttention(config)
        else:
            self.attn = MultiHeadSelfAttention(config) # Default MHA

        # --- Optional RWKV Block ---
        self.use_rwkv = config.use_rwkv
        if self.use_rwkv:
             self.rwkv_norm = RMSNorm(config.n_embd) # Normalize input to RWKV
             self.rwkv_attn = RWKVAttention(config)

        # --- Optional SSM Block ---
        self.use_ssm = config.use_ssm
        if self.use_ssm:
             self.ssm_norm = RMSNorm(config.n_embd) # Normalize input to SSM
             # Example state dim, make configurable if needed
             self.ssm = SelectiveSSM(config.n_embd, ssm_state_dim=max(16, config.n_embd // 8))
             self.ssm_dropout = nn.Dropout(config.dropout_prob) # Separate dropout for SSM path


        # --- FFN Block (choose between standard SwiGLU and MoE) ---
        self.ffn_norm = RMSNorm(config.n_embd)
        self.use_moe = config.use_moe
        if self.use_moe:
            self.mlp = SparseMoE(config.n_embd, num_experts=config.num_experts, top_k=config.top_k_experts)
            # MoE layer might have its own internal dropout
        else:
            # Pass dropout prob to SwiGLU
            self.mlp = SwiGLU(config.n_embd, dropout_prob=config.dropout_prob)

        # Store auxiliary loss from MoE if used
        self.aux_loss = None


    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
                alibi_bias: torch.Tensor | None = None, use_cache: bool = False,
                position_ids: torch.LongTensor | None = None) -> torch.Tensor:

        # --- Attention Path ---
        residual = x
        h = self.attn_norm(x)
        # Attention module's forward method applies output projection and residual dropout
        attn_output = self.attn(h, attention_mask=attention_mask, alibi_bias=alibi_bias,
                                use_cache=use_cache, position_ids=position_ids)
        x = residual + attn_output # First residual connection

        # --- Optional RWKV Path ---
        if self.use_rwkv:
            residual = x
            h = self.rwkv_norm(x)
            rwkv_output = self.rwkv_attn(h)
            # RWKVAttention includes output projection, add residual dropout here
            x = residual + self.attn.resid_dropout(rwkv_output) # Reusing attn's dropout

        # --- Optional SSM Path ---
        if self.use_ssm:
             residual = x
             h = self.ssm_norm(x)
             ssm_output = self.ssm(h, use_cache=use_cache) # Pass cache flag if SSM uses it
             # SSM includes output projection, add residual dropout
             x = residual + self.ssm_dropout(ssm_output)

        # --- FFN Path ---
        residual = x
        h = self.ffn_norm(x)
        if self.use_moe:
             ffn_output, aux_loss = self.mlp(h)
             self.aux_loss = aux_loss # Store aux loss for later retrieval
        else:
             ffn_output = self.mlp(h)
        # FFN module (SwiGLU/MoE) applies dropout internally, add residual here
        x = residual + ffn_output # Second residual connection

        return x

    # --- Gradient Checkpointing Wrapper ---
    # This method is designed to be called when gradient checkpointing is active.
    def _forward_checkpointed(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                                   alibi_bias: torch.Tensor | None, use_cache: bool,
                                   position_ids: torch.LongTensor | None) -> torch.Tensor:

        # Checkpointing requires functions that take inputs and return outputs.
        # We wrap each main computation step. use_reentrant=False is generally recommended.

        # --- Attention Path ---
        def run_attn(current_x, norm, attn_layer, mask, bias, cache, pos_ids):
            h = norm(current_x)
            # Note: Attention layer applies its own proj+dropout
            return attn_layer(h, attention_mask=mask, alibi_bias=bias, use_cache=cache, position_ids=pos_ids)

        residual = x
        attn_output = checkpoint(run_attn, x, self.attn_norm, self.attn, attention_mask, alibi_bias, use_cache, position_ids, use_reentrant=False)
        x = residual + attn_output


        # --- Optional RWKV Path ---
        if self.use_rwkv:
            def run_rwkv(current_x, norm, rwkv_layer):
                h = norm(current_x)
                # RWKV layer applies its own proj
                return rwkv_layer(h)

            residual = x
            rwkv_output = checkpoint(run_rwkv, x, self.rwkv_norm, self.rwkv_attn, use_reentrant=False)
            # Apply dropout *after* checkpoint boundary
            x = residual + self.attn.resid_dropout(rwkv_output)


        # --- Optional SSM Path ---
        if self.use_ssm:
             def run_ssm(current_x, norm, ssm_layer, cache_flag):
                  h = norm(current_x)
                  # SSM layer applies its own proj
                  return ssm_layer(h, use_cache=cache_flag)

             residual = x
             ssm_output = checkpoint(run_ssm, x, self.ssm_norm, self.ssm, use_cache, use_reentrant=False)
             # Apply dropout *after* checkpoint boundary
             x = residual + self.ssm_dropout(ssm_output)


        # --- FFN Path ---
        def run_ffn(current_x, norm, ffn_layer):
            h = norm(current_x)
            # FFN layer applies its own dropout
            if isinstance(ffn_layer, SparseMoE):
                 output, aux_loss = ffn_layer(h)
                 # How to handle aux loss with checkpointing? Difficult.
                 # Usually compute aux loss outside checkpoint or ignore during checkpoint pass.
                 # Storing it on the module instance within the checkpointed function might work
                 # if the instance reference is preserved.
                 self.aux_loss = aux_loss
                 return output
            else:
                 return ffn_layer(h)

        residual = x
        ffn_output = checkpoint(run_ffn, x, self.ffn_norm, self.mlp, use_reentrant=False)
        x = residual + ffn_output

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
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"GPTModel Parameter Count: {param_count:,}")
        print(f"GPTModel Parameter Count: {param_count:,}")


    def _init_weights(self, module: nn.Module):
        """Initialize weights using scheme similar to LLaMA."""
        if isinstance(module, nn.Linear):
            # Normal init for most linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            # Initialize norm weights to 1, bias to 0
            if hasattr(module, 'weight'):
                torch.nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # Apply special scaled init to specific layers if needed (e.g., output projections)
        # From GPT-2 paper: scale weights of residual projections
        for name, p in module.named_parameters():
             if name == 'out_proj.weight' or name == 'w2.weight': # Output projection in Attn/FFN
                  if hasattr(self.config, 'n_layer'): # Ensure config is available
                       torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                use_cache: bool = False, position_ids: torch.LongTensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass of the core GPT model """
        B, T = input_ids.shape

        # Validate input length against config
        if T > self.config.max_seq_len:
             logger.warning(f"Input sequence length ({T}) exceeds model max ({self.config.max_seq_len}). Truncating.")
             input_ids = input_ids[:, :self.config.max_seq_len]
             if position_ids is not None:
                  position_ids = position_ids[:, :self.config.max_seq_len]
             if attention_mask is not None and attention_mask.dim() > 1 and attention_mask.shape[1] > self.config.max_seq_len:
                 attention_mask = attention_mask[:, :self.config.max_seq_len]
             elif attention_mask is not None and attention_mask.dim() > 2 and attention_mask.shape[-1] > self.config.max_seq_len:
                 attention_mask = attention_mask[..., :self.config.max_seq_len] # Handle 4D mask

             T = self.config.max_seq_len # Update T after truncation


        # 1. Token Embeddings
        try:
             token_embeddings = self.wte(input_ids) # (B, T, C)
        except IndexError as e:
             logger.error(f"Embedding IndexError: input_ids range error? Max ID: {input_ids.max()}, Min ID: {input_ids.min()}, Vocab size: {self.config.vocab_size}. Error: {e}")
             raise
        x = self.drop(token_embeddings)

        # 2. Build ALiBi bias if needed (once per forward pass)
        alibi_bias = None
        if self.config.alibi:
            # Build bias appropriate for the current sequence length T
            alibi_bias = build_alibi_tensor(self.config.n_head, T, device=x.device, dtype=x.dtype)

        # 3. Process through Transformer Blocks
        all_aux_losses = []
        for i, block in enumerate(self.blocks):
             block.aux_loss = None # Reset aux loss before block forward
             # Checkpointing logic
             if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                  # Use the checkpointed forward method of the block
                  x = block._forward_checkpointed(x, attention_mask, alibi_bias, use_cache, position_ids)
             else:
                  # Normal forward pass
                  x = block(x, attention_mask=attention_mask, alibi_bias=alibi_bias,
                            use_cache=use_cache, position_ids=position_ids)

             # Collect auxiliary loss if present (e.g., from MoE)
             if hasattr(block, 'aux_loss') and block.aux_loss is not None:
                  all_aux_losses.append(block.aux_loss)


        # 4. Final Normalization
        x = self.norm_f(x)

        # Combine auxiliary losses if any
        total_aux_loss = None
        if all_aux_losses:
             total_aux_loss = torch.stack(all_aux_losses).mean() # Average loss across blocks

        # Return hidden states and potentially auxiliary loss
        if total_aux_loss is not None:
             return x, total_aux_loss
        else:
             return x # Return only hidden states


    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
             max_new_tokens: int = 50, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
             eos_token_id: int | list[int] | None = None, do_sample: bool = True) -> torch.Tensor:
        """Generates sequences autoregressively using KV caching."""
        self.eval() # Set model to evaluation mode
        B, T_prompt = input_ids.shape
        current_device = input_ids.device

        # Ensure eos_token_id is a list for easier checking
        if eos_token_id is not None and not isinstance(eos_token_id, list):
            eos_token_id_list = [eos_token_id]
        else:
            eos_token_id_list = eos_token_id


        # Reset KV caches and any stateful modules (like RWKV, SSM)
        for block in self.blocks:
            if hasattr(block.attn, '_init_kv_cache'): # Use the init method which handles reset
                block.attn._init_kv_cache(batch_size=B, dtype=self.wte.weight.dtype, device=current_device)
            if hasattr(block, 'rwkv_attn') and hasattr(block.rwkv_attn, 'reset_state'):
                block.rwkv_attn.reset_state()
            if hasattr(block, 'ssm') and hasattr(block.ssm, 'reset_state'):
                 block.ssm.reset_state()


        generated_ids = input_ids.clone()

        # Create or clone attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=current_device)
        else:
            # Ensure mask is correct shape (B, T) for simple extension
             if attention_mask.dim() != 2:
                 logger.warning(f"Generation received attention_mask with dim != 2 ({attention_mask.dim()}). Attempting to use last dimension.")
                 # Try to adapt mask, assuming last dim is sequence length
                 if attention_mask.dim() == 4: # (B, 1, T_q, T_k) -> (B, T_k)
                     attention_mask = attention_mask[:, 0, -1, :]
                 elif attention_mask.dim() == 3: # (B, T_q, T_k)? -> (B, T_k)
                     attention_mask = attention_mask[:, -1, :]
                 else:
                      logger.error("Cannot adapt attention mask shape for generation.")
                      attention_mask = torch.ones_like(input_ids, device=current_device) # Fallback


             attention_mask = attention_mask.clone()


        with torch.no_grad():
            # --- Process Prompt Phase (Fill KV Cache) ---
            # Calculate position_ids for the prompt
            prompt_position_ids = torch.arange(0, T_prompt, device=current_device).unsqueeze(0) #.expand(B, -1) Is expand needed? test.

            # Run prompt through the model to fill the KV cache
            # We only need the side effect of filling the cache, output is discarded
            _ = self(input_ids, attention_mask=attention_mask, use_cache=True, position_ids=prompt_position_ids)
            # logger.debug("KV cache filled with prompt.")


            # --- Generation Phase ---
            for step in range(max_new_tokens):
                T_current = generated_ids.size(1)

                # Prepare input for the next token prediction: last token ID
                next_token_input_ids = generated_ids[:, -1:] # (B, 1)

                # Calculate position_id for the *current* token being generated
                # Position is the index it will occupy: T_current
                next_position_ids = torch.tensor([[T_current]], device=current_device, dtype=torch.long) #.expand(B,-1) ?

                # Update attention mask for the generation step
                # The mask should cover all tokens up to the current one
                # During generation, the attention mask for the new token is implicitly handled
                # by the causal nature + KV cache. We don't need a complex mask.
                # A simple mask indicating valid *cache* positions might be useful if padding was in prompt.
                # For simplicity, assuming no padding in prompt or handled by cache mechanism.
                # Let's pass the extended mask.
                current_attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_input_ids)], dim=1)


                try:
                    # Forward pass for the single next token, using KV cache
                    # Pass only the new token and its position
                    outputs = self(next_token_input_ids,
                                   attention_mask=None, # Not needed when T_q=1 with KV cache
                                   use_cache=True,
                                   position_ids=next_position_ids) # Pass correct position

                    # Handle potential tuple output (hidden_states, aux_loss)
                    if isinstance(outputs, tuple):
                         hidden_states = outputs[0]
                    else:
                         hidden_states = outputs

                    # Get logits for the last generated token
                    # Project hidden state to vocabulary using the embedding matrix (tied weights)
                    logits = F.linear(hidden_states[:, -1, :], self.wte.weight) # (B, vocab_size)

                except Exception as e:
                     logger.error(f"Error during generation forward pass (step {step}): {e}", exc_info=True)
                     # If we've generated at least some tokens, return what we have
                     if step > 0:
                          logger.warning("Generation stopped early due to error. Returning partial result.")
                          break
                     else:
                          raise # Re-raise if error on first step

                # --- Apply Sampling Strategies ---
                if do_sample and temperature > 0:
                    # Temperature scaling
                    logits = logits / max(temperature, 1e-8) # Avoid division by zero

                    # Top-K filtering
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # Ensure k <= vocab_size
                        logits[logits < v[:, [-1]]] = -float('Inf') # Mask out lower probability tokens

                    # Top-P (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits.float(), dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift right to keep the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        # Scatter back to original order -> create mask
                        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = -float('Inf') # Mask out low probability mass tokens

                    # Sample from the filtered distribution
                    probs = F.softmax(logits.float(), dim=-1) # Use float32 for stability
                    next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True) # (B, 1)


                # Append the generated token
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                # Update the attention mask for the next iteration (if needed, though often implicit with cache)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)


                # Check for EOS token
                if eos_token_id_list is not None:
                     # Check if *all* sequences in the batch ended
                     if (torch.isin(next_token, torch.tensor(eos_token_id_list, device=current_device))).all():
                          # logger.debug(f"EOS token detected at step {step}. Stopping generation.")
                          break

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
            dict: Contains 'loss', 'logits', 'hidden_states', 'aux_loss' (if applicable).
        """
        # 1. Get hidden states from the base transformer model
        outputs = self.transformer(input_ids, attention_mask=attention_mask,
                                   use_cache=use_cache, position_ids=position_ids)

        # Handle potential tuple output (hidden_states, aux_loss)
        aux_loss = None
        if isinstance(outputs, tuple):
             hidden_states = outputs[0]
             aux_loss = outputs[1]
        else:
             hidden_states = outputs

        # 2. Project hidden states to vocabulary logits
        # Uses the same weight matrix as the token embeddings (weight tying)
        try:
             logits = F.linear(hidden_states, self.transformer.wte.weight) # (B, T, vocab_size)
        except Exception as e:
             logger.error(f"Error during LM head projection: {e}. Hidden state shape: {hidden_states.shape}, Weight shape: {self.transformer.wte.weight.shape}")
             raise

        # 3. Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            # Logits: Ignore last token's prediction -> (B, T-1, V)
            shift_logits = logits[:, :-1, :].contiguous()
            # Labels: Ignore first token -> (B, T-1)
            shift_labels = labels[:, 1:].contiguous()

            # Flatten and compute cross-entropy loss
            # Ignore index -100 (common practice for padding in labels)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Calculate primary CE loss
            ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Add auxiliary loss (e.g., MoE load balancing) if present
            if aux_loss is not None:
                 loss = ce_loss + aux_loss
            else:
                 loss = ce_loss

        return {
             "loss": loss,
             "logits": logits,
             "hidden_states": hidden_states,
             "aux_loss": aux_loss # Return aux_loss separately as well
        }


# --- Multihead Latent Attention (MLA) ---

class MLA(nn.Module):
    """
    Multihead Latent Attention (MLA) uses learnable latent tokens to process
    sequence information, potentially enhancing reasoning or summarization.
    Inspired by Perceiver IO / Set Transformer concepts.
    """
    def __init__(self, n_embd: int, n_latent: int, n_head: int, dropout_prob: float = 0.1, thinking_steps: int = 1):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dim must be divisible by number of heads"
        self.n_latent = n_latent
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.thinking_steps = thinking_steps # Iterative refinement steps
        if thinking_steps < 1:
             logger.warning("MLA thinking_steps < 1, setting to 1.")
             self.thinking_steps = 1


        # Learnable latent query tokens
        self.latent_queries = nn.Parameter(torch.randn(1, n_latent, n_embd)) # (1, N_latent, C)

        # --- Layers for Cross-Attention (Latents attend to Input) ---
        self.cross_attn_norm_latent = RMSNorm(n_embd)
        self.cross_attn_norm_input = RMSNorm(n_embd)
        self.q_proj_latent = nn.Linear(n_embd, n_embd, bias=False) # Query from latents
        self.k_proj_input = nn.Linear(n_embd, n_embd, bias=False) # Key from input sequence
        self.v_proj_input = nn.Linear(n_embd, n_embd, bias=False) # Value from input sequence
        self.cross_attn_out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.cross_attn_dropout = nn.Dropout(dropout_prob)

        # --- Layers for Self-Attention (Latents attend to Latents - for refinement) ---
        # Using separate layers for clarity, could potentially share
        self.self_attn_norm = RMSNorm(n_embd)
        self.q_proj_self = nn.Linear(n_embd, n_embd, bias=False) # Q from latents
        self.k_proj_self = nn.Linear(n_embd, n_embd, bias=False) # K from latents
        self.v_proj_self = nn.Linear(n_embd, n_embd, bias=False) # V from latents
        self.self_attn_out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.self_attn_dropout = nn.Dropout(dropout_prob)

        # FFN for latents
        self.ffn_norm = RMSNorm(n_embd)
        self.ffn = SwiGLU(n_embd, dropout_prob=dropout_prob) # Use SwiGLU for latent FFN

        # Final projection to add back to original sequence (optional)
        # self.final_proj = nn.Linear(n_embd, n_embd, bias=False)


    def _attention(self, q_norm, kv_norm, q_proj, k_proj, v_proj, out_proj, dropout_layer):
         """Generic multi-head attention calculation."""
         B, T_q, C = q_norm.shape
         T_kv = kv_norm.shape[1]

         q = q_proj(q_norm)
         k = k_proj(kv_norm)
         v = v_proj(kv_norm)

         # Reshape for multi-head: (B, T, C) -> (B, H, T, D_h)
         q = q.view(B, T_q, self.n_head, self.head_dim).transpose(1, 2)
         k = k.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)
         v = v.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)

         # Compute attention scores (use float32 for stability)
         attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
         attn_weights = F.softmax(attn_scores, dim=-1).type_as(q) # Cast back
         attn_weights = dropout_layer(attn_weights)

         # Compute output
         attn_output = torch.matmul(attn_weights, v) # (B, H, T_q, D_h)

         # Reshape back and project
         attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, C)
         attn_output = out_proj(attn_output)
         return attn_output


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input sequence using latent attention and refinement.

        Args:
            x: Input tensor (B, T, C) from the underlying transformer.

        Returns:
            Tensor: Enhanced input tensor (B, T, C). Output depends on design choice.
                    Here, we return modified latents, assuming they are used downstream
                    or added back differently. Let's modify to add back to `x`.
        """
        B, T, C = x.size()

        # Expand latent queries for the batch
        latents = self.latent_queries.expand(B, -1, -1) # (B, N_latent, C)

        # Input sequence (normalized once)
        x_norm = self.cross_attn_norm_input(x)

        # Iterative refinement loop
        for _ in range(self.thinking_steps):
            # --- Cross-Attention: Latents attend to Input ---
            latents_norm = self.cross_attn_norm_latent(latents)
            cross_attn_output = self._attention(
                 q_norm=latents_norm, kv_norm=x_norm, # Use pre-normed input K/V
                 q_proj=self.q_proj_latent, k_proj=self.k_proj_input, v_proj=self.v_proj_input,
                 out_proj=self.cross_attn_out_proj, dropout_layer=self.cross_attn_dropout
            )
            # Residual connection for latents
            latents = latents + self.cross_attn_dropout(cross_attn_output) # Apply dropout on attn output


            # --- Self-Attention: Latents attend to Latents ---
            latents_norm = self.self_attn_norm(latents)
            self_attn_output = self._attention(
                 q_norm=latents_norm, kv_norm=latents_norm, # Self attention uses latents for K/V too
                 q_proj=self.q_proj_self, k_proj=self.k_proj_self, v_proj=self.v_proj_self,
                 out_proj=self.self_attn_out_proj, dropout_layer=self.self_attn_dropout
            )
            # Residual connection for latents
            latents = latents + self.self_attn_dropout(self_attn_output)

            # --- FFN for Latents ---
            residual_latents = latents
            latents_norm = self.ffn_norm(latents)
            ffn_output = self.ffn(latents_norm) # Dropout is inside SwiGLU
            latents = residual_latents + ffn_output


        # --- Combine latents and add back to original sequence ---
        # Option 1: Mean pool latents and add broadcasted projection
        aggregated_latent = latents.mean(dim=1) # (B, C)
        # enhancement = self.final_proj(aggregated_latent) # Project aggregated info
        enhancement = aggregated_latent # Or just use the mean directly if final_proj is omitted
        enhanced_x = x + enhancement.unsqueeze(1) # Add to each token (broadcasts over T)

        # Option 2: Return the processed latents (if used differently later)
        # return latents

        # Option 3: Return the enhanced sequence (most common)
        return enhanced_x


# --- Optimized Model Wrapper (integrating enhancements and MPS specifics) ---

class MPSOptimizedEnhancedGPTLMHeadModel(nn.Module):
    """
    Optimized GPT model wrapper integrating enhancements with MPS-specific considerations.
    This class replaces the separate EnhancedGPTLMHeadModel and directly uses GPTLMHeadModel.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # --- Core Language Model ---
        # Contains GPTModel (Embeddings, Blocks, Norm) + LM Head Projection
        self.gpt_lm_head = GPTLMHeadModel(config)

        # --- Optional Enhancement Modules ---
        self.use_mla = config.use_mla
        if self.use_mla:
            self.mla = MLA(config.n_embd, config.mla_n_latent, config.n_head,
                           dropout_prob=config.dropout_prob, thinking_steps=1) # Example steps

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

        # --- Tree of Thought (Conceptual/Optional) ---
        self.use_tree_of_thought = config.use_tree_of_thought
        self.tree_of_thought: TreeOfThought | None = None # Initialize later if needed

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None, use_cache: bool = False,
            position_ids: torch.LongTensor | None = None) -> dict[str, torch.Tensor | None]:

        # MPS Optimization: Optional cache clearing (can be aggressive)
        # if torch.backends.mps.is_available() and not use_cache and torch.is_grad_enabled():
        #     torch.mps.empty_cache()

        # --- Base Transformer Pass ---
        # Get hidden states and potentially auxiliary loss from the base model
        transformer_outputs = self.gpt_lm_head.transformer(
            input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            position_ids=position_ids
        )

        base_aux_loss = None
        if isinstance(transformer_outputs, tuple):
             hidden_states = transformer_outputs[0]
             base_aux_loss = transformer_outputs[1] # e.g., from MoE in base blocks
        else:
             hidden_states = transformer_outputs


        # --- Apply Enhancements (only during training/eval, not generation with cache) ---
        reasoning_output = None
        algo_output = None

        if not use_cache: # Apply enhancements only when not using KV cache
             if self.use_mla:
                  hidden_states = self.mla(hidden_states)

             if self.use_reasoning_tracker:
                  # Reasoning tracker returns refined states, final GRU state, confidence
                  refined_states, _, confidence = self.reasoning_tracker(hidden_states)
                  hidden_states = refined_states # Use refined states for subsequent steps
                  reasoning_output = {"reasoning_confidence": confidence}

             if self.use_algorithmic_reasoner:
                  # Algorithmic reasoner returns final registers and history
                  final_registers, _ = self.algorithmic_reasoner(hidden_states)
                  algo_output = {"algorithmic_registers": final_registers}
                  # Note: The registers aren't directly integrated back into hidden_states here.
                  # This might require a different architecture if they should influence logits.


        # --- Final LM Head Projection ---
        # Project potentially enhanced hidden states to vocabulary logits
        logits = F.linear(hidden_states, self.gpt_lm_head.transformer.wte.weight)

        # --- Calculate Loss ---
        loss = None
        if labels is not None:
             # Shift logits and labels for next token prediction
             shift_logits = logits[:, :-1, :].contiguous()
             shift_labels = labels[:, 1:].contiguous()

             # Compute CE loss, ignoring padding (-100)
             loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
             try:
                  # Ensure shapes match before loss calculation
                  vocab_size = shift_logits.size(-1)
                  ce_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

                  # Add base auxiliary loss (e.g., MoE loss from transformer blocks)
                  if base_aux_loss is not None:
                       loss = ce_loss + base_aux_loss.type_as(ce_loss) # Ensure same dtype
                  else:
                       loss = ce_loss

             except ValueError as e:
                  logger.error(f"Error in loss calculation: {e}. Logits shape: {shift_logits.shape}, Labels shape: {shift_labels.shape}")
                  # Provide a fallback loss (zero) to prevent training crash, but log error
                  loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
             except IndexError as e:
                 logger.error(f"IndexError in loss calculation: {e}. Logits shape: {shift_logits.shape}, Labels shape: {shift_labels.shape}, Labels Max: {shift_labels.max()}, Min: {shift_labels.min()}")
                 loss = torch.tensor(0.0, device=logits.device, requires_grad=True)


        # --- Combine outputs ---
        output_dict = {"loss": loss, "logits": logits, "hidden_states": hidden_states, "aux_loss": base_aux_loss}
        if reasoning_output:
            output_dict.update(reasoning_output)
        if algo_output:
            output_dict.update(algo_output)

        return output_dict


    # --- Generation Method ---
    # Delegate generation to the underlying GPTModel's generate method,
    # which correctly handles KV caching and sampling logic.
    def generate(self, *args, **kwargs):
        # logger.debug("MPSOptimizedEnhancedGPTLMHeadModel delegating generate call...")
        # Generation should bypass MLA, ReasoningTracker etc. by using use_cache=True internally
        return self.gpt_lm_head.transformer.generate(*args, **kwargs)


    # --- Methods for Tool Use and Advanced Reasoning ---

    def run_calculator(self, expression: str) -> str:
        """Interface to the calculator tool."""
        if not self.use_calculator or not hasattr(self, 'calculator'):
            return "[Calculator tool is disabled or not initialized]"
        result_dict = self.calculator.calculate(expression)
        # Return result or error message clearly marked
        return result_dict.get("result", f"[Calculator Error: {result_dict.get('error', 'Unknown')}]")


    def run_tot_search(self, prompt: str, tokenizer: PreTrainedTokenizerBase, generation_length: int = 50) -> str:
        """Interface to run Tree of Thought search (if enabled and implemented)."""
        if not self.use_tree_of_thought:
             return "[Tree of Thought is disabled in config]"

        # Initialize ToT component on first call if needed
        if self.tree_of_thought is None:
             logger.info("Initializing Tree of Thought component...")
             try:
                  self.tree_of_thought = TreeOfThought(self, tokenizer) # Pass self (model) and tokenizer
             except Exception as e:
                  logger.error(f"Failed to initialize TreeOfThought: {e}", exc_info=True)
                  return f"[Tree of Thought Initialization Error: {e}]"


        if not isinstance(self.tree_of_thought, TreeOfThought):
             return "[Tree of Thought component not available or not initialized correctly]"

        logger.info(f"Starting Tree of Thought search for prompt: '{prompt[:50]}...'")
        try:
            return self.tree_of_thought.search(prompt, generation_length=generation_length)
        except Exception as e:
             logger.error(f"Error during ToT search: {e}", exc_info=True)
             return f"[Tree of Thought Search Error: {e}]"

    def reset_reasoning_states(self):
        """Resets states of internal stateful reasoning modules."""
        logger.info("Resetting reasoning module states...")
        if self.use_reasoning_tracker and hasattr(self.reasoning_tracker, 'reset_state'):
            self.reasoning_tracker.reset_state()
        if self.use_algorithmic_reasoner and hasattr(self.algorithmic_reasoner, 'reset_state'):
            # Algorithmic reasoner state is usually reset implicitly in forward, but add if needed
            pass
        # Reset RWKV/SSM states within the transformer blocks
        for block in self.gpt_lm_head.transformer.blocks:
            if hasattr(block, 'rwkv_attn') and hasattr(block.rwkv_attn, 'reset_state'):
                block.rwkv_attn.reset_state()
            if hasattr(block, 'ssm') and hasattr(block.ssm, 'reset_state'):
                 block.ssm.reset_state()

        # Reset KV caches (important for starting new generation sequences)
        for block in self.gpt_lm_head.transformer.blocks:
            if hasattr(block.attn, 'kv_cache') and block.attn.kv_cache is not None:
                 # Determine current batch size (tricky here, maybe pass B=1?) or just clear content
                 block.attn.kv_cache.reset() # Simple content reset

        logger.info("Reasoning module states and KV Caches reset.")


# --- Enhanced Training Loop with MPS Optimizations ---

def train_model_optimized(model: MPSOptimizedEnhancedGPTLMHeadModel, train_loader: MPSDataLoader,
                         optimizer: torch.optim.Optimizer, epochs: int, device: torch.device,
                         accumulation_steps: int = 1, scheduler=None,
                         mixed_precision: bool = True, max_grad_norm: float = 1.0,
                         checkpoint_path: str = "checkpoints/model_optim.pt",
                         log_interval: int = 10, eval_interval: int = 500):
    """Optimized training loop for Apple Silicon GPUs with enhancements."""
    import torch
    import time
    import os
    import gc
    from tqdm.auto import tqdm
    from pathlib import Path

    # Setup mixed precision handler based on detected device
    mp_handler = MPSMixedPrecision(enabled=mixed_precision and device.type == 'mps')

    # Ensure checkpoint directory exists
    Path(os.path.dirname(checkpoint_path)).mkdir(parents=True, exist_ok=True)

    # Training metrics
    total_steps = len(train_loader) * epochs // accumulation_steps
    global_step = 0
    nan_counter = 0  # Track NaN losses
    warmup_steps = min(1000, max(100, total_steps // 10)) # Min 100 steps, max 10% or 1000
    base_lrs = [pg['lr'] for pg in optimizer.param_groups] # Store initial LRs

    logger.info(f"\n===== Starting Optimized Training =====")
    logger.info(f"Device: {device}, Epochs: {epochs}, Batch Size (eff): {train_loader.batch_size * accumulation_steps}")
    logger.info(f"Accum Steps: {accumulation_steps}, Total Steps: {total_steps}, LR Scheduler: {scheduler is not None}")
    logger.info(f"Mixed Precision ({'MPS' if device.type == 'mps' else 'CUDA/CPU'}): {mp_handler.enabled}, Max Grad Norm: {max_grad_norm}")
    logger.info(f"Warmup Steps: {warmup_steps}, Checkpoint Path: {checkpoint_path}")

    # Progress tracking
    progress_bar = tqdm(range(total_steps), desc="Optimized Training")

    # Training loop
    model.train() # Ensure model is in train mode
    model.to(device) # Ensure model is on the correct device
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_start = time.time()

        # Reset gradients at the start of epoch
        optimizer.zero_grad(set_to_none=True) # More memory efficient

        for i, batch in enumerate(train_loader):
            # Move batch to device
            try:
                input_ids = batch['input_ids'].to(device, non_blocking=device.type != 'mps') # non_blocking might help non-MPS
                attention_mask = batch['attention_mask'].to(device, non_blocking=device.type != 'mps')
                labels = batch['labels'].to(device, non_blocking=device.type != 'mps')
            except Exception as e:
                 logger.error(f"Error moving batch {i} to device {device}: {e}. Skipping batch.", exc_info=True)
                 continue # Skip batch if moving fails

            # Learning Rate Warmup (Linear)
            current_lr = base_lrs[0] # Default LR
            if global_step < warmup_steps:
                lr_scale = float(global_step + 1) / float(warmup_steps)
                current_lr = base_lrs[0] * lr_scale # Assuming single param group for simplicity
                for idx, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = base_lrs[idx] * lr_scale
            elif scheduler is not None and global_step >= warmup_steps:
                 # Get LR from scheduler *after* warmup
                 current_lr = scheduler.get_last_lr()[0]


            # Forward pass within autocast context if enabled
            with mp_handler as ctx: # Autocast for MPS or dummy context
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False # Important: Disable KV cache during training
                )
                loss = outputs['loss']
                aux_loss = outputs.get('aux_loss') # Get auxiliary loss if returned

                # Check for valid loss
                if loss is None:
                     logger.warning(f"Epoch {epoch+1}, Batch {i}: Loss is None. Skipping batch.")
                     continue # Skip batch if loss is None


                # Check for NaN/Inf loss and handle
                if not torch.isfinite(loss).all():
                    logger.error(f"NaN or Inf loss detected at step {global_step}! Loss: {loss.item()}. Skipping step.")
                    nan_counter += 1
                    if nan_counter >= 5:
                        logger.critical("Too many consecutive NaN/Inf losses. Stopping training.")
                        raise FloatingPointError("Training unstable - encountered too many NaN/Inf losses.")
                    # Skip optimizer step for this batch by clearing gradients
                    optimizer.zero_grad(set_to_none=True)
                    continue # Skip backpropagation and optimizer step
                else:
                    nan_counter = 0 # Reset counter on valid loss


            # Scale loss for gradient accumulation
            if accumulation_steps > 1:
                 # Ensure loss requires grad before division
                 if loss.requires_grad:
                      loss = loss / accumulation_steps
                 else:
                      logger.warning("Loss does not require grad before accumulation scaling. Check model output.")


            # Scale loss for mixed precision using the handler
            scaled_loss = mp_handler.scale_loss(loss, optimizer)

            # Backward pass on the scaled loss
            try:
                 scaled_loss.backward()
            except RuntimeError as e:
                 logger.error(f"RuntimeError during backward pass at step {global_step}: {e}. Skipping step.", exc_info=True)
                 optimizer.zero_grad(set_to_none=True) # Clear potentially corrupted grads
                 continue # Skip optimizer step


            # Update weights (optimizer step) after accumulation steps
            if (i + 1) % accumulation_steps == 0 or (i + 1 == len(train_loader)):
                # Use MP handler to step optimizer (handles unscaling, clipping, stepping, updating scaler)
                mp_handler.step(
                    optimizer=optimizer,
                    clip_grad=max_grad_norm, # Pass max norm for clipping
                    model=model # Pass model for clipping
                )
                # Zero gradients *after* stepping
                optimizer.zero_grad(set_to_none=True)

                # Step LR scheduler *after* optimizer step (if not using warmup phase)
                if scheduler is not None and global_step >= warmup_steps:
                    scheduler.step()


                # --- Logging and Checkpointing ---
                global_step += 1
                progress_bar.update(1)

                # Log metrics periodically
                if global_step % log_interval == 0:
                    elapsed = time.time() - start_time
                    samples_processed = global_step * train_loader.batch_size * accumulation_steps
                    samples_per_sec = samples_processed / elapsed if elapsed > 0 else 0

                    # Memory info (MPS specific)
                    mem_str = ""
                    if device.type == 'mps':
                        try:
                             mem_allocated = torch.mps.current_allocated_memory() / 1e9
                             mem_str = f", Mem: {mem_allocated:.2f} GB"
                        except Exception: pass # Ignore if memory info fails

                    # Log loss (use unscaled loss item), LR, throughput
                    log_loss = loss.item() * accumulation_steps # Log the effective loss for the step
                    log_msg = (f"E:{epoch+1}, S:{global_step}/{total_steps}, Loss:{log_loss:.4f}, "
                               f"LR:{current_lr:.2e}, Samples/s:{samples_per_sec:.2f}{mem_str}")
                    if aux_loss is not None:
                         log_msg += f", AuxLoss: {aux_loss.item():.4f}"

                    logger.info(log_msg)
                    progress_bar.set_postfix({
                        "Loss": f"{log_loss:.3f}",
                        "LR": f"{current_lr:.1e}",
                        "Samples/s": f"{samples_per_sec:.1f}"
                    })


                # Save checkpoint periodically based on eval_interval (steps)
                if global_step % eval_interval == 0:
                     logger.info(f"Saving checkpoint at step {global_step}...")
                     # Ensure model is on CPU for saving? Optional, might help compatibility.
                     # model.cpu()
                     save_dict = {
                          'epoch': epoch,
                          'global_step': global_step,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'loss': loss.item() * accumulation_steps, # Save last step loss
                          'config': model.config.__dict__ # Save config too
                     }
                     if scheduler:
                          save_dict['scheduler_state_dict'] = scheduler.state_dict()

                     torch.save(save_dict, checkpoint_path)
                     logger.info(f"Checkpoint saved to {checkpoint_path}")
                     # model.to(device) # Move back to training device

                     # Optional: Garbage collection and cache clearing
                     gc.collect()
                     if device.type == 'mps':
                          torch.mps.empty_cache()


            # Accumulate loss for epoch statistics (use unscaled loss)
            epoch_loss += loss.item() * accumulation_steps
            if aux_loss is not None:
                 epoch_aux_loss += aux_loss.item()


            # Periodic memory cleanup during epoch
            if i % 100 == 0 and device.type == 'mps':
                gc.collect()
                torch.mps.empty_cache()


        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / len(train_loader) / accumulation_steps # Avg loss per sample
        avg_epoch_aux_loss = epoch_aux_loss / len(train_loader) if epoch_aux_loss > 0 else 0.0
        epoch_time = time.time() - epoch_start

        logger.info(f"\n--- Epoch {epoch+1} Finished ---")
        logger.info(f"Time: {epoch_time:.2f}s, Avg Loss: {avg_epoch_loss:.4f}, Avg Aux Loss: {avg_epoch_aux_loss:.4f}")
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s. Avg Loss: {avg_epoch_loss:.4f}")

        # Save end-of-epoch checkpoint
        epoch_checkpoint_path = checkpoint_path.replace(".pt", f"_epoch_{epoch+1}.pt")
        logger.info(f"Saving end-of-epoch checkpoint to {epoch_checkpoint_path}...")
        save_dict = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
             'config': model.config.__dict__
        }
        if scheduler:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(save_dict, epoch_checkpoint_path)
        logger.info(f"End-of-epoch checkpoint saved.")

        # Optional: Check model parameters for stability issues at end of epoch
        with torch.no_grad():
             for name, param in model.named_parameters():
                  if not torch.isfinite(param).all():
                       logger.error(f"NaN/Inf detected in parameter '{name}' at end of epoch {epoch+1}!")
                  # Add check for large values if desired
                  # if param.abs().max() > 100:
                  #      logger.warning(f"Large value detected in parameter {name}: {param.abs().max().item()}")


    # --- End of Training ---
    progress_bar.close()
    total_time = time.time() - start_time
    logger.info(f"\n===== Training Complete =====")
    logger.info(f"Total time: {total_time:.2f}s, Final Avg Loss: {avg_epoch_loss:.4f}")
    print(f"Training complete! Total time: {total_time:.2f}s, Final Avg Loss: {avg_epoch_loss:.4f}")

    # Final memory report
    if device.type == 'mps':
         try:
              print(f"Final MPS memory allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
         except Exception: pass

    return global_step, avg_epoch_loss


# --- Optimized Prompting Function ---

def prompt_model_optimized(
    model: MPSOptimizedEnhancedGPTLMHeadModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    do_sample: bool = True,
    use_calculator_tool: bool = True, # Controlled by model config usually
    calculator_trigger: str = "[CALC:", # Trigger string
    calculator_end: str = "]",       # End string
    device: torch.device = None,     # Allow specifying device
    use_tot_search: bool = False     # Flag to use ToT search
) -> str:
    """Optimized prompting function with MPS considerations and tool/ToT integration."""
    import torch
    import time

    # Determine device if not provided
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    logger.info(f"Prompting using device: {device}")

    # Ensure model is on the correct device and in eval mode
    model.eval()
    model.to(device)

    # Reset reasoning states and clear cache before generation
    if hasattr(model, 'reset_reasoning_states'):
        model.reset_reasoning_states()
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
         torch.cuda.empty_cache()


    # --- Check if using Tree of Thought ---
    if use_tot_search:
        if not model.use_tree_of_thought:
             logger.warning("ToT search requested, but 'use_tree_of_thought' is False in model config. Skipping ToT.")
        else:
             logger.info("Attempting generation using Tree of Thought search...")
             # ToT search handles its own generation loop and tokenization
             return model.run_tot_search(prompt_text, tokenizer, generation_length=max_new_tokens)
             # Note: ToT search needs to be robustly implemented in the model class


    # --- Standard Autoregressive Generation with Optional Tool Use ---
    start_time = time.time()
    current_text = prompt_text
    generated_sequence = "" # Track only the generated part for tool checking

    try:
        # Tokenize initial prompt
        all_generated_ids = tokenizer.encode(current_text, return_tensors="pt").to(device)
        prompt_length = all_generated_ids.shape[1]
    except Exception as e:
         logger.error(f"Tokenization Error for prompt: {e}", exc_info=True)
         return "[Prompt Tokenization Error]"


    print(f"\n--- Prompting (Device: {device}) ---")
    print(f"Initial Prompt: {current_text}")

    active_generation_length = 0
    eos_token_id = tokenizer.eos_token_id
    # Handle lists of EOS tokens if tokenizer provides them
    if isinstance(eos_token_id, list):
        eos_token_id_list = eos_token_id
    elif eos_token_id is not None:
        eos_token_id_list = [eos_token_id]
    else:
        eos_token_id_list = None


    # Main generation loop
    for _ in range(max_new_tokens):
        # Prepare inputs for the model's generate method
        input_ids = all_generated_ids
        attention_mask = torch.ones_like(input_ids) # Simple mask, KV cache handles history

        # Use model's internal generate function - generate one token at a time
        # if checking for tools, otherwise could generate more.
        # Let's simplify: generate one token always for tool check loop.
        generate_step_length = 1

        try:
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, # Pass mask, generate should handle it
                max_new_tokens=generate_step_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id, # Pass EOS id(s) to internal generate
                do_sample=do_sample,
            )
        except Exception as e:
            logger.error(f"Error during model.generate call: {e}", exc_info=True)
            break # Stop generation on error


        # Extract *only* the newly generated token ID(s)
        new_token_ids = output_ids[0, input_ids.shape[1]:]

        if len(new_token_ids) == 0:
            logger.info("Generation stopped: model returned empty sequence.")
            break # Generation stopped (e.g., internal EOS)

        # Decode the new token(s) - skip special tokens for intermediate checks
        # Use skip_special_tokens=False initially to correctly detect EOS etc.
        new_text = tokenizer.decode(new_token_ids, skip_special_tokens=False)

        # Check for EOS token in the *newly* generated tokens
        if eos_token_id_list and any(eos in new_token_ids for eos in eos_token_id_list):
            # Include the EOS token in the output? Usually yes.
            # Decode *with* special tokens up to EOS if needed.
            # Let's decode skipping special tokens for final output, but detect here.
            logger.info("[EOS token detected in generated token]")
            generated_sequence += tokenizer.decode(new_token_ids, skip_special_tokens=True) # Add final part
            all_generated_ids = output_ids # Update full sequence
            active_generation_length += len(new_token_ids)
            break # Stop generation loop


        # Append non-special decoded text to ongoing generated sequence
        generated_sequence += tokenizer.decode(new_token_ids, skip_special_tokens=True)
        all_generated_ids = output_ids # Update the full sequence ID tensor


        active_generation_length += len(new_token_ids)

        # --- Tool Check: Calculator ---
        # Check if the trigger appears in the *cumulative* generated sequence
        can_use_calc = use_calculator_tool and model.use_calculator and hasattr(model, 'run_calculator')
        if can_use_calc and calculator_trigger in generated_sequence:
            start_idx = generated_sequence.rfind(calculator_trigger) # Find the *last* trigger
            if start_idx != -1:
                # Check if the end trigger also exists *after* the start trigger
                end_idx = generated_sequence.find(calculator_end, start_idx + len(calculator_trigger))
                if end_idx != -1:
                    # Extract expression
                    expression = generated_sequence[start_idx + len(calculator_trigger):end_idx].strip()
                    logger.info(f"[Tool Call Detected] Expression: '{expression}'")
                    print(f"  [Calculator Call: '{expression}']")


                    # Call calculator tool method on the model
                    calc_result = model.run_calculator(expression)
                    logger.info(f"[Tool Result] Output: '{calc_result}'")
                    print(f"  [Calculator Result: {calc_result}]")


                    # --- Inject result back into the context ---
                    # Replace the trigger, expression, and end marker with the result
                    generated_sequence_before = generated_sequence[:start_idx]
                    generated_sequence_after = generated_sequence[end_idx + len(calculator_end):]
                    generated_sequence = generated_sequence_before + f" {calc_result} " + generated_sequence_after

                    # Rebuild the full text context
                    current_text = prompt_text + generated_sequence

                    # Re-tokenize the *entire updated* text to reset the generation state
                    try:
                         all_generated_ids = tokenizer.encode(current_text, return_tensors="pt").to(device)
                         logger.info(f"[Context Updated] New context token length: {all_generated_ids.shape[1]}")
                         print(f"  [Context Updated. New tail: ...{current_text[-80:]}]")
                    except Exception as e:
                         logger.error(f"Tokenization Error after tool use: {e}", exc_info=True)
                         break # Stop generation if re-tokenization fails

                    # Clear KV cache after tool use as history has changed significantly
                    if hasattr(model, 'reset_reasoning_states'):
                         model.reset_reasoning_states() # This should reset caches too
                    # Re-fill cache with the new context (expensive but necessary)
                    logger.info("Re-filling KV cache after tool use...")
                    model.eval()
                    with torch.no_grad():
                         new_mask = torch.ones_like(all_generated_ids)
                         new_pos_ids = torch.arange(0, all_generated_ids.shape[1], device=device).unsqueeze(0)
                         _ = model(all_generated_ids, attention_mask=new_mask, use_cache=True, position_ids=new_pos_ids)
                    logger.info("KV cache re-filled.")


        # Periodic memory cleanup during long generations
        if active_generation_length % 20 == 0: # Every 20 tokens
             if device.type == 'mps': torch.mps.empty_cache()
             elif device.type == 'cuda': torch.cuda.empty_cache()
             gc.collect()


    # --- Generation Complete ---
    generation_time = time.time() - start_time
    tokens_per_second = active_generation_length / generation_time if generation_time > 0 else 0

    print(f"\n--- Generation Complete ---")
    print(f"Generated {active_generation_length} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/sec)")

    # Decode the final generated sequence, excluding the prompt
    # Use the final `all_generated_ids` which includes EOS if detected.
    # Decode *without* skip_special_tokens to see EOS, then potentially strip manually.
    final_response_ids = all_generated_ids[0, prompt_length:]
    final_response = tokenizer.decode(final_response_ids, skip_special_tokens=True).strip()

    print(f"Final Response:\n{final_response}")

    # Final memory cleanup
    gc.collect()
    if device.type == 'mps': torch.mps.empty_cache()
    elif device.type == 'cuda': torch.cuda.empty_cache()

    return final_response


# --- Optimized Checkpoint Saver ---

def save_checkpoint_periodically_optimized(model: nn.Module, save_path: str, interval_sec: int = 120):
    """Periodically saves model state dict in a background thread with MPS/CPU handling."""
    import os
    import threading
    import time
    import torch
    from pathlib import Path

    # Create directory if it doesn't exist
    save_dir = Path(os.path.dirname(save_path))
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Setting up periodic checkpoint saving to {save_path} every {interval_sec}s.")

    # Use a flag to signal the thread to stop
    stop_event = threading.Event()

    def save_func():
        while not stop_event.is_set():
            # Wait for the interval initially and between saves
            stop_event.wait(interval_sec)
            if stop_event.is_set(): # Check again after waiting
                 break

            try:
                logger.info(f"Attempting periodic checkpoint save to {save_path}...")
                current_device = next(model.parameters()).device

                # Option 1: Save directly from current device (simpler, might have issues)
                # torch.save(model.state_dict(), save_path)

                # Option 2: Move to CPU before saving (safer for compatibility/MPS issues)
                cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(cpu_state_dict, save_path)
                del cpu_state_dict # Clean up memory
                # Optional: Clear MPS cache after potentially large CPU copy
                if current_device.type == 'mps':
                     torch.mps.empty_cache()

                logger.info(f"Periodic checkpoint successfully saved at {save_path}")

            except Exception as e:
                logger.error(f"Error saving periodic checkpoint: {e}", exc_info=True)

    # Start the saving thread as a daemon so it doesn't block program exit
    save_thread = threading.Thread(target=save_func, daemon=True, name="CheckpointSaverThread")
    save_thread.start()
    logger.info(f"Started automatic checkpoint saving thread.")

    # Return the stop event so the main thread can signal shutdown
    return stop_event, save_thread


# --- Deprecated/Original Training/Prompting Functions (Keep for reference?) ---
# It's generally better to remove unused code, but keeping them commented out
# might be useful if the user needs to refer back to the non-MPS-optimized versions.

# class SimpleDataset(Dataset): ... # Original SimpleDataset if needed
# def train_model(...): ... # Original train_model loop
# def prompt_model(...): ... # Original prompt_model function

# --- End of flashgpt_model.py ---
