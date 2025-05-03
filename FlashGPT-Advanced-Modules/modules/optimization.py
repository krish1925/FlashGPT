import torch
import torch.nn as nn
import os
from typing import Optional

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
    else:
        device = torch.device("cpu")
        print("MPS device not found, using CPU")
    
    # Return optimized device
    return device

def chunked_matmul(a: torch.Tensor, b: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    """Perform matrix multiplication in chunks to reduce memory usage."""
    # Implementation of chunked matrix multiplication
    pass

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, expansion_factor: float = 8/3, dropout_prob: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_dim = int(hidden_dim * expansion_factor)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize weights
        self.w1 = nn.Linear(hidden_dim, self.expansion_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, self.expansion_dim, bias=False)
        self.w3 = nn.Linear(self.expansion_dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add checks for NaN and clipping for stability
        x1 = self.w1(x)
        x2 = self.w2(x)
        
        # Apply Swish activation
        x1 = x1 * torch.sigmoid(x1)
        
        # Combine and apply final linear layer
        x = self.w3(x1 * x2)
        x = self.dropout(x)
        
        return x 