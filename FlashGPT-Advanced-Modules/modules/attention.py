import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .config import GPTConfig

class BaseAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        
        # Initialize KV cache
        self.kv_cache = None
        
    def _init_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        if self.kv_cache is None:
            self.kv_cache = KVCache(
                max_batch_size=batch_size,
                max_seq_len=self.config.max_seq_len,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                dtype=dtype,
                device=device
            )
    
    def _process_kv_cache(self, k: torch.Tensor, v: torch.Tensor, 
                         use_cache: bool, position_ids: Optional[torch.LongTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_cache:
            self._init_kv_cache(k.size(0), k.dtype, k.device)
            self.kv_cache.update(k, v, position_ids)
            k, v = self.kv_cache.get(k.size(0))
        return k, v
    
    def _manual_attention(self, q, k, v, attention_mask, alibi_bias, is_causal):
        # Implementation of manual attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            # Ensure attention_mask has the right shape
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # Expand attention_mask to match scores shape if needed
            if attention_mask.size(-1) != scores.size(-1):
                attention_mask = attention_mask.expand(-1, -1, scores.size(-2), -1)
            
            scores = scores + attention_mask
            
        if alibi_bias is not None:
            scores = scores + alibi_bias
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output
    
    def _compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         attention_mask: Optional[torch.Tensor], 
                         alibi_bias: Optional[torch.Tensor]) -> torch.Tensor:
        # Implementation of attention computation
        return self._manual_attention(q, k, v, attention_mask, alibi_bias, is_causal=True)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                alibi_bias: Optional[torch.Tensor] = None, use_cache: bool = False,
                position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Split into heads
        q = x.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = x.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = x.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Process KV cache if needed
        k, v = self._process_kv_cache(k, v, use_cache, position_ids)
        
        # Compute attention
        attn_output = self._compute_attention(q, k, v, attention_mask, alibi_bias)
        
        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        return attn_output

class MultiHeadSelfAttention(BaseAttention):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        # Initialize query, key, value projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Initialize dropout
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                alibi_bias: Optional[torch.Tensor] = None, use_cache: bool = False,
                position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Split into heads
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Process KV cache if needed
        k, v = self._process_kv_cache(k, v, use_cache, position_ids)
        
        # Compute attention
        attn_output = self._compute_attention(q, k, v, attention_mask, alibi_bias)
        
        # Apply dropout
        attn_output = self.dropout(attn_output)
        
        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Project output
        attn_output = self.out_proj(attn_output)
        
        return attn_output

class GroupedQueryAttention(BaseAttention):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        # Initialize query, key, value projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd // config.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd // config.n_kv_heads, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Initialize dropout
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                alibi_bias: Optional[torch.Tensor] = None, use_cache: bool = False,
                position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Split into heads
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Process KV cache if needed
        k, v = self._process_kv_cache(k, v, use_cache, position_ids)
        
        # Compute attention
        attn_output = self._compute_attention(q, k, v, attention_mask, alibi_bias)
        
        # Apply dropout
        attn_output = self.dropout(attn_output)
        
        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Project output
        attn_output = self.out_proj(attn_output)
        
        return attn_output

class RWKVAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        # Initialize RWKV parameters
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, config.n_embd))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, config.n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, config.n_embd))
        
        # Initialize projections
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Initialize dropout
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                alibi_bias: Optional[torch.Tensor] = None, use_cache: bool = False,
                position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Apply time mixing
        xk = x * self.time_mix_k + x * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x * (1 - self.time_mix_r)
        
        # Project to key, value, and receptance
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        
        # Split into heads
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        r = r.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Compute RWKV attention
        w = torch.exp(k)
        wkv = torch.cumsum(w * v, dim=2) / torch.cumsum(w, dim=2)
        wkv = wkv * r
        
        # Merge heads
        wkv = wkv.transpose(1, 2).contiguous()
        wkv = wkv.view(batch_size, seq_len, -1)
        
        # Apply dropout
        wkv = self.dropout(wkv)
        
        # Project output
        output = self.output(wkv)
        
        return output

class KVCache:
    def __init__(self, max_batch_size: int, max_seq_len: int, n_kv_heads: int, 
                 head_dim: int, dtype=torch.float32, device: Optional[torch.device] = None):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        self.k_cache = None
        self.v_cache = None
        self.current_position = 0
        
    def _initialize_device(self, input_tensor: torch.Tensor):
        if self.device is None:
            self.device = input_tensor.device
            self.k_cache = torch.zeros(
                (self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device
            )
            self.v_cache = torch.zeros_like(self.k_cache)
    
    def update(self, k: torch.Tensor, v: torch.Tensor, position: Optional[int] = None):
        if self.k_cache is None:
            self._initialize_device(k)
        
        if position is None:
            position = self.current_position
            
        self.k_cache[:, position] = k
        self.v_cache[:, position] = v
        self.current_position = position + 1
    
    def get(self, current_batch_size: int):
        return self.k_cache[:current_batch_size, :self.current_position], \
               self.v_cache[:current_batch_size, :self.current_position]
    
    def reset(self, batch_size: Optional[int] = None):
        if batch_size is not None:
            self.max_batch_size = batch_size
        self.current_position = 0
        if self.k_cache is not None:
            self.k_cache.zero_()
            self.v_cache.zero_() 