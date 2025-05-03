import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int
    n_embd: int
    n_layer: int
    n_head: int
    dropout_prob: float = 0.1
    alibi: bool = False
    use_rope: bool = True
    flash_attention: bool = True
    n_kv_heads: Optional[int] = None
    use_gqa: bool = False
    use_rwkv: bool = False
    use_ssm: bool = False
    use_moe: bool = False
    num_experts: int = 8
    top_k_experts: int = 2
    gradient_checkpointing: bool = False
    
    # Added for reasoning modules
    mla_n_latent: int = 16
    use_mla: bool = True
    reasoning_steps: int = 3
    use_reasoning_tracker: bool = True
    algorithmic_reasoner_registers: int = 4
    use_algorithmic_reasoner: bool = False
    
    # Added for tools
    use_calculator: bool = True

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_head 