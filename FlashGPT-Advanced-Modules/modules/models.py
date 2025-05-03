import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .config import GPTConfig
from .attention import MultiHeadSelfAttention, GroupedQueryAttention, RWKVAttention
from .embeddings import RMSNorm, RotaryPositionEmbeddings
from .optimization import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Initialize attention mechanism based on config
        if config.use_rwkv:
            self.attn = RWKVAttention(config)
        elif config.use_gqa:
            self.attn = GroupedQueryAttention(config)
        else:
            self.attn = MultiHeadSelfAttention(config)
            
        # Initialize normalization layers
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)
        
        # Initialize feed-forward network
        self.ffn = SwiGLU(
            hidden_dim=config.n_embd,
            expansion_factor=8/3,
            dropout_prob=config.dropout_prob
        )
        
        # Initialize dropout
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                alibi_bias: Optional[torch.Tensor] = None, use_cache: bool = False,
                position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        
        # Attention path
        attn_output = self.attn(
            self.norm1(x),
            attention_mask=attention_mask,
            alibi_bias=alibi_bias,
            use_cache=use_cache,
            position_ids=position_ids
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward path
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        
        return x

class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Position embeddings
        self.wpe = nn.Embedding(config.max_seq_len, config.n_embd)
        
        # Dropout
        self.drop = nn.Dropout(config.dropout_prob)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = RMSNorm(config.n_embd)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False, position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        
        # Get sequence length
        seq_len = input_ids.size(1)
        
        # Get position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        # Get token embeddings
        tok_emb = self.wte(input_ids)
        
        # Get position embeddings
        pos_emb = self.wpe(position_ids)
        
        # Combine embeddings
        x = self.drop(tok_emb + pos_emb)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, use_cache=use_cache, position_ids=position_ids)
            
        # Apply final layer norm
        x = self.ln_f(x)
        
        return x

class GPTLMHeadModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.transformer.wte.weight
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, use_cache: bool = False,
                position_ids: Optional[torch.LongTensor] = None) -> Dict[str, Optional[torch.Tensor]]:
        
        # Get transformer outputs
        hidden_states = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            position_ids=position_ids
        )
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text from input_ids using the model."""
        # Set model to eval mode
        self.eval()
        
        # Initialize output sequence
        output_ids = input_ids.clone()
        
        # Generate tokens
        for _ in range(max_new_tokens):
            # Get logits for next token
            outputs = self(
                input_ids=output_ids,
                attention_mask=attention_mask,
                use_cache=True
            )
            next_token_logits = outputs['logits'][:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append next token to output sequence
            output_ids = torch.cat([output_ids, next_token], dim=1)
            
            # Update attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
                ], dim=1)
            
            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break
        
        return output_ids 