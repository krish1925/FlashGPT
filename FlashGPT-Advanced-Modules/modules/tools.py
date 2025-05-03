import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
import sympy as sp
import numpy as np

class CalculatorTool:
    def __init__(self):
        self.supported_operations = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '^': lambda x, y: x ** y,
            'sqrt': lambda x: np.sqrt(x),
            'sin': lambda x: np.sin(x),
            'cos': lambda x: np.cos(x),
            'tan': lambda x: np.tan(x),
            'log': lambda x: np.log(x),
            'exp': lambda x: np.exp(x)
        }
        
    def calculate(self, expression: str) -> Dict[str, Optional[str]]:
        try:
            # Parse expression using sympy
            expr = sp.sympify(expression)
            
            # Convert to numerical value
            result = float(expr.evalf())
            
            return {
                "result": str(result),
                "error": None
            }
        except Exception as e:
            return {
                "result": None,
                "error": str(e)
            }

class TreeOfThought:
    def __init__(self, model, tokenizer, num_branches: int = 3, max_depth: int = 3, beam_size: int = 2):
        self.model = model
        self.tokenizer = tokenizer
        self.num_branches = num_branches
        self.max_depth = max_depth
        self.beam_size = beam_size
        
    @torch.no_grad()
    def _evaluate_state(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
        # Evaluate the quality of a state
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)
        return probs.mean().item()
        
    @torch.no_grad()
    def _expand_node(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                    temperature: float = 0.7) -> List[tuple[torch.Tensor, torch.Tensor, float]]:
        # Generate multiple continuations from a state
        states = []
        for _ in range(self.num_branches):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                temperature=temperature,
                do_sample=True
            )
            new_input_ids = outputs
            new_attention_mask = torch.ones_like(new_input_ids)
            score = self._evaluate_state(new_input_ids, new_attention_mask)
            states.append((new_input_ids, new_attention_mask, score))
        return states
        
    @torch.no_grad()
    def search(self, initial_prompt: str, generation_length: int = 50) -> str:
        # Initialize search
        input_ids = self.tokenizer.encode(initial_prompt, return_tensors='pt')
        attention_mask = torch.ones_like(input_ids)
        
        # Initialize beam
        beam = [(input_ids, attention_mask, 0.0)]
        
        # Search loop
        for depth in range(self.max_depth):
            new_beam = []
            for state in beam:
                # Expand current state
                new_states = self._expand_node(state[0], state[1])
                new_beam.extend(new_states)
            
            # Select top states
            beam = sorted(new_beam, key=lambda x: x[2], reverse=True)[:self.beam_size]
            
            # Check if any state has reached generation length
            for state in beam:
                if state[0].size(1) >= generation_length:
                    return self.tokenizer.decode(state[0][0])
        
        # Return best state
        return self.tokenizer.decode(beam[0][0][0])

class AlgorithmicReasoner(nn.Module):
    def __init__(self, hidden_dim: int, num_registers: int = 4, max_steps: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_registers = num_registers
        self.max_steps = max_steps
        
        # Initialize registers
        self.registers = nn.Parameter(torch.zeros(num_registers, hidden_dim))
        
        # Initialize operations
        self.operation_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_registers * 3)  # For each register: operation, source1, source2
        )
        
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, List[torch.Tensor]]:
        # Initialize reasoning trace
        reasoning_trace = []
        
        # Process each step
        for step in range(self.max_steps):
            # Get operations for this step
            operations = self.operation_net(hidden_states)
            operations = operations.view(-1, self.num_registers, 3)
            
            # Apply operations to registers
            for reg_idx in range(self.num_registers):
                op, src1, src2 = operations[0, reg_idx]
                
                # Apply operation
                if op > 0.5:  # Add
                    self.registers[reg_idx] = self.registers[int(src1)] + self.registers[int(src2)]
                else:  # Multiply
                    self.registers[reg_idx] = self.registers[int(src1)] * self.registers[int(src2)]
            
            # Store reasoning step
            reasoning_trace.append(self.registers.clone())
            
            # Update hidden states
            hidden_states = hidden_states + self.registers.mean(dim=0)
        
        return hidden_states, reasoning_trace 