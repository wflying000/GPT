from typing import List
from dataclasses import dataclass


@dataclass
class GPTConfig:
    hidden_size: int = 768
    num_heads: int = 12
    max_sequence_length: int = 512
    intermediate_size: int = 3072
    num_layers: int = 6
    vocab_size: int = 30000
    dropout: float = 0.1
    bias: bool = False


@dataclass
class TrainingArguments:
    num_epochs: int = 10
    num_grad_accumulation: int = 1
    write_step: int = 1
    cuda_item: List = None
    is_master_process: bool = True
