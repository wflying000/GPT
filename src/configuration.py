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


