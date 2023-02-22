import math
import torch
from torch import nn

class GPTAttention(nn.Module):
    def __init__(self, config):
        super(GPTAttention, self).__init__()
        assert config.hidden_size % config.num_heads == 0, \
            f"hidden_size {config.hidden_size} can not be devided by num_heads {config.num_heads}"
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.input_proj = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=config.bias)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        max_seq_len = config.max_sequence_length
        subsequent_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.uint8)).view(
            1, 1, max_seq_len, max_seq_len
        )
        self.register_buffer("subsequent_mask", subsequent_mask)

    def forward(self, hidden_states):

        batch_size, seq_len, hidden_size = hidden_states.size()

        query, key, value = self.input_proj(hidden_states).split(hidden_size, dim=2)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [bsz, num_heads, seq_len, head_dim]
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [bsz, num_heads, seq_len, head_dim]
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [bsz, num_heads, seq_len, head_dim]

        # scaled dot product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights.masked_fill(self.subsequent_mask[:, :, :seq_len, :seq_len] == 0, -1e10)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights) # [bsz, num_heads, seq_len, seq_len]
        
        attn_output = torch.matmul(attn_weights, value) # [bsz, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_output = self.output_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class GPTMLP(nn.Module):
    def __init__(self, config):
        super(GPTMLP, self).__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class GPTLayer(nn.Module):
    def __init__(self, config):
        super(GPTLayer, self).__init__()
        self.attn = GPTAttention(config)
        self.mlp = GPTMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states):
        # x = x + attn(norm(x))
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_output = self.attn(hidden_states)
        hidden_states = residual + attn_output
        
        # x = x + mlp(norm(x))
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states

class GPTModel(nn.Module):
    def __init__(self, config):
        super(GPTModel, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.layers = nn.ModuleList([GPTLayer(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.config = config

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        seq_len = input_ids.size(-1)
        assert seq_len <= self.config.max_sequence_length, \
            f"Input length {seq_len} is longer than model's max length {self.config.max_sequence_length}"

        device = input_ids.device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

        position_embedding = self.position_embedding(position_ids)
        token_embedding = self.token_embedding(input_ids)
        hidden_states = token_embedding + position_embedding
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        hidden_states = self.norm(hidden_states)

        outputs = {"hidden_states": hidden_states}

        return outputs


class GPTLMModel(nn.Module):
    def __init__(self, config):
        super(GPTLMModel, self).__init__()
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, inputs):
        hidden_states = self.transformer(inputs)["hidden_states"]
        logits = self.lm_head(hidden_states)
        outputs = {"logits": logits}

        return outputs






    



        
