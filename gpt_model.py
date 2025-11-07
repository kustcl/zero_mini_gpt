import torch
import torch.nn as nn
from dataclasses import dataclass
import math
import torch.nn.functional as F


@dataclass
class LLMConfig:
    d_model: int = 512
    context_length: int = 30
    num_heads: int = 8
    head_size: int = d_model // num_heads
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout: float = 0.1
    vocab_size: int = 60000
    n_layer: int = 3


gpt_config = LLMConfig()


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.ReLU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wq = nn.Linear(config.d_model, config.head_size)
        self.wk = nn.Linear(config.d_model, config.head_size)
        self.wv = nn.Linear(config.d_model, config.head_size)
        self.register_buffer('mask', torch.tril(torch.ones(config.context_length, config.context_length)))
        self.dropout = nn.Dropout(config.dropout)
        self.head_size = config.head_size

    def forward(self, x):
        B, T, D = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        weight = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        weight = weight.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        out = weight @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Attention(config) for _ in range(config.num_heads)])
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mha = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        out = x + self.ffn(self.ln2(x))
        return out


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding_table = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.context_length = config.context_length

    def forward(self, idx, targets=None):
        batch_size, seq_length = idx.size()
        token_emb = self.token_embedding_table(idx)
        token_pos = self.position_embedding_table(torch.arange(seq_length, device=idx.device))
        x = token_emb + token_pos
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(input=logits, target=targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) < self.context_length else idx[:, -self.context_length:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

