import torch
from torch import nn
import torch.nn.functional as F

from config import n_head, n_embed

class SelfAttention(nn.Module):

    def __init__(self):
        super().__init__()
        assert n_embed % n_head == 0
        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.n_head = n_head
        self.n_embd = n_embed

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.c_fc    = nn.Linear(n_embed, 4*n_embed)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = SelfAttention()
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Model(nn.Module):

    def __init__(self, num_blocks=1):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(13, n_embed),
            h = nn.ModuleList([Block() for _ in range(num_blocks)]),
            ln_f = nn.LayerNorm(n_embed),
        ))

        self.lm_head = nn.Linear(30*n_embed, 4*10, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer.wte(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        x = x.flatten(start_dim=1)
        logits = self.lm_head(x)
        # logits = self.sigmoid(out)

        return logits