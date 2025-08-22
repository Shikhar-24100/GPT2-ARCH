import torch
import numpy
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        K = self.W_key(x)
        Q = self.W_query(x)
        V = self.W_value(x)

        attn_scores = Q @ K.transpose(-2, -1)     # (batch, seq, seq)
        attn_scores = attn_scores / (K.shape[-1] ** 0.5)

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, seq, seq)
        context_vector = attn_weights @ V                  # (batch, seq, d_out)

        return context_vector



# torch.manual_seed(121)
# d_in = 3
# d_out = 2
# sa_v2 = SelfAttention(d_in, d_out)
# inputs = torch.tensor([
#     [0.43, 0.15, 0.89],  # Your       (x^1)
#     [0.55, 0.87, 0.66],  # journey    (x^2)
#     [0.57, 0.85, 0.64],  # starts     (x^3)
#     [0.22, 0.58, 0.33],  # with       (x^4)
#     [0.77, 0.25, 0.10],  # one        (x^5)
#     [0.05, 0.80, 0.55]   # step       (x^6)
# ])
# print("Context_vector =",   sa_v2(inputs))

