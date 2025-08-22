import torch
import numpy
import torch.nn as nn

class simplifiedAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        attn_scores = x @ x.T
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_vector = attn_weights @ x
        return context_vector
    
# inputs = torch.tensor([
#     [0.43, 0.15, 0.89],  # Your       (x^1)
#     [0.55, 0.87, 0.66],  # journey    (x^2)
#     [0.57, 0.85, 0.64],  # starts     (x^3)
#     [0.22, 0.58, 0.33],  # with       (x^4)
#     [0.77, 0.25, 0.10],  # one        (x^5)
#     [0.05, 0.80, 0.55]   # step       (x^6)
# ])

# v1 = simplifiedAttention()
# context = v1(inputs)
# print(context)
        