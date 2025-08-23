import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    """
    
    
    """
    def __init__(self, d_in, d_out, context_length, drop_out, qkv_bias = False):
        super().__init__()
        self.d_out = d_out

        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(drop_out)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch, num_tokens, input_dim = x.shape
        K = self.W_key(x)
        Q = self.W_query(x)
        V = self.W_value(x)

        attn_scores = Q @ K.transpose(1,2)
        attn_scores.masked_fill_(
        self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores/(K.shape[-1]**0.5), dim = -1)

        # dropout
        attn_weights = self.dropout(attn_weights)
        context_vector = attn_weights @ V
        return context_vector
    

# inputs = torch.tensor([
#     [0.43, 0.15, 0.89],  # Your       (x^1)
#     [0.55, 0.87, 0.66],  # journey    (x^2)
#     [0.57, 0.85, 0.64],  # starts     (x^3)
#     [0.22, 0.58, 0.33],  # with       (x^4)
#     [0.77, 0.25, 0.10],  # one        (x^5)
#     [0.05, 0.80, 0.55]   # step       (x^6)
# ])


# batch = torch.stack((inputs, inputs), dim=0)
# # print(batch.shape)
# print("batch:\n", batch)


# d_in = 3 
# d_out = 2
# torch.manual_seed(123)
# context_length = batch.shape[1]
# ca = CausalAttention(d_in, d_out, context_length, 0.2)
# context_vecs = ca(batch)
# print("context_vecs.shape:", context_vecs.shape)
# print(context_vecs)

