import torch
import torch.nn as nn

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],
    ]  # step     (x^6)
)


query = inputs[1]  # 2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0])

# 1. cal attention score for each row -> single element value for this row
for i, x_i in enumerate(inputs):
    # 1x3 dot 1x3 -> 1x3 dot 1x3.T -> 1x3 dot 3x1 -> 1 ele value for entire row
    attn_scores_2[i] = torch.dot(x_i, query)

# 2. normalize each single value (represent each row)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

query = inputs[1]

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    # 1x3 dot 1x3 -> 1x3 dot 1x3.T -> 1x3 dot 3x1 -> single ele for entire row
    context_vec_2 += attn_weights_2[i] * x_i

# attn_scores = torch.empty(6, 6)

# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)

# attention score
# 6x3 dot 3x6 -> 6x6
# this retain all input rows
attn_scores = inputs @ inputs.T


# attention weight -> 6x6
attn_weights = torch.softmax(attn_scores, dim=-1)


# context 6x6 dot 6x3 -> 6x3
# attention goes 1st means how much attention I need to pay for each input
all_context_vecs = attn_weights @ inputs


# ========================
# 2nd input ele
x_2 = inputs[1]
# (6, 3)
# d_in = 3
d_in = inputs.shape[1]
# output dim = 2
d_out = 2

torch.manual_seed(123)

a = torch.rand(d_in, d_out)
b = torch.rand(d_in, d_out)
c = torch.rand(d_in, d_out)
W_query = torch.nn.Parameter(a, requires_grad=False)
W_key = torch.nn.Parameter(b, requires_grad=False)
W_value = torch.nn.Parameter(c, requires_grad=False)

# 6x3 @ 3x2
query_2 = x_2 @ W_query  # _2 because it's with respect to the 2nd input element
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

"""
tensor([0.4306, 1.4551]) <---- q_2, 1x2
tensor([0.4433, 1.1419]) <---- k_2
tensor([0.3951, 1.0037]) <---- v_2
"""


# 1. it means other ele in inputs cannot be directly used, but attach with W_key and W_value
# 2. after this we can use
keys = inputs @ W_key
values = inputs @ W_value


"""
6x2
tensor([[0.3669, 0.7646],
        [0.4433, 1.1419], <----- k_2
        [0.4361, 1.1156],
        [0.2408, 0.6706],
        [0.1827, 0.3292],
        [0.3275, 0.9642]])

"""

"""
6x2
tensor([[0.1855, 0.8812],
        [0.3951, 1.0037],
        [0.3879, 0.9831],
        [0.2393, 0.5493],
        [0.1492, 0.3346],
        [0.3221, 0.7863]])

"""


# the key for 2nd input token
keys_2 = keys[1]
# [0.4306, 1.4551] dot [0.4433, 1.1419] = 0.4306 * 0.4433 + 1.4551 * 1.1419 = 1.85 (*)
# [0.4306, 1.4551] dot [0.4433, 1.1419] = 0.4306 * 1.1419 + 1.4551 * 0.4433 = 1.13 (x)
attn_score_22 = query_2.dot(keys_2)

# 1x2 @ 6x2.T -> 1x6
attn_scores_2 = query_2 @ keys.T  # All attention scores for given query



d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)



# 1. the individual input for the query
# 2. q_2 = W_q * x_2
# 3. att_score_2 = q_2 @ all_keys.T
# 4. att_score_2 -> normalize -> att_weight_2
# 5. all_context = att_weight_2 @ value (because value is the last one not used.)
context_vec_2 = attn_weights_2 @ values



# ============


# self attention v1
# nn.Module
class SelfAttention_v1(nn.Module):
    # def init
    # self, d_in, d_out
    def __init__(self, d_in, d_out):
        # super init
        super().__init__()
        # W_q, W_k, W_val
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    # forward, all inputs x
    def forward(self, x):
        # all inputs form key, query, value
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        # query @ keys.T -> att score*
        attn_scores = queries @ keys.T # omega
        # normalize -> softmax (sum 1) -> att weight*
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        # att weight @ values -> context
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)





# =============

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)



# =====

# Reuse the query and key weight matrices of the
# SelfAttention_v2 object from the previous section for convenience
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs) 
attn_scores = queries @ keys.T
print(attn_scores)

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)


context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)


masked_simple = attn_weights * mask_simple
print(masked_simple)