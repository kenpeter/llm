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
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    # forward, all inputs x
    def forward(self, x):
        # all inputs form key, query, value
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # query @ keys.T -> att score*
        attn_scores = queries @ keys.T  # omega
        # normalize -> softmax (sum 1) -> att weight*
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

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
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

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


# ====
context_length = attn_scores.shape[0]
# we apply the mask
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)


# then we do the normalization
# -inf -> softmax -> 0, because e^(-inf), how the softmax formula works
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)

# =======================
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)  # dropout rate of 50%
example = torch.ones(6, 6)  # create a matrix of ones


# =================

# dim = 0, dim = 1 start at y (x, y, z), dim = 2 start at z (x, y, z)
# 2 inputs with 6 tokens each, and each token has embedding dimension 3
batch = torch.stack((inputs, inputs), dim=0)


class CausalAttention(nn.Module):
    # def init
    # self, d_in, d_out, context_len, dropout, bias
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        # super init
        super().__init__()
        # d_in, d_out
        self.d_out = d_out
        # w query (linear func)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        # w key (linear func)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        # w value (linear func)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # dropout
        self.dropout = nn.Dropout(dropout)  # New
        # causal mask
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )  # New

    def forward(self, x):
        # b, num token, d_in, all have x shape
        b, num_tokens, d_in = x.shape  # New batch dimension b
        # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
        # in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs
        # do not exceed `context_length` before reaching this forward method.

        # k, q, v
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # (2, 6, 2) @ (2, 2, 6) --> first few can stay same, or 1 wildcard, last 2 dim can matrix cal
        attn_scores = queries @ keys.transpose(1, 2)

        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # after mask, sum up to 1
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # dropout
        attn_weights = self.dropout(attn_weights)  # New

        # context vec
        context_vec = attn_weights @ values
        # context vec
        return context_vec


torch.manual_seed(123)

# # e.g. 6. how many data in arr
# context_length = batch.shape[1]
# # return context vec, with causal attention
# ca = CausalAttention(d_in, d_out, context_length, 0.0)

# context_vecs = ca(batch)


# multi head attention
class MultiHeadAttentionWrapper(nn.Module):
    # def init
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)

context_length = batch.shape[1]  # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)


# ===================


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # x: (2, 6, 3)
        b, num_tokens, d_in = x.shape

        # linear: transformation only change the last dim, as out = x @ W.transpose + bias
        # linear: stores weight as (d_out, d_in) but uses as (d_in, d_out)
        # (b, token_n, d_out) @ (d_in, d_out)
        # (2, 6, 3) @ (3, 2) -> (2, 6, 2)
        keys = self.W_key(x)  # Shape: (2, 6, 2)
        queries = self.W_query(x)
        values = self.W_value(x)

        # (b, token_n, d_out) -> (b, token_n, head_n, head_dim); giant single matrix, vitually split
        # head_n * head_dim = d_out; 2 * 1 = 2
        # (2, 6, 2) -> (2, 6, 2, 1)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, token_n, head_n, head_dim) -> (b, head_n, token_n, head_dim)
        # hierachy meaning change. orig: single token has 2 head. now: single head has 6 token
        # computation: do head 1st
        # (2, 6, 2, 1) -> (2, 2, 6, 1)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # basically we use single matrix, virtually split, before attn score, attn weight, context

        # (b, head_n, token_n, head_dim) @ (b, head_n, head_dim, token_n).Tr -> (b, head_n, token_n, token_n)
        # (2, 2, 6, 1) @ (2, 2, 1, 6) -> (2, 2, 6, 6); last (6, 6) token
        attn_scores = queries @ keys.transpose(2, 3)
        # because last 2 dim are (6, 6) token, we can easily apply mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # top right mask
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # softmax/dropput, (2, 2, 6, 6) -> (2, 2, 6, 6)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, head_n, token_n, token_n) @ (b, head_n, token_n, head_dim) -> (b, head_n, token_n, head_dim) -> tr -> (b, token_n, head_n, head_dim)
        # (2, 2, 6, 6) @ (2, 2, 6, 1) -> (2, 2, 6, 1) -> Tr -> (2, 6, 2, 1), slowlly back to orig
        context_vec = (attn_weights @ values).transpose(1, 2)

        # (b, token_n, head_n, head_dim) @ (b, token_n, d_out)
        # original now, (2, 6, 2, 1) -> (2, 6, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # (2, 6, 2) @ (2, 2) -> (2, 6, 2), same shape, but went through out = x @ weight.T + bias
        context_vec = self.out_proj(context_vec)

        return context_vec


torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)
