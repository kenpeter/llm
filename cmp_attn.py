import torch

# seed 123
torch.manual_seed(123)
# device cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"PyTorch version: {torch.__version__}")

# batch size 8
batch_size = 8
# context len 1024, 1024*768 dim
context_len = 1024
# embed dim = 768 (768 features in single token)
embed_dim = 768

# 768 = 2^8 × 3 = 256 × 3
# 768 = 2^6 × 12 = 64 × 12
# 768 = 2^4 × 48 = 16 × 48
# each token has 768 features = 2^6 × 12_head = 64_each_head_look_at × 12_head

# (8, 1024, 768)
# basically single batch, single context len, has 768 dim
embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)


# ==============

import torch.nn as nn

# casual attention
# single head
class CausalAttention(nn.Module):

    # w_q, w_key, w_val
    # dropout
    # mask
    # init
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # (768, 64)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        # (768, 64)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        # (768, 64)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # drop out
        self.dropout = nn.Dropout(dropout)
        # mask
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))  # New

    # x is input, also embedding
    def forward(self, x):
        # x: (8, 1024, 768)
        b, num_tokens, d_in = x.shape  
        # q, k, v: each (8, 1024, 768) → (8, 1024, 64), because linear
        keys = self.W_key(x)      # (8, 1024, 64)
        queries = self.W_query(x) # (8, 1024, 64)
        values = self.W_value(x)  # (8, 1024, 64)

        # (8, 1024, 64) @ (8, 1024, 64).Tr -> (8, 1024, 64) @ (8, 64, 1024) → (8, 1024, 1024) -> (b, context_len, context_len)
        attn_scores = queries @ keys.transpose(1, 2)
        # Mask future tokens: set to -inf
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # Normalize to probabilities: (8, 1024, 1024)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (8, 1024, 1024) @ (8, 1024, 64) → (8, 1024, 64) -> later 12 * (8, 1024, 64) -> (8, 1024, 768)
        context_vec = attn_weights @ values
        return context_vec


# each head has own weight, multi head
class Ch03_MHA_Wrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            # d_in: 768, d_out: 64, context_length: 1024
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)] # 12 head
        )

        # d_out * num_heads = 64 * 12 = 768
        self.out_proj = nn.Linear(d_out*num_heads, d_out*num_heads)

    def forward(self, x):
        # x: (b, token_n, d_in) = (8, 1024, 768)
        # each head: (8, 1024, 768) → (8, 1024, 64)
        # concat: 12 heads × 64 dims = (8, 1024, 768)
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.out_proj(context_vec)


# create wrapper instance
mha_ch03_wrapper = Ch03_MHA_Wrapper(
    # d_in = 768
    d_in=embed_dim,
    # 768 = 2^6 × 12 = 64 × 12; d_out = 64
    d_out=embed_dim//12,
    # 1024
    context_length=context_len,
    # no drop
    dropout=0.0,
    # 12 head
    num_heads=12,
    # qkv
    qkv_bias=False
).to(device) # to device

# embedding (8, 1024, 768)
out = mha_ch03_wrapper(embeddings)





# =========

# each head has no weight, but virtual multi head
class Ch03_MHA(nn.Module):
    # d_in: 768
    # d_out: 768
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # 768
        self.d_out = d_out
        # 12
        self.num_heads = num_heads
        # head_dim = 768 // 12 = 64
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        # (768, 768)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # (768, 768)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # (1024, 1024)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # x: (8, 1024, 768)
        b, num_tokens, d_in = x.shape

        # (b, num_tokens, d_out)
        # (8, 1024, 768) @ (768, 768).Tranpose -> (8, 1024, 768) @ (768, 768) -> (8, 1024, 768)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # (8, 1024, 768) -> (8, 1024, 12, 64)
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # (8, 1024, 12, 64) -> (8, 12, 1024, 64)
        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # (b, num_heads, num_tokens, head_dim) @ (b, num_heads, num_tokens, head_dim).T(2, 3)
        # (8, 12, 1024, 64) @ (8, 12, 1024, 64).T(2, 3) -> (8, 12, 1024, 64) @ (8, 12, 64, 1024) -> (8, 12, 1024, 1024)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, num_tokens, num_tokens) @ (b, num_heads, num_tokens, head_dim)
        # (8, 12, 1024, 1024) @ (8, 12, 1024, 64) -> (8, 12, 1024, 64).T(1, 2) -> (8, 1024, 12, 64) -> 12 * 64 = 768
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


mha_ch03 = Ch03_MHA(
    # 768
    d_in=embed_dim,
    # 768
    d_out=embed_dim,
    # 1024
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

out = mha_ch03(embeddings)
print(out.shape)



# =========

import torch.nn as nn


class MultiHeadAttentionCombinedQKV(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        # Ensure d_out (768) is divisible by num_heads (12)
        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        # Store parameters: 12 heads, 1024 context length
        self.num_heads = num_heads
        self.context_length = context_length
        # Each head processes: 768 / 12 = 64 dimensions
        self.head_dim = d_out // num_heads

        # (768, 768*3) -> (768, 2304)
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        # Output projection: (768, 768)
        self.proj = nn.Linear(d_out, d_out)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Causal mask: upper triangular matrix (1024, 1024)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # Input shape: x = (8, 1024, 768)
        batch_size, num_tokens, embed_dim = x.shape

        # Apply single QKV linear layer: (8, 1024, 768) → (8, 1024, 2304)
        # 2304 = 3 * 768 (concatenated Q, K, V projections)
        qkv = self.qkv(x)

        # Reshape to separate Q, K, V: (8, 1024, 2304) → (8, 1024, 3, 12, 64)
        # 2304 = 3 * 12 * 64 (3 projections × 12 heads × 64 dims per head)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # Rearrange dimensions: (8, 1024, 3, 12, 64) → (3, 8, 12, 1024, 64)
        # (b, token_n, projection, head_n, head_dim) → (projection, b, head_n, token_n, head_dim)
        # Move the "3" (Q, K, V) to the front for easy separation
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Split into Q, K, V: (3, 8, 12, 1024, 64) → 3 × (8, 12, 1024, 64)
        # (projection, batch, head_n, token_n, head_dim) → 3 × (batch, head_n, token_n, head_dim)
        # queries, keys, values each have shape (8, 12, 1024, 64)
        # unbind at index 0, means split
        queries, keys, values = qkv.unbind(0)

        # Compute attention scores: (8, 12, 1024, 64) @ (8, 12, 64, 1024) → (8, 12, 1024, 1024)
        # Each head computes token-to-token attention scores
        attn_scores = queries @ keys.transpose(-2, -1)
        # Apply causal mask: set future positions to -inf
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        # Convert scores to probabilities: (8, 12, 1024, 1024)
        # NOTE: keys.shape[-1]**-0.5 should be **0.5 (positive exponent for scaling)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: (8, 12, 1024, 1024) @ (8, 12, 1024, 64) → (8, 12, 1024, 64)
        # Get weighted combination of value vectors
        context_vec = attn_weights @ values

        # Transpose back: (8, 12, 1024, 64) → (8, 1024, 12, 64)
        # Move heads dimension back to prepare for concatenation
        context_vec = context_vec.transpose(1, 2)

        # Concatenate heads: (8, 1024, 12, 64) → (8, 1024, 768)
        # Flatten the last two dimensions: 12 * 64 = 768
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, embed_dim)

        # Final output projection: (8, 1024, 768) → (8, 1024, 768)
        context_vec = self.proj(context_vec)

        return context_vec


# Create the combined QKV multi-head attention model
mha_combined_qkv = MultiHeadAttentionCombinedQKV(
    d_in=embed_dim,        # Input dimension: 768
    d_out=embed_dim,       # Output dimension: 768  
    context_length=context_len,  # Context length: 1024
    dropout=0.0,           # No dropout
    num_heads=12,          # 12 attention heads
    qkv_bias=False         # No bias in QKV projection
).to(device)

# Run forward pass: (8, 1024, 768) → (8, 1024, 768)
out = mha_combined_qkv(embeddings)
print(out.shape)  # Should print: torch.Size([8, 1024, 768])