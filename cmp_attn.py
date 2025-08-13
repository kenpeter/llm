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



# =============

import math


class MHAEinsum(nn.Module):
    """Multi-Head Attention using Einstein Summation (einsum) for more readable tensor operations."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # Ensure output dimension (768) is divisible by number of heads (12)
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # Store dimensions: 768 output, 12 heads
        self.d_out = d_out
        self.num_heads = num_heads
        # Each head processes: 768 / 12 = 64 dimensions
        self.head_dim = d_out // num_heads

        # Manual weight parameters (instead of nn.Linear)
        # Shape: (768, 768) - note: transposed compared to nn.Linear internal storage
        self.W_query = nn.Parameter(torch.randn(d_out, d_in))
        self.W_key = nn.Parameter(torch.randn(d_out, d_in))
        self.W_value = nn.Parameter(torch.randn(d_out, d_in))

        # Optional bias terms for Q, K, V projections
        if qkv_bias:
            # Each bias has shape (768,)
            self.bias_q = nn.Parameter(torch.zeros(d_out))
            self.bias_k = nn.Parameter(torch.zeros(d_out))
            self.bias_v = nn.Parameter(torch.zeros(d_out))
        else:
            # Register as None to indicate no bias (cleaner than delattr)
            self.register_parameter("bias_q", None)
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        # Output projection layer: (768, 768)
        self.out_proj = nn.Linear(d_out, d_out)
        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)
        # Causal mask: upper triangular matrix (1024, 1024)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        # Initialize all parameters with proper scaling
        self.reset_parameters()


    def reset_parameters(self):
        """Initialize weights and biases following PyTorch's nn.Linear initialization."""
        # Kaiming uniform initialization for weights (good for ReLU-like activations)
        nn.init.kaiming_uniform_(self.W_query, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_key, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_value, a=math.sqrt(5))
        # Initialize biases (if present) with small uniform values
        if self.bias_q is not None:
            # Calculate fan_in for proper bias scaling
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_query)
            bound = 1 / math.sqrt(fan_in)  # Standard bias initialization bound
            nn.init.uniform_(self.bias_q, -bound, bound)
            nn.init.uniform_(self.bias_k, -bound, bound)
            nn.init.uniform_(self.bias_v, -bound, bound)

    def forward(self, x):
        # Input: x = (8, 1024, 768) = (batch, tokens, embed_dim)
        b, n, _ = x.shape

        # Linear transformations using einsum: (batch, tokens, embed) @ (embed, out) → (batch, tokens, out)
        # "bnd,di->bni" means: (b,n,d) × (d,i) → (b,n,i)
        Q = torch.einsum("bnd,di->bni", x, self.W_query)  # (8, 1024, 768)
        K = torch.einsum("bnd,di->bni", x, self.W_key)    # (8, 1024, 768)
        V = torch.einsum("bnd,di->bni", x, self.W_value)  # (8, 1024, 768)

        # Add bias terms if enabled (broadcast across batch and sequence dimensions)
        if self.bias_q is not None:
            Q += self.bias_q  # (8, 1024, 768) + (768,) → (8, 1024, 768)
            K += self.bias_k  # Broadcasting handles dimension matching
            V += self.bias_v

        # Reshape for multi-head attention: (8, 1024, 768) → (8, 1024, 12, 64) → (8, 12, 1024, 64)
        Q = Q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores using einsum: (b,h,n,d) × (b,h,m,d) → (b,h,n,m)
        # "bhnd,bhmd->bhnm" means: Q(b,h,n,d) × K(b,h,m,d) → scores(b,h,n,m)
        scores = torch.einsum("bhnd,bhmd->bhnm", Q, K) / (self.head_dim ** 0.5)

        # Apply causal mask: expand (1024, 1024) → (8, 12, 1024, 1024)
        mask = self.mask[:n, :n].unsqueeze(0).unsqueeze(1).expand(b, self.num_heads, n, n)
        scores = scores.masked_fill(mask.bool(), -torch.inf)

        # Convert scores to attention probabilities and apply dropout
        attn_weights = torch.softmax(scores, dim=-1)    # (8, 12, 1024, 1024)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values using einsum: (b,h,n,m) × (b,h,m,d) → (b,h,n,d)
        # "bhnm,bhmd->bhnd" means: weights(b,h,n,m) × V(b,h,m,d) → context(b,h,n,d)
        context_vec = torch.einsum("bhnm,bhmd->bhnd", attn_weights, V)

        # Recombine heads: (8, 12, 1024, 64) → (8, 1024, 12, 64) → (8, 1024, 768)
        context_vec = context_vec.transpose(1, 2).reshape(b, n, self.d_out)
        # Final output projection: (8, 1024, 768) → (8, 1024, 768)
        context_vec = self.out_proj(context_vec)

        return context_vec


# Create MHA model using einsum operations
mha_einsum = MHAEinsum(
    d_in=embed_dim,        # Input dimension: 768
    d_out=embed_dim,       # Output dimension: 768
    context_length=context_len,  # Context length: 1024
    dropout=0.0,           # No dropout
    num_heads=12,          # 12 attention heads
    qkv_bias=False         # No bias in QKV projections
).to(device)

# Run forward pass: (8, 1024, 768) → (8, 1024, 768)
out = mha_einsum(embeddings)
print(out.shape)  # Should print: torch.Size([8, 1024, 768])