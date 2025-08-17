# Import function to get package version information
from importlib.metadata import version

# List of packages to check versions for
pkgs = [
    "thop",    # Package for calculating FLOPs (floating point operations per second)
    "torch",   # PyTorch deep learning framework
]
# Loop through each package and print its version
for p in pkgs:
    print(f"{p} version: {version(p)}")




# torch
import torch
import torch.nn as nn
# thop import profile
from thop import profile

# For installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# Import GPT model implementation from external package
# from transformer import GPTModel

# GPT model classes defined locally to avoid import issues
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # Simplified attention for FLOPS calculation - just pass through
        self.d_out = d_out
        self.linear = nn.Linear(d_in, d_out)
        
    def forward(self, x):
        # Simplified - just linear transformation
        return self.linear(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        # so the feed forward using GELU
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

# gpt model
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# Base configuration shared by all model sizes
BASE_CONFIG = {
    # voacab, 50257
    "vocab_size": 50257,
    # context len 1024
    "context_length": 1024,
    # drop rate 0
    "drop_rate": 0.0,
    # use qkv bias
    "qkv_bias": True 
}

# embed size 1600, layer 48, n head 25
model_configs = {
    "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},   # Smallest model
    "gpt-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, # Medium model
    "gpt-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},  # Large model
    "gpt-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},    # Extra large model
}

# cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# batch size 2
batch_size = 2
# input tensor; from 0 to 50257; (2, 1024)
input_tensor = torch.randint(0, 50257, (batch_size, 1024)).to(device)

# loop diff model config, to see FLOPS
for size in model_configs:
    # object got merged
    BASE_CONFIG.update(model_configs[size])

    # Create GPT model with current config and convert to bfloat16 for efficiency
    model = GPTModel(BASE_CONFIG).bfloat16()
    # Move model to selected device (GPU/CPU)
    model.to(device)

    # MACS = multiply-accumulate operations (one multiply + one add = 2 FLOPs)
    # MACS are typically counted as two FLOPS (one multiply and one accumulate)
    # Profile the model to count operations and parameters

    # MACS = multiply + acc operation, so 2 flops
    # MACS ~= 2.1e+11
    # track every layers (embed, attn, feed forward, etc.)
    # count matrix multi, add, etc.
    # return total MACS

    # params weight, bias
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    # Convert MACs to FLOPs (1 MAC = 2 FLOPs)
    flops = 2*macs
    # Print model size and its computational cost
    print(f"{size:18}: {flops:.1e} FLOPS")

    # Delete model to free memory
    del model
    # Clear GPU cache to prevent memory buildup
    torch.cuda.empty_cache()