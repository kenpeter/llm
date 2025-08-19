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
# model_configs = {
#     "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},   # Smallest model
#     "gpt-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, # Medium model
#     "gpt-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},  # Large model
#     "gpt-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},    # Extra large model
# }

model_configs = {
    "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},   # Smallest model
    "gpt-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, # Medium model
}


# Device selection: CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# batch size 2
batch_size = 2
# input tensor; from 0 to 50257; (2, 1024)
input_tensor = torch.randint(0, 50257, (batch_size, 1024)).to(device)

# loop diff model config, to see FLOPS
for size in model_configs:
    # object got merged
    BASE_CONFIG.update(model_configs[size])

    # Create GPT model with current config
    model = GPTModel(BASE_CONFIG)
    # Convert to bfloat16 only if on CUDA (CPU doesn't support bfloat16 well)
    if device.type == "cuda":
        model = model.bfloat16()
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
    # Clear GPU cache to prevent memory buildup (only if using CUDA)
    if device.type == "cuda":
        torch.cuda.empty_cache()



# ===============
flops_per_second = {
    # GPU specifications
    # https://www.techpowerup.com/gpu-specs/h100-pcie-80-gb.c3899
    "H100": {
        torch.float32: 51.22e12,  # 51.22 TFLOPs for FP32 on NVIDIA H100
        torch.float16: 204.9e12,  # 204.9 TFLOPs for FP16 on NVIDIA H100
        torch.bfloat16: 204.9e12
    },
    # https://www.techpowerup.com/gpu-specs/l4.c4091
    "L4": {
        torch.float32: 30.29e12,  # 30.29 TFLOPs for FP32 on NVIDIA L4
        torch.float16: 30.29e12,  # 30.29 TFLOPs for FP16 on NVIDIA L4
        torch.bfloat16: 30.29e12
    },
    # https://www.techpowerup.com/gpu-specs/tesla-t4.c3316
    "T4": {
        torch.float32: 8.1e12,  # 8.1 TFLOPs for FP32 on NVIDIA T4
        torch.float16: 65.13e12,  # 65.13 TFLOPs for FP16 on NVIDIA T4
        torch.bfloat16: 65.13e12
    },
    # https://www.techpowerup.com/gpu-specs/a10g.c3798
    "A10G": {
        torch.float32: 31.52e12,  # 31.52 TFLOPs for FP32 on NVIDIA A10G
        torch.float16: 31.52e12,  # 31.52 TFLOPs for FP16 on NVIDIA A10G
        torch.bfloat16: 31.52e12
    },
    # https://www.techpowerup.com/gpu-specs/a100-pcie-40-gb.c3623
    "A100": {
        torch.float32: 19.49e12,  # 19.49 TFLOPs for FP32 on NVIDIA A100
        torch.float16: 77.97e12,  # 77.97 TFLOPs for FP16 on NVIDIA A100
        torch.bfloat16: 77.97e12
    },
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621
    "RTX_3080": {
        torch.float32: 29.77e12,  # 29.77 TFLOPs for FP32 on NVIDIA RTX 3080
        torch.float16: 29.77e12,  # 29.77 TFLOPs for FP16 on NVIDIA RTX 3080
        torch.bfloat16: 29.77e12
    },
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622
    "RTX_3090": {
        torch.float32: 35.58e12,  # 35.58 TFLOPs for FP32 on NVIDIA RTX 3090
        torch.float16: 35.58e12,  # 35.58 TFLOPs for FP16 on NVIDIA RTX 3090
        torch.bfloat16: 35.58e12
    },
    # CPU approximate performance (varies by processor)
    # These are rough estimates for modern CPUs
    "CPU": {
        torch.float32: 1.0e12,   # ~1 TFLOP for modern CPU (very rough estimate)
        torch.float16: 1.0e12,   # CPUs typically don't optimize for float16
        torch.bfloat16: 1.0e12   # CPUs typically don't optimize for bfloat16
    }
}


import time

def get_device_model(flops_per_second_dict, device):
    if device.type == "cpu":
        return "CPU"
    elif device.type == "cuda":
        try:
            device_name = torch.cuda.get_device_name(0)
            for model in flops_per_second_dict.keys():
                if model in device_name and model != "CPU":
                    return model
            return "Unknown"  # Default if no matching GPU model is found
        except:
            return "Unknown"
    else:
        return "Unknown"


device_model = get_device_model(flops_per_second, device)
print("Device Model:", device_model)

if device_model != "Unknown":

    for size in model_configs:
        print(f"\nProcessing {size}")
        config = BASE_CONFIG.copy()
        config.update(model_configs[size])

        min_batch_size = 1
        max_batch_size = None
        max_possible_batch_size = 4096

        while min_batch_size <= max_possible_batch_size:
            batch_size = (min_batch_size + max_possible_batch_size) // 2
            try:
                input_tensor = torch.randint(
                    0, config["vocab_size"],
                    (batch_size, config["context_length"]),
                    device=device
                )

                # Create model and set precision based on device
                model = GPTModel(config)
                if device.type == "cuda":
                    model = model.bfloat16()
                model = model.to(device)
                model.train()

                # Start timing with proper synchronization
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()

                # Forward & backward pass
                output = model(input_tensor)
                loss = output.sum()  # Compute a dummy loss
                loss.backward()

                # End timing with proper synchronization
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()

                total_time_seconds = end_time - start_time

                # Calculate FLOPs for forward pass
                macs, params = profile(model, inputs=(input_tensor,), verbose=False)
                flops_forward = 2 * macs  # Assuming one MAC equals two FLOPs

                # Estimate FLOPs for backward pass (typically 2x forward FLOPs)
                flops_backward = 2 * flops_forward

                # Total FLOPs for forward + backward passes
                total_flops = flops_forward + flops_backward  # Or total_flops = flops_forward * 3

                data_type = next(model.parameters()).dtype
                max_flops_per_second = flops_per_second[device_model].get(data_type, 0)

                # Compute tokens per second
                tokens_processed = batch_size * config["context_length"]
                tokens_per_second = tokens_processed / total_time_seconds

                # Compute FLOPs per token
                flops_per_token = total_flops / tokens_processed

                # Compute theoretical max tokens per second
                if flops_per_token > 0:
                    theoretical_max_tokens_per_second = max_flops_per_second / flops_per_token
                else:
                    theoretical_max_tokens_per_second = 0  # Avoid division by zero

                # Compute MFU
                if theoretical_max_tokens_per_second > 0:
                    mfu = tokens_per_second / theoretical_max_tokens_per_second
                else:
                    mfu = 0  # Avoid division by zero

                print(f"  Batch size {batch_size}: Tokens/sec: {tokens_per_second:.2f}, MFU: {mfu:.4f}")

                # If successful, try a larger batch size
                min_batch_size = batch_size + 1
                max_batch_size = batch_size

                # Clean up
                del model, input_tensor, output, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Try smaller batch size
                    max_possible_batch_size = batch_size - 1

                    # Clean up
                    try:
                        del model, input_tensor
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    except NameError:
                        pass
                else:
                    raise e

else:
    print("Unknown device model. Please update the flops_per_second dictionary with your device information.")