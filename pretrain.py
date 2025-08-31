#!/usr/bin/env python3

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
from datetime import datetime
import urllib.request
import numpy as np


#####################################
# Dataset and DataLoader
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


#####################################
# Model Components
#####################################


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


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
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


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


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # final norm before out head
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        # [b, token_n, vocab_size]
        logits = self.out_head(x)
        return logits


#####################################
# Configuration
#####################################

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-key-value bias
}


#####################################
# Pretrained Weight Loading
#####################################


# if shape not match, err
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    # Convert to Parameter properly to avoid warnings
    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.detach().clone())
    else:
        return torch.nn.Parameter(torch.tensor(right))


# load weight into gpt - this weight file is already in our model format!
def load_weights_into_gpt(gpt, params):
    # Direct mapping since weights are already in our model's format
    print("Loading embeddings...")
    # own model context len
    model_context_len = gpt.pos_emb.weight.shape[0]
    # param has posi embed weight, with content len
    pretrained_pos_emb = params["pos_emb.weight"][:model_context_len]
    # posi embed weight
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, pretrained_pos_emb)
    # token embed weight
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["tok_emb.weight"])

    print(
        f"Loading {len([k for k in params.keys() if k.startswith('trf_blocks')])} transformer blocks..."
    )
    # Get number of transformer blocks from the model
    num_blocks = len(gpt.trf_blocks)

    for b in range(num_blocks):
        # q weight and bias
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight,
            params[f"trf_blocks.{b}.att.W_query.weight"],
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias,
            params[f"trf_blocks.{b}.att.W_query.bias"],
        )

        # key weight and bias
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight,
            params[f"trf_blocks.{b}.att.W_key.weight"],
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, params[f"trf_blocks.{b}.att.W_key.bias"]
        )

        # value weight and bias
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight,
            params[f"trf_blocks.{b}.att.W_value.weight"],
        )
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias,
            params[f"trf_blocks.{b}.att.W_value.bias"],
        )

        # project weight and bias
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params[f"trf_blocks.{b}.att.out_proj.weight"],
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params[f"trf_blocks.{b}.att.out_proj.bias"],
        )

        # feed forward weight
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params[f"trf_blocks.{b}.ff.layers.0.weight"],
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params[f"trf_blocks.{b}.ff.layers.0.bias"],
        )

        # there is 1 gap here, GELU
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params[f"trf_blocks.{b}.ff.layers.2.weight"],
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params[f"trf_blocks.{b}.ff.layers.2.bias"],
        )

        # layer normalize
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params[f"trf_blocks.{b}.norm1.scale"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params[f"trf_blocks.{b}.norm1.shift"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params[f"trf_blocks.{b}.norm2.scale"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params[f"trf_blocks.{b}.norm2.shift"]
        )

    print("Loading final layers...")

    # output = scale * normalized + shift
    # norm is too strict, so need scale and shift
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["final_norm.scale"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["final_norm.shift"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["out_head.weight"])


def download_progress_hook(block_num, block_size, total_size):
    """Progress hook for urllib.request.urlretrieve"""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100.0, (downloaded / total_size) * 100.0)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)

        # Print progress with carriage return to overwrite same line
        print(
            f"\rDownloading... {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
            end="",
            flush=True,
        )

        # Print newline when complete
        if percent >= 100.0:
            print()


# download weight to hugging face
def download_and_load_gpt2_weights(model_size="gpt2-small (124M)", file_name=None):
    # 4 diff pre weights
    if file_name is None:
        file_mapping = {
            "gpt2-small (124M)": "gpt2-small-124M.pth",
            "gpt2-medium (355M)": "gpt2-medium-355M.pth",
            "gpt2-large (774M)": "gpt2-large-774M.pth",
            "gpt2-xl (1558M)": "gpt2-xl-1558M.pth",
        }
        file_name = file_mapping[model_size]

    # url
    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"

    # Check if file already exists and get its size
    if os.path.exists(file_name):
        file_size_mb = os.path.getsize(file_name) / (1024 * 1024)
        print(f"Found existing {file_name} ({file_size_mb:.1f} MB) - skipping download")
    else:
        # url download
        print(f"Starting download of {file_name} from HuggingFace...")
        try:
            urllib.request.urlretrieve(
                url, file_name, reporthook=download_progress_hook
            )
            file_size_mb = os.path.getsize(file_name) / (1024 * 1024)
            print(f"Download complete! Saved to {file_name} ({file_size_mb:.1f} MB)")
        except Exception as e:
            print(f"Download failed: {e}")
            if os.path.exists(file_name):
                os.remove(file_name)  # Clean up partial download
            raise

    print(f"Loading weights from {file_name}...")
    # torch load we map to cpu first
    weights = torch.load(file_name, map_location="cpu")
    print("Weights loaded successfully!")
    return weights


#####################################
# Utility Functions
#####################################


def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=0.8):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply temperature scaling
        logits = logits / temperature

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Sample from the probability distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=-1)  # (batch, n_tokens+1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    # by default most of things start in cpu, e.g. load file, torch.tensor, etc, so we need to move to GPU
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # use model's forward func to get logit
    logits = model(input_batch)
    # 1. input_batch: [1_b, 3_token_n]
    # 2. logit: [1_b, 3_token_n, 50257_vocab_size] -> flat(0, 1) -> [3_token_n, 50257_vocab_size]
    # 3. target: [1_b, 3_token_n] -> flat_all -> [3_token_n]
    # 4. cross entropy has internal loop -> loop logit match target
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    # return loss
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    # total loss
    total_loss = 0.0
    # data loader empty
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        # 1. data loader return input batch and target batch. and we loop them together.
        # 2. input batch (no shift one), target batch (shift one)
        num_batches = len(data_loader)
    else:
        # limit
        num_batches = min(num_batches, len(data_loader))
    # now loop both
    for i, (input_batch, target_batch) in enumerate(data_loader):
        # Process only the specified number of batches
        if i < num_batches:
            # return single num, because big num no good.
            # input_batch -> logit -> cal cross entropy loss with target_batch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # just acc all loss
            total_loss += loss.item()
        else:
            # Stop processing after reaching num_batches
            break
    # Return average loss across all processed batches
    return total_loss / num_batches


#####################################
# Training Infrastructure
#####################################


def get_device():
    """
    Detect and return the best available device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        # Clear cache and set memory fraction
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB"
        )
    elif torch.backends.mps.is_available():
        # MPS has precision issues with this model - use CPU for now
        device = torch.device("cpu")
        print("⚠️  MPS detected but using CPU due to precision issues")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_text_data(file_path="the-verdict.txt"):
    """
    Load text data from file or download if not available
    """
    if not os.path.exists(file_path):
        print(f"Downloading {file_path}...")
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    return text_data


def save_checkpoint(model, optimizer, epoch, loss, lr, checkpoint_dir="checkpoints"):
    """
    Save model checkpoint including model state, optimizer state, and training info
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "lr": lr,
        "config": GPT_CONFIG_124M,
        "timestamp": datetime.now().isoformat(),
    }

    # Save latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)

    # Save epoch-specific checkpoint
    epoch_checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt"
    )
    torch.save(checkpoint, epoch_checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Load model checkpoint and resume training state
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None, 0, float("inf")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["loss"]

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Resumed from epoch {checkpoint['epoch']}, loss: {best_loss:.4f}")
    return model, optimizer, start_epoch, best_loss


def train_epoch(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epoch,
    global_step,
    accumulation_steps=8,
    log_interval=25,
    start_context_param="Who are you?",
    temperature=0.8,
    warmup_steps=0,
    initial_lr=1e-4,
    peak_lr=5e-5,
    lr_increment=0,
):
    """
    Train model for one epoch with gradient accumulation and periodic validation
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    optimizer.zero_grad()

    for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
        )

        # Scale loss by accumulation steps for backward pass
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        # Add unscaled loss for correct averaging
        total_loss += loss.item()

        # Update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == num_batches - 1:
            # Apply learning rate warmup
            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_increment
            else:
                lr = peak_lr

            # Apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            # global step is model run step so far
            global_step += 1

            # Print progress with both train and validation loss every log_interval steps
            if global_step % log_interval == 0:
                current_train_loss = total_loss / (batch_idx + 1)

                # Quick validation loss calculation
                model.eval()
                val_loss = 0.0
                val_samples = 0
                with torch.no_grad():
                    for val_batch_idx, (val_input, val_target) in enumerate(val_loader):
                        if val_batch_idx >= 5:  # Only use first 5 batches for speed
                            break
                        val_input, val_target = val_input.to(device), val_target.to(
                            device
                        )
                        val_logits = model(val_input)
                        val_batch_loss = torch.nn.functional.cross_entropy(
                            val_logits.flatten(0, 1), val_target.flatten()
                        )
                        val_loss += val_batch_loss.item()
                        val_samples += 1

                val_loss = val_loss / val_samples if val_samples > 0 else float("inf")

                # Generate sample text to show progress
                import tiktoken

                tokenizer = tiktoken.get_encoding("gpt2")
                start_context = start_context_param

                try:
                    token_ids = text_to_token_ids(start_context, tokenizer).to(device)
                    with torch.no_grad():
                        generated_ids = generate_text_simple(
                            model=model,
                            idx=token_ids,
                            max_new_tokens=8,
                            context_size=256,  # Use context length from config
                            temperature=temperature,
                        )
                    generated_text = token_ids_to_text(generated_ids, tokenizer)
                    # Extract only the new tokens (remove the original context)
                    new_tokens = generated_text[len(start_context) :].strip()
                except:
                    new_tokens = "[generation failed]"

                model.train()

                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"ep {epoch} (step {global_step}): train loss {current_train_loss:.4f}, val loss {val_loss:.4f}, lr {current_lr:.6f} | {start_context}{new_tokens}"
                )

    return total_loss / num_batches, global_step


def evaluate(model, val_loader, device):
    """
    Evaluate model on validation set
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        total_loss = calc_loss_loader(val_loader, model, device)

    return total_loss


def run_inference(
    model, device, prompt, max_tokens=50, temperature=0.8, context_size=256
):
    """
    Run inference with the model to generate text from a prompt
    """
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"🤖 GPT Inference Mode")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    print("-" * 50)

    model.eval()

    # Encode the prompt
    token_ids = text_to_token_ids(prompt, tokenizer).to(device)

    # Generate text
    with torch.no_grad():
        generated_ids = generate_text_simple(
            model=model,
            idx=token_ids,
            max_new_tokens=max_tokens,
            context_size=context_size,
            temperature=temperature,
        )

    # Debug: check token IDs
    print(
        f"Debug - Generated token IDs: {generated_ids[0].tolist()[:10]}..."
    )  # First 10 tokens
    print(
        f"Debug - Token ID range: {generated_ids.min().item()} to {generated_ids.max().item()}"
    )
    print(f"Debug - Vocab size: {tokenizer.n_vocab}")

    # Decode the generated text
    generated_text = token_ids_to_text(generated_ids, tokenizer)

    # Extract only the new tokens (remove the original prompt)
    new_tokens = generated_text[len(prompt) :]

    print(f"Generated text:")
    print(f"{prompt}{new_tokens}")
    print("-" * 50)
    print(f"Total tokens generated: {len(tokenizer.encode(new_tokens))}")

    return generated_text


#####################################
# Main Training Function
#####################################


def main():
    parser = argparse.ArgumentParser(
        description="Train GPT model with resumable training"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for AdamW optimizer (default: 5e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Micro batch size for training (default: 2)",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=32,
        help="Effective batch size via gradient accumulation (default: 32)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="the-verdict.txt",
        help="Text file to train on (default: the-verdict.txt)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="Log train/val loss every N steps (default: 25)",
    )
    parser.add_argument(
        "--start-context",
        type=str,
        default="Who are you?",
        help="Start context for text generation during training (default: 'Who are you?')",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for text generation sampling (default: 0.8)",
    )
    parser.add_argument(
        "--load-pretrained",
        type=str,
        default=None,
        choices=[
            "gpt2-small (124M)",
            "gpt2-medium (355M)",
            "gpt2-large (774M)",
            "gpt2-xl (1558M)",
        ],
        help="Load pretrained GPT-2 weights (default: None)",
    )
    parser.add_argument(
        "--pretrained-file",
        type=str,
        default=None,
        help="Custom path to pretrained weights file (default: None)",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run in inference mode instead of training",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Prompt for inference mode (default: 'The future of artificial intelligence is')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate in inference mode (default: 50)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Ratio of total training steps for learning rate warmup (default: 0.1 = 10%)",
    )
    parser.add_argument(
        "--initial-lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for warmup phase (default: 1e-4)",
    )

    args = parser.parse_args()

    # Calculate gradient accumulation steps
    accumulation_steps = max(1, args.effective_batch_size // args.batch_size)
    effective_batch_size = args.batch_size * accumulation_steps

    print("=== GPT Training Script ===")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Micro batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Save every: {args.save_every} epochs")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Start context: '{args.start_context}'")
    print(f"Log interval: {args.log_interval}")
    print(f"Temperature: {args.temperature}")

    # Warning for high learning rates
    if args.lr > 1e-3:
        print(
            f"⚠️  WARNING: Learning rate {args.lr} is quite high! Consider using 5e-4 or lower."
        )

    # Temperature advice
    if args.temperature < 0.1:
        print(f"⚠️  Temperature {args.temperature} is very low - text may be repetitive")
    elif args.temperature > 1.5:
        print(
            f"⚠️  Temperature {args.temperature} is very high - text may be incoherent"
        )

    # Get device
    device = get_device()

    # Load text data
    print("\nLoading text data...")
    text_data = load_text_data(args.data_file)
    print(f"Loaded {len(text_data):,} characters")

    # Split data
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # Create data loaders
    print("Creating data loaders...")
    torch.manual_seed(123)

    # Use consistent context length
    context_length = 256

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=args.batch_size,
        max_length=context_length,
        stride=context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=args.batch_size,
        max_length=context_length,
        stride=context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Initialize model
    print("\nInitializing model...")
    torch.manual_seed(123)

    # Choose configuration based on pretrained model if specified
    if args.load_pretrained:
        config = BASE_CONFIG.copy()
        config.update(model_configs[args.load_pretrained])
        config["context_length"] = 256  # Keep shorter context for training
        config["drop_rate"] = 0.1  # Add dropout for fine-tuning
        config["qkv_bias"] = True  # Match pretrained weights
        print(f"Using {args.load_pretrained} configuration")
    else:
        config = GPT_CONFIG_124M
        print("Using default configuration")

    model = GPTModel(config)

    # Load pretrained weights if specified
    if args.load_pretrained:
        print(f"Loading pretrained weights: {args.load_pretrained}")
        # we download gpt2 weight and assign to var
        pretrained_weights = download_and_load_gpt2_weights(
            args.load_pretrained, args.pretrained_file
        )
        # model and weight there
        load_weights_into_gpt(model, pretrained_weights)
        print("Pretrained weights loaded successfully!")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # If inference mode, run inference and exit
    if args.inference:
        if not args.load_pretrained:
            print(
                "⚠️  Warning: Running inference without pretrained weights. Results may be poor."
            )

        context_size = config.get("context_length", 256)
        run_inference(
            model=model,
            device=device,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            context_size=context_size,
        )
        return  # Exit after inference

    # Initialize optimizer (only needed for training)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        loaded_model, loaded_optimizer, start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, str(device)
        )
        if loaded_model is not None:
            model = loaded_model
            optimizer = loaded_optimizer
        else:
            print("Failed to load checkpoint. Starting from scratch.")
            start_epoch = 0
            best_val_loss = float("inf")

    # train loader return batch, because input batch and target batch
    # epoch * batch (N step) = total step
    total_steps = len(train_loader) * args.epochs
    # warm up step = some of total step
    warmup_steps = int(args.warmup_ratio * total_steps)
    # min learning rate and max learning rate
    peak_lr = args.lr
    initial_lr = args.initial_lr
    # learning rate inc per warm up step = (peak - init) // warm up step
    lr_increment = (peak_lr - initial_lr) / warmup_steps if warmup_steps > 0 else 0

    print(f"Learning rate schedule:")
    print(f"  Initial LR: {initial_lr}")
    print(f"  Peak LR: {peak_lr}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps} ({args.warmup_ratio*100:.1f}%)")
    print(f"  LR increment per step: {lr_increment:.8f}")

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    print("=" * 60)

    # global step start zero
    global_step = 0
    tokenizer = tiktoken.get_encoding("gpt2")

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            epoch + 1,
            global_step,
            accumulation_steps,
            args.log_interval,
            args.start_context,
            args.temperature,
            warmup_steps,
            initial_lr,
            peak_lr,
            lr_increment,
        )

        # Evaluate
        val_loss = evaluate(model, val_loader, device)

        # Print epoch summary in requested format
        print(
            f"ep {epoch + 1} (step {global_step}): train loss {train_loss:.4f}, val loss {val_loss:.4f}"
        )

        # Generate sample text to verify model performance
        model.eval()
        token_ids = text_to_token_ids(args.start_context, tokenizer).to(device)

        with torch.no_grad():
            generated_ids = generate_text_simple(
                model=model,
                idx=token_ids,
                max_new_tokens=15,
                context_size=GPT_CONFIG_124M["context_length"],
                temperature=args.temperature,
            )

        generated_text = token_ids_to_text(generated_ids, tokenizer)
        print(f"End of epoch sample: {generated_text}")
        print("-" * 60)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                model, optimizer, epoch, val_loss, args.lr, args.checkpoint_dir
            )

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save final model
    final_checkpoint = save_checkpoint(
        model, optimizer, args.epochs - 1, best_val_loss, args.lr, args.checkpoint_dir
    )
    print(f"Final model saved to: {final_checkpoint}")


if __name__ == "__main__":
    main()
