#!/usr/bin/env python3

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
from datetime import datetime
import urllib.request


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
# Utility Functions
#####################################


def generate_text_simple(model, idx, max_new_tokens, context_size):
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

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

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
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
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


def train_epoch(model, train_loader, optimizer, device, epoch, global_step):
    """
    Train model for one epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()

        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        # Print progress every 100 steps instead of every 10 batches
        if global_step % 100 == 0:
            print(f"ep {epoch} (step {global_step}): train loss {loss.item():.4f}")

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
        default=1024,
        help="Batch size for training (default: 4)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
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

    args = parser.parse_args()

    print("=== GPT Training Script ===")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save every: {args.save_every} epochs")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

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

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=args.batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=args.batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Initialize model
    print("\nInitializing model...")
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Initialize optimizer
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

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    print("=" * 60)
    
    global_step = 0
    tokenizer = tiktoken.get_encoding("gpt2")

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, global_step = train_epoch(model, train_loader, optimizer, device, epoch + 1, global_step)

        # Evaluate
        val_loss = evaluate(model, val_loader, device)

        # Print epoch summary in requested format
        print(f"ep {epoch + 1} (step {global_step}): train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        # Generate sample text to verify model performance
        model.eval()
        start_context = "Every effort moves"
        token_ids = text_to_token_ids(start_context, tokenizer).to(device)
        
        with torch.no_grad():
            generated_ids = generate_text_simple(
                model=model,
                idx=token_ids,
                max_new_tokens=10,
                context_size=GPT_CONFIG_124M["context_length"]
            )
        
        generated_text = token_ids_to_text(generated_ids, tokenizer)
        print(f"Generated sample: {generated_text}")
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
