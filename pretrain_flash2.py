#!/usr/bin/env python3

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
import tiktoken
from datetime import datetime
from datasets import load_dataset
import random
import math


#####################################
# Dataset and DataLoader
#####################################


# python pretrain_flash2.py --batch-size 4 --effective-batch-size 256 --resume checkpoints/latest_checkpoint.pt --peak-lr 8e-4


# text to token id OR token id to text
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(
            txt, allowed_special={"<|endoftext|>"}, disallowed_special=()
        )

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


# open web text dataset
class OpenWebTextDataset(IterableDataset):
    """Streaming dataset for large-scale web text data (FineWeb, C4, RedPajama)"""

    # init
    # self, tokenizer, max_len, stride, buffer size 1000, skip_samples for starting position
    def __init__(
        self, tokenizer, max_length, stride=None, buffer_size=1000, skip_samples=0
    ):
        # tokenizer
        self.tokenizer = tokenizer
        # max len
        self.max_length = max_length
        # how much forward
        self.stride = stride if stride is not None else max_length
        # buff size for token
        self.buffer_size = buffer_size
        # randome skip sample, so start from diff position
        self.skip_samples = skip_samples

        # load 1 of three
        print("Loading large-scale web text dataset...")
        if skip_samples > 0:
            print(
                f"Will skip first {skip_samples} samples to start from different position"
            )
        dataset_loaded = False

        # Try different datasets in order of preference
        datasets_to_try = [
            ("HuggingFaceFW/fineweb", "train", None),  # FineWeb
            ("allenai/c4", "train", "en"),  # C4 English
            ("EleutherAI/pile", "train", None),  # The Pile
        ]

        # load data set
        for dataset_name, split_name, config_name in datasets_to_try:
            try:
                print(f"Trying {dataset_name} dataset...")
                if config_name:
                    # load data set from lib
                    self.dataset = load_dataset(
                        # dataset name: fine web
                        # config name: en english
                        # split name: training, test, val
                        # stream true
                        dataset_name,
                        config_name,
                        split=split_name,
                        streaming=True,
                    )
                else:
                    # no config name just load data set
                    self.dataset = load_dataset(
                        dataset_name, split=split_name, streaming=True
                    )
                print(f"✅ Successfully loaded {dataset_name}")
                dataset_loaded = True
                break
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")
                continue

        if not dataset_loaded:
            raise RuntimeError("Failed to load any large-scale web text dataset")

        print("✅ Large-scale web text streaming dataset loaded successfully")

    # iter the dataset
    def __iter__(self):
        # buffer to acc document
        token_buffer = []
        # track skipped samples
        skipped_count = 0

        # common text field name in data set
        text_fields = ["text", "content", "raw_content"]

        for example in self.dataset:
            # find the text field name, loop them
            text = None
            for field in text_fields:
                if field in example:
                    # find the right text
                    text = example[field]
                    break

            if text is None:
                print(
                    f"Warning: No text field found in example: {list(example.keys())}"
                )
                continue

            # Skip empty texts
            if not text or len(text.strip()) == 0:
                continue

            # Skip samples to start from different position
            if skipped_count < self.skip_samples:
                skipped_count += 1
                continue

            try:
                # tokenize the text, become one seq token
                tokens = self.tokenizer.encode(
                    text, allowed_special={"<|endoftext|>"}, disallowed_special=()
                )

                # add end of text between documents
                tokens.append(
                    self.tokenizer.encode(
                        "<|endoftext|>",
                        allowed_special={"<|endoftext|>"},
                        disallowed_special=(),
                    )[0]
                )

                # 1 doc in buffer
                token_buffer.extend(tokens)

                # clide win the buffer
                while len(token_buffer) >= self.max_length + 1:
                    # Create input and target sequences
                    input_chunk = token_buffer[: self.max_length]
                    target_chunk = token_buffer[1 : self.max_length + 1]

                    # this is generator, next will pull out the data
                    yield torch.tensor(input_chunk), torch.tensor(target_chunk)

                    # move buffer forward by stride
                    token_buffer = token_buffer[self.stride :]

            except Exception as e:
                # Reduce verbosity - only print first few errors
                if not hasattr(self, "_error_count"):
                    self._error_count = 0
                if self._error_count < 3:
                    print(f"Warning: Error processing text sample: {e}")
                    self._error_count += 1
                elif self._error_count == 3:
                    print("... (suppressing further tokenization warnings)")
                    self._error_count += 1
                continue


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


def create_openwebtext_dataloader(
    # batch size 4
    batch_size=4,
    # max len 256
    max_length=256,
    # stride 128 slide win
    stride=128,
    # zero worker
    num_workers=0,
    # this is token buffer
    buffer_size=1000,
    # skip samples to start from different position
    skip_samples=0,
):
    # tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # create stream dataset
    dataset = OpenWebTextDataset(
        # tokenizer
        tokenizer=tokenizer,
        # max len 256; the whole sentense can process
        max_length=max_length,
        # stride means how much go forward, no repeat
        stride=stride,
        # buff size
        buffer_size=buffer_size,
        # skip samples to start from different position
        skip_samples=skip_samples,
    )

    # Create dataloader for streaming dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # Note: shuffle not applicable for streaming
    )

    return dataloader


#####################################
# Model Components
#####################################


try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
    print("✅ Flash Attention 2 available")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("⚠️  Flash Attention 2 not available, falling back to PyTorch SDPA")


# flash attention
class FlashAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        # super
        super().__init__()
        # Ensure output dimension is evenly divisible by number of heads
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        # store attn param
        # total output dim
        self.d_out = d_out
        # number of attn head
        self.num_heads = num_heads
        # head dim
        self.head_dim = d_out // num_heads
        # drop prob
        self.dropout = dropout

        # qkv projection in one linear layer for efficiency (instead of 3 separate)
        self.qkv_proj = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        # output projection to combine multi-head results (via y = ax+b)
        self.out_proj = nn.Linear(d_out, d_out)

        # dropout layer for regularization
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # flash attn there
        global FLASH_ATTN_AVAILABLE
        # x: [b, seq_len, embed_dim] -> [b, 512, 896]
        batch_size, seq_len, embed_dim = x.shape

        # project x into qkv, in a single operation
        # (b, seq_len, 3 * d_out)
        qkv = self.qkv_proj(x)

        # [b, seq_len, 3, 14, 64]
        # head_n * head_dim = 896
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        # new shape: (3, batch_size, seq_len, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)
        # q, k, v, start at random, then later converted
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Try to use Flash Attention 2 if available
        if FLASH_ATTN_AVAILABLE:
            # flash attn 2 expects (batch, seq_len, num_heads, head_dim)
            # this layer works the best, as batch, len, then head_n, head_dim
            try:
                # flash aten works best with fp16/bf16 precision, why? because middle ground
                if x.dtype in [torch.float32]:
                    # fp32 to fp16 best
                    q, k, v = q.half(), k.half(), v.half()
                    use_fp16 = True
                else:
                    use_fp16 = False

                # attn2 with qkv
                attn_output = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=(
                        self.dropout if self.training else 0.0
                    ),  # Apply dropout only during training
                    causal=True,  # Causal mask for autoregressive generation (GPT-style)
                    softmax_scale=1.0
                    / (self.head_dim**0.5),  # Scale factor for attention scores
                )

                # Convert back to original dtype if we used fp16
                if use_fp16:
                    attn_output = attn_output.float()

            except Exception as e:
                # If Flash Attention fails, disable it and fall back to SDPA
                print(f"Flash Attention failed, falling back to SDPA: {e}")
                FLASH_ATTN_AVAILABLE = False
                # Fallback to PyTorch SDPA with proper tensor reshaping
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),  # Transpose for SDPA format
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                ).transpose(
                    1, 2
                )  # Transpose back to original format

        # Use PyTorch's Scaled Dot Product Attention (SDPA) as fallback
        if not FLASH_ATTN_AVAILABLE:
            # SDPA (scaled dot product attention) expects format: (batch, num_heads, seq_len, head_dim)
            q = q.transpose(1, 2)  # Swap seq_len and num_heads dimensions
            k = k.transpose(1, 2)  # Swap seq_len and num_heads dimensions
            v = v.transpose(1, 2)  # Swap seq_len and num_heads dimensions

            # use scaled dot product attn
            # scaled dot product attn (pytorch, more mem) VS flash attn 2 (NVDA, smaller block, tile, less mem)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=(
                    self.dropout if self.training else 0.0
                ),  # Dropout during training only
                is_causal=True,  # Causal mask prevents looking at future tokens
            )

            # Transpose back to expected format: (batch, seq_len, num_heads, head_dim)
            attn_output = attn_output.transpose(1, 2)

        # Reshape attention output to combine all heads
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_out)
        # Apply final linear projection to mix information across heads
        output = self.out_proj(attn_output)

        # Apply dropout for regularization
        output = self.dropout_layer(output)

        return output


# layer norm
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


# GELU
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
        # Use Flash Attention instead of MultiHeadAttention
        self.att = FlashAttention(
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

        # normalized beofre final
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

GPT_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 512,  # Moderate context (2x original)
    "emb_dim": 896,  # Smaller embedding (~150M parameters)
    "n_heads": 14,  # More attention heads (divisible by 896)
    "n_layers": 18,  # Deeper than original
    "drop_rate": 0.05,  # Lower dropout for faster convergence
    "qkv_bias": False,  # Query-key-value bias
}


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


# text to token id
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(
        text, allowed_special={"<|endoftext|>"}, disallowed_special=()
    )
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


# token to text
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


# val loss === cross entropy loss
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
# Learning Rate Scheduler
#####################################


class CosineDecayWithWarmup:
    def __init__(
        self, optimizer, warmup_steps, total_steps, peak_lr, min_lr, initial_lr
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.initial_lr = initial_lr
        self.current_step = 0

    def step(self):
        # interesting, slow -> fast -> slow
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_increment = (self.peak_lr - self.initial_lr) / self.warmup_steps
            lr = self.initial_lr + self.current_step * lr_increment
        else:
            # Cosine annealing after warmup
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        # Apply the calculated learning rate to all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.current_step += 1
        return lr

    def get_last_lr(self):
        return [param_group["lr"] for param_group in self.optimizer.param_groups]


#####################################
# Training Infrastructure
#####################################


# get device
def get_device(low_power_mode=False):
    """
    Detect and return the best available device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        # Clear cache and set memory fraction
        torch.cuda.empty_cache()
        
        if low_power_mode:
            # Conservative memory usage for cooler operation
            torch.cuda.set_per_process_memory_fraction(0.6)  # Use 60% of GPU memory
            print("🔋 Low power mode: Using 60% GPU memory for cooler operation")
        else:
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
            
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB"
        )
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# big dataset use streaming
def load_openwebtext_streaming():
    """
    Load large-scale web text dataset in streaming mode for training
    """
    print("🔄 Initializing large-scale web text streaming dataset...")
    print("This will download data on-demand during training.")

    # Try different datasets in order of preference
    datasets_to_try = [
        ("HuggingFaceFW/fineweb", "train", None),  # FineWeb
        ("allenai/c4", "train", "en"),  # C4 English
        ("EleutherAI/pile", "train", None),  # The Pile
    ]

    for dataset_name, split_name, config_name in datasets_to_try:
        try:
            print(f"Testing access to {dataset_name}...")
            if config_name:
                dataset = load_dataset(
                    dataset_name, config_name, split=split_name, streaming=True
                )
            else:
                dataset = load_dataset(dataset_name, split=split_name, streaming=True)
            # Just verify we can access it
            next(iter(dataset))
            print(f"✅ {dataset_name} dataset ready for training")
            return True

        except Exception as e:
            print(f"Failed to access {dataset_name}: {e}")
            continue

    print("❌ Failed to access any large-scale web text dataset.")
    print("Please ensure you have internet connection and datasets library installed.")
    return False


# save check point
def save_checkpoint(
    model, optimizer, epoch, loss, lr, model_config, checkpoint_dir="checkpoints"
):
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
        "config": model_config,
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


# load checkpoint
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


# single epoch
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
    max_batches=None,
    grad_clip=1.0,
    lr_scheduler=None,
    scaler=None,
):
    """
    Train model for one epoch with gradient accumulation and periodic validation
    Supports streaming datasets with optional batch limit
    """
    model.train()
    total_loss = 0.0
    num_batches = max_batches if max_batches else float("inf")  # Handle streaming
    optimizer.zero_grad()
    batch_idx = 0  # Initialize batch_idx
    last_grad_norm = 0.0  # Track gradient norm for logging

    # in single epoch, we have input batch and target batch
    for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        # Use mixed precision if enabled
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_batch)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), target_batch.flatten()
                )
            # Scale loss by accumulation steps for backward pass
            scaled_loss = loss / accumulation_steps
            # cal gradient val with scaler
            scaler.scale(scaled_loss).backward()
        else:
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), target_batch.flatten()
            )
            # Scale loss by accumulation steps for backward pass
            scaled_loss = loss / accumulation_steps
            # cal gradient val
            scaled_loss.backward()

        # so the loss just acc
        total_loss += loss.item()

        # Update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0 or (
            max_batches and batch_idx == max_batches - 1
        ):
            # Handle gradient clipping and optimization with mixed precision
            if scaler is not None:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                # Get gradient norm before clipping
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float("inf")
                )
                last_grad_norm = grad_norm_before.item()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                # Update with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Get gradient norm before clipping for monitoring
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float("inf")
                )
                last_grad_norm = grad_norm_before.item()  # Store for logging
                # gradient clip to prevent too much
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                # update gradient
                optimizer.step()
            
            # update learning rate if scheduler is provided
            if lr_scheduler is not None:
                current_lr = lr_scheduler.step()
            # now clean them up
            optimizer.zero_grad()
            # global step  done
            global_step += 1

            # at each log interval, we need to print out train loss and val loss
            if global_step % log_interval == 0:
                # total loss // batch id = single train loss
                current_train_loss = total_loss / (batch_idx + 1)

                # model enter eval mode
                model.eval()
                # val loss zero
                val_loss = 0.0
                # val sample
                val_samples = 0
                # no grad update
                with torch.no_grad():
                    # loop val loader, so we get batch idx, input and target in val scope
                    for val_batch_idx, (val_input, val_target) in enumerate(val_loader):
                        # only use first 5 batches
                        if val_batch_idx >= 5:  # Only use first 5 batches for speed
                            break

                        # ok, so basically, val
                        val_input, val_target = val_input.to(device), val_target.to(
                            device
                        )

                        # logit, cross entropy
                        val_logits = model(val_input)
                        val_batch_loss = torch.nn.functional.cross_entropy(
                            val_logits.flatten(0, 1), val_target.flatten()
                        )
                        val_loss += val_batch_loss.item()
                        val_samples += 1

                val_loss = val_loss / val_samples if val_samples > 0 else float("inf")

                # gen some text to show progress
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

                # Get current learning rate
                current_lr = (
                    lr_scheduler.get_last_lr()[0]
                    if lr_scheduler
                    else optimizer.param_groups[0]["lr"]
                )

                print(
                    f"ep {epoch} (step {global_step}): train loss {current_train_loss:.4f}, val loss {val_loss:.4f}, lr {current_lr:.2e}, grad_norm {last_grad_norm:.4f} | {start_context}{new_tokens}"
                )

        # Break if we've reached max_batches for streaming datasets
        if max_batches and batch_idx + 1 >= max_batches:
            break

    # Handle case where no batches were processed
    actual_batches = batch_idx + 1 if batch_idx >= 0 else 0
    if max_batches:
        actual_batches = min(actual_batches, max_batches)

    return total_loss / actual_batches if actual_batches > 0 else 0.0, global_step


def evaluate(model, val_loader, device, max_val_batches=100):
    """
    Evaluate model on validation set (limited batches for streaming)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(val_loader):
            if batch_idx >= max_val_batches:
                break

            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), target_batch.flatten()
            )
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float("inf")


#####################################
# Inference Function
#####################################


def run_inference(args, model_config=None):
    """
    Run text generation inference with trained model
    """
    print("=== GPT Inference with Flash Attention 2 ===")
    print(f"Model path: {args.model_path}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")

    if not FLASH_ATTN_AVAILABLE:
        print("📦 Flash Attention 2 not available, using PyTorch SDPA fallback")

    # Get device
    device = get_device()

    # Initialize model
    print("\nInitializing model...")
    if model_config is None:
        model_config = GPT_CONFIG  # fallback
    model = GPTModel(model_config)
    model = model.to(device)

    # Load checkpoint
    if not os.path.exists(args.model_path):
        print(f"❌ Model checkpoint not found: {args.model_path}")
        print("Please train a model first or specify correct --model-path")
        return

    loaded_model, _, _, _ = load_checkpoint(args.model_path, model, device=str(device))
    if loaded_model is None:
        print("❌ Failed to load model checkpoint")
        return

    model = loaded_model
    model.eval()

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"\n✅ Model loaded successfully!")
    print("=" * 50)

    if args.interactive:
        # Interactive mode
        print("🔄 Interactive mode - Type 'quit' to exit")
        while True:
            try:
                prompt = input("\n📝 Enter prompt: ").strip()
                if prompt.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                if not prompt:
                    continue

                # Generate text
                print(
                    f"🤖 Generating (temp={args.temperature}, max_tokens={args.max_tokens})..."
                )

                token_ids = text_to_token_ids(prompt, tokenizer).to(device)

                with torch.no_grad():
                    generated_ids = generate_text_simple(
                        model=model,
                        idx=token_ids,
                        max_new_tokens=args.max_tokens,
                        context_size=model_config["context_length"],
                        temperature=args.temperature,
                    )

                generated_text = token_ids_to_text(generated_ids, tokenizer)
                print(f"📖 Generated: {generated_text}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Generation error: {e}")

    else:
        # Single prompt mode
        print(f"📝 Input prompt: {args.prompt}")
        print(
            f"🤖 Generating (temp={args.temperature}, max_tokens={args.max_tokens})..."
        )

        try:
            token_ids = text_to_token_ids(args.prompt, tokenizer).to(device)

            with torch.no_grad():
                generated_ids = generate_text_simple(
                    model=model,
                    idx=token_ids,
                    max_new_tokens=args.max_tokens,
                    context_size=model_config["context_length"],
                    temperature=args.temperature,
                )

            generated_text = token_ids_to_text(generated_ids, tokenizer)
            print(f"📖 Generated text:")
            print("-" * 50)
            print(generated_text)
            print("-" * 50)

        except Exception as e:
            print(f"❌ Generation error: {e}")


#####################################
# Main Training Function
#####################################


def main():
    parser = argparse.ArgumentParser(
        description="Train or run inference with GPT model using Flash Attention 2"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Mode: 'train' for training, 'inference' for text generation (default: train)",
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
        default=200,
        help="Total number of training epochs (default: 200)",
    )
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=None,
        help="Additional epochs to train beyond current checkpoint (overrides --epochs when resuming)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for AdamW optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--peak-lr",
        type=float,
        default=5e-4,
        help="Peak learning rate after warmup (default: 5e-4)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine decay (default: 1e-6)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Number of warmup steps for learning rate (default: 2000)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (default: 1.0)",
    )
    parser.add_argument(
        "--low-power",
        action="store_true",
        help="Enable low power mode for cooler GPU operation",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training (faster, less memory)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Micro batch size for training (default: 2)",
    )
    # batch-size = 4 (each time fit to GPU), then effective batch size = 256 (total), 256/4 = 64 steps
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=128,
        help="Effective batch size via gradient accumulation (default: 128)",
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
        "--batches-per-epoch",
        type=int,
        default=50000,
        help="Number of batches per epoch for streaming dataset (default: 50000)",
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
        "--early-stopping-patience",
        type=int,
        default=50,
        help="Stop training if validation loss doesn't improve for N epochs (default: 50)",
    )

    # Inference-specific arguments
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/latest_checkpoint.pt",
        help="Path to trained model checkpoint for inference (default: checkpoints/latest_checkpoint.pt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Who are you?",
        help="Input prompt for text generation (default: 'Who are you?')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for continuous text generation",
    )

    args = parser.parse_args()

    # Use model configuration
    model_config = GPT_CONFIG

    # Route to appropriate function based on mode
    if args.mode == "inference":
        run_inference(args, model_config)
        return

    # Training mode - Calculate gradient accumulation steps
    accumulation_steps = max(1, args.effective_batch_size // args.batch_size)
    effective_batch_size = args.batch_size * accumulation_steps

    print("=== GPT Training Script with Flash Attention 2 ===")
    if args.additional_epochs is not None and args.resume:
        print(f"Additional epochs: +{args.additional_epochs}")
    else:
        print(f"Target epochs: {args.epochs}")
    print(f"Initial learning rate: {args.lr}")
    print(f"Peak learning rate: {args.peak_lr}")
    print(f"Min learning rate: {args.min_lr}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Gradient clipping: {args.grad_clip}")
    if args.low_power:
        print("🔋 Low power mode enabled")
    if args.mixed_precision:
        print("⚡ Mixed precision enabled")
    print(f"Micro batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Batches per epoch: {args.batches_per_epoch}")
    print(f"Save every: {args.save_every} epochs")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Start context: '{args.start_context}'")
    print(f"Log interval: {args.log_interval}")
    print(f"Temperature: {args.temperature}")
    print("📊 Training on large-scale web text dataset with streaming")

    if not FLASH_ATTN_AVAILABLE:
        print("\n📦 To install Flash Attention 2:")
        print("   pip install flash-attn --no-build-isolation")
        print("   (Requires CUDA and compatible PyTorch version)")
        print("   Current fallback: PyTorch Scaled Dot Product Attention (SDPA)")

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

    # Get device with power management
    device = get_device(low_power_mode=args.low_power)

    # Load large-scale web text streaming dataset
    print("\n🚀 Setting up large-scale web text streaming dataset...")

    if not load_openwebtext_streaming():
        print("❌ Failed to initialize large-scale web text dataset. Exiting.")
        return

    # Create streaming data loaders
    print("Creating streaming data loaders...")
    torch.manual_seed(123)

    # train from beginning of dataset
    random_skip = 0
    print(f"📚 Training will start from beginning: skipping {random_skip} samples")

    # Create main training dataloader (streams entire dataset)
    train_loader = create_openwebtext_dataloader(
        batch_size=args.batch_size,
        max_length=model_config["context_length"],
        stride=model_config["context_length"],
        num_workers=0,
        skip_samples=random_skip,
    )

    # Create validation dataloader from end of dataset for held-out data
    val_random_skip = 1000000  # Skip 1M samples to get end portion
    print(
        f"📖 Validation will start from end portion: skipping {val_random_skip} samples"
    )
    val_loader = create_openwebtext_dataloader(
        batch_size=args.batch_size,
        max_length=model_config["context_length"],
        stride=model_config["context_length"],
        num_workers=0,
        skip_samples=val_random_skip,
    )

    print("✅ Streaming dataloaders created")
    print(
        "📊 Note: Batch count unknown for streaming dataset (will process large-scale web text data)"
    )

    # Initialize model
    print("\nInitializing model...")
    torch.manual_seed(123)
    model = GPTModel(model_config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Initialize optimizer with better settings for large model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,  # Lower weight decay for larger model
        betas=(0.9, 0.95),  # Better beta values for language modeling
        eps=1e-8,  # Stable epsilon for mixed precision
    )
    
    # Initialize mixed precision scaler if enabled
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    if scaler:
        print("✅ Mixed precision training enabled for efficiency")

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

            # Handle additional epochs parameter
            if args.additional_epochs is not None:
                target_epochs = start_epoch + args.additional_epochs
                print(
                    f"📈 Training {args.additional_epochs} additional epochs: {start_epoch} → {target_epochs}"
                )
            else:
                target_epochs = args.epochs
                print(
                    f"📈 Training until epoch {target_epochs} (current: {start_epoch})"
                )
        else:
            print("Failed to load checkpoint. Starting from scratch.")
            start_epoch = 0
            best_val_loss = float("inf")
            target_epochs = args.epochs
    else:
        target_epochs = args.epochs

    # Calculate total training steps for cosine decay
    total_training_steps = target_epochs * args.batches_per_epoch // accumulation_steps

    # Add cosine decay with warmup learning rate scheduler
    lr_scheduler = CosineDecayWithWarmup(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_training_steps,
        peak_lr=args.peak_lr,
        min_lr=args.min_lr,
        initial_lr=args.lr,
    )

    print(f"\nLearning Rate Schedule:")
    print(f"  Total training steps: {total_training_steps:,}")
    print(
        f"  Warmup: {args.lr:.2e} → {args.peak_lr:.2e} over {args.warmup_steps} steps"
    )
    print(
        f"  Cosine decay: {args.peak_lr:.2e} → {args.min_lr:.2e} over remaining steps"
    )

    # Training loop with early stopping
    print(f"\nStarting training from epoch {start_epoch}...")
    print("=" * 60)

    global_step = 0
    tokenizer = tiktoken.get_encoding("gpt2")
    epochs_without_improvement = 0

    for epoch in range(start_epoch, target_epochs):
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
            args.batches_per_epoch,
            args.grad_clip,
            lr_scheduler,
            scaler,
        )

        # Evaluate
        val_loss = evaluate(model, val_loader, device)

        # Print epoch summary with current learning rate
        current_lr = lr_scheduler.get_last_lr()[0]
        print(
            f"ep {epoch + 1} (step {global_step}): train loss {train_loss:.4f}, val loss {val_loss:.4f}, lr {current_lr:.2e}"
        )

        # Generate sample text to verify model performance
        model.eval()
        token_ids = text_to_token_ids(args.start_context, tokenizer).to(device)

        with torch.no_grad():
            generated_ids = generate_text_simple(
                model=model,
                idx=token_ids,
                max_new_tokens=15,
                context_size=model_config["context_length"],
                temperature=args.temperature,
            )

        generated_text = token_ids_to_text(generated_ids, tokenizer)
        print(f"End of epoch sample: {generated_text}")
        print("-" * 60)

        # Early stopping and best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print(f"New best validation loss: {best_val_loss:.4f}")
            # Save best model immediately
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                args.lr,
                model_config,
                args.checkpoint_dir,
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement}/{args.early_stopping_patience} epochs"
            )

            # Early stopping check
            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"\n🛑 Early stopping! No improvement for {args.early_stopping_patience} epochs."
                )
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

        # Note: LR scheduler steps are handled in train_epoch per batch

        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                args.lr,
                model_config,
                args.checkpoint_dir,
            )

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save final model
    final_checkpoint = save_checkpoint(
        model,
        optimizer,
        target_epochs - 1,
        best_val_loss,
        args.lr,
        model_config,
        args.checkpoint_dir,
    )
    print(f"Final model saved to: {final_checkpoint}")


if __name__ == "__main__":
    main()
