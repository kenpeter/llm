# GPT Training CLI

This script provides a comprehensive CLI for training GPT models with resumable training capabilities, automatic device detection, and periodic checkpointing.

## Features

- **Device Detection**: Automatically detects and uses the best available device (CUDA, MPS, or CPU)
- **Resumable Training**: Save and resume training from checkpoints
- **AdamW Optimizer**: Uses AdamW optimizer with configurable learning rate
- **Periodic Checkpointing**: Save model weights at regular intervals
- **Flexible Configuration**: Command-line arguments for all training parameters

## Usage

### Basic Training
```bash
python main.py --epochs 10 --lr 5e-4
```

### Resume Training
```bash
python main.py --resume checkpoints/latest_checkpoint.pt --epochs 20 --lr 1e-4
```

### Full Configuration
```bash
python main.py \
  --epochs 50 \
  --lr 3e-4 \
  --batch-size 8 \
  --save-every 5 \
  --checkpoint-dir models \
  --data-file my_text.txt
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--resume` | str | None | Path to checkpoint file to resume training from |
| `--epochs` | int | 10 | Number of training epochs |
| `--lr` | float | 5e-4 | Learning rate for AdamW optimizer |
| `--batch-size` | int | 4 | Batch size for training |
| `--save-every` | int | 5 | Save checkpoint every N epochs |
| `--checkpoint-dir` | str | checkpoints | Directory to save checkpoints |
| `--data-file` | str | the-verdict.txt | Text file to train on |

## Device Support

The script automatically detects and uses:
1. **CUDA** - If NVIDIA GPU is available
2. **MPS** - If Apple Silicon GPU is available  
3. **CPU** - As fallback

## Checkpoint Format

Checkpoints include:
- Model state dictionary
- Optimizer state dictionary
- Current epoch
- Training loss
- Learning rate
- Model configuration
- Timestamp

## Files Created

- `checkpoints/latest_checkpoint.pt` - Always contains the most recent checkpoint
- `checkpoints/checkpoint_epoch_XXXX.pt` - Epoch-specific checkpoints
- `the-verdict.txt` - Downloaded automatically if not present

## Example Training Session

```bash
# Start initial training
python main.py --epochs 10 --lr 1e-3 --save-every 2

# Resume and continue training with lower learning rate
python main.py --resume checkpoints/latest_checkpoint.pt --epochs 20 --lr 5e-4

# Fine-tune with even lower learning rate
python main.py --resume checkpoints/latest_checkpoint.pt --epochs 30 --lr 1e-4
```

## Model Architecture

- **Parameters**: ~162M parameters (GPT-2 124M configuration)
- **Context Length**: 256 tokens
- **Vocabulary**: 50,257 tokens (GPT-2 tokenizer)
- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **Layers**: 12

## Requirements

- Python 3.7+
- PyTorch 2.0+
- tiktoken
- CUDA toolkit (optional, for GPU acceleration)