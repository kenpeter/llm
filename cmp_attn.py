import torch

# seed 123
torch.manual_seed(123)
# device cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")

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