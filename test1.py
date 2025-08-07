import torch


# Suppose we have the following 3 training examples,
# which may represent token IDs in a LLM context
idx = torch.tensor([2, 3, 1])

# The number of rows in the embedding matrix can be determined
# by obtaining the largest token ID + 1.
# If the highest token ID is 3, then we want 4 rows, for the possible
# token IDs 0, 1, 2, 3
num_idx = max(idx)+1 # 4 row

# The desired embedding dimension is a hyperparameter
out_dim = 5 # 5 col




# We use the random seed for reproducibility since
# weights in the embedding layer are initialized with
# small random values
torch.manual_seed(123)

# row and col
embedding = torch.nn.Embedding(num_idx, out_dim)

# tensor is training data, feed to embed, will get vector
# tensor([[ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015]], grad_fn=<EmbeddingBackward0>)
# 1 row and 5 col
out = embedding(torch.tensor([0, 1, 2, 3]))

print(out)