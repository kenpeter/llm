import torch


# Suppose we have the following 3 training examples,
# which may represent token IDs in a LLM context
idx = torch.tensor([2, 3, 1])

# The number of rows in the embedding matrix can be determined
# by obtaining the largest token ID + 1.
# If the highest token ID is 3, then we want 4 rows, for the possible
# token IDs 0, 1, 2, 3
num_idx = max(idx) + 1  # 4 row

# The desired embedding dimension is a hyperparameter
out_dim = 5  # 5 col


# same
torch.manual_seed(123)

# 4x5 matrix
embedding = torch.nn.Embedding(num_idx, out_dim)

# select 2, 3, 1 index from 4x5 -> 3x5
out = embedding(torch.tensor([2, 3, 1]))

print("== embedding ==")
print(out)


# [2, 3, 1] -> [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]] 3x4   --> 4x5 (looking for)
# because onehot is int, matrix is float
onehot = torch.nn.functional.one_hot(idx)

# same
torch.manual_seed(123)
# linear, num_inx is col, out_dim is row --> swap --> 5x4
linear = torch.nn.Linear(num_idx, out_dim, bias=False)

# print out dim
print(linear.weight.shape)


# 5x4 -> 4x5
linear.weight = torch.nn.Parameter(embedding.weight.T)


# each ele become float
myfloat = onehot.float()


print("== onehot ==")
print(linear(myfloat))

# basically embedding is direct ind
# linear is for output = input * W.T + bias cal
