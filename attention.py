import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],
    ]  # step     (x^6)
)


query = inputs[1]  # 2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0])

# 1. cal attention score for each row -> single element value for this row
for i, x_i in enumerate(inputs):
    # 1x3 dot 1x3 -> 1x3 dot 1x3.T -> 1x3 dot 3x1 -> 1 ele value for entire row
    attn_scores_2[i] = torch.dot(x_i, query)

# 2. normalize each single value (represent each row)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

query = inputs[1]

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    # 1x3 dot 1x3 -> 1x3 dot 1x3.T -> 1x3 dot 3x1 -> single ele for entire row
    context_vec_2 += attn_weights_2[i] * x_i

# context_vec_2 is 1x3
print(context_vec_2)


attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)
