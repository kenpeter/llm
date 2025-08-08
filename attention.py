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

# create empty attention score
# size 6
attn_scores_2 = torch.empty(inputs.shape[0])

# loop 6 times, as 6 rows
for i, x_i in enumerate(inputs):
    # the 2nd token attention to all 6 -> each attention score
    # attention score * x_i
    attn_scores_2[i] = torch.dot(
        x_i, query
    )  # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2)

# we normalized the all scores in single row, dim=0 row, dim=1, in 2D
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())


query = inputs[1]  # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    # if do a max, then only attention to this thing, other ele lost
    # this blend together
    context_vec_2 += attn_weights_2[i] * x_i


# print(context_vec_2)
