import os
import urllib.request
import re
import importlib.metadata
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch

# get tokenizer from gpt2
tokenizer = tiktoken.get_encoding("gpt2")

# open the file read
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# gpt data set v1
# pass dataset
class GPTDatasetV1(Dataset):
    # def init
    # self, txt, tokenizer, max_len, stride
    def __init__(self, txt, tokenizer, max_length, stride):
        # input ids arr
        self.input_ids = []
        # target ids arr
        self.target_ids = []

        # we receive tokernizer from top, then encode txt to token ids
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        # e.g. max_len === 4
        # then at least token_ids = [1, 2, 3, 4, 5], to create 1 slide move [1, 2, 3, 4] -> 1 slide move -> [2, 3, 4, 5]
        assert (
            len(token_ids) > max_length
        ), "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # stride is like for loop i=i+2 jumping
        for i in range(0, len(token_ids) - max_length, stride):
            # slide win to predict next token
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # get input ids
    # both target and input ids have same len
    def __len__(self):
        return len(self.input_ids)

    # return both
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    # txt
    txt,
    # why call batch size (row), because finish one batch then another
    batch_size=4,
    # max len 256
    max_length=256,
    # stride 128
    stride=128,
    # make model learn pattern, not order
    shuffle=True,
    # if the last batch size is smaller, we don't want to train, because unstable
    drop_last=True,
    # zero worker, parallel worker
    num_workers=0,
):

    # init the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # slide win, so we have input arr, and target arr
    # max_len is the max arr length (4)
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # data loader use the internal methods from GPTDatasetV1
    dataloader = DataLoader(
        # dataset is input_ids and target_ids
        dataset,
        # batch size (8)
        batch_size=batch_size,
        # shuffle
        shuffle=shuffle,
        # drop last
        drop_last=drop_last,
        num_workers=num_workers,
    )

    # return data loader
    return dataloader


# max len control how many token in single arr
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)

# create iterator from data loader
data_iter = iter(dataloader)
# then next get first batch
first_batch = next(data_iter)
# print(first_batch)


# ============= position embedding ================
vocab_size = 50257
output_dim = 256

# 50257 x 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# print(token_embedding_layer)


max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# print("Token IDs:\n", inputs)
# print("\nInputs shape:\n", inputs.shape)

# inputs: 8x4
token_embeddings = token_embedding_layer(inputs)
# print(token_embeddings.shape)
# print(token_embeddings)


# ======= what are we doing ====

# context leng is max len 4
context_length = max_length
# 4 x 256
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
print(pos_embedding_layer)
print(pos_embedding_layer.weight)

# max len = 4
# ran tensor([0, 1, 2, 3])
position_ind = torch.arange(max_length)
print(position_ind)
pos_embeddings = pos_embedding_layer(position_ind)
print(pos_embeddings.shape)
print(pos_embeddings)
