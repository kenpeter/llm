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
    # batch size 4
    batch_size=4,
    # max len 256
    max_length=256,
    # stride 128
    stride=128,
    # random
    shuffle=True,
    # drop last true
    drop_last=True,
    # zero worker
    num_workers=0,
):

    # init the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # slide win, so we have input arr, and target arr
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # # data loader use the internal methods from GPTDatasetV1
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    # return data loader
    return dataloader


# max len control how many token in single arr
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
