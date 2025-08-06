import os
import urllib.request
import re
import importlib.metadata
import tiktoken
from torch.utils.data import Dataset, DataLoader

# get tokenizer from gpt2
tokenizer = tiktoken.get_encoding("gpt2")

# open the file read
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# encode the text
enc_text = tokenizer.encode(raw_text)


enc_sample = enc_text[50:]

context_size = 4


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert (
            len(token_ids) > max_length
        ), "Number of tokenized inputs must at least be equal to max_length+1"

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
