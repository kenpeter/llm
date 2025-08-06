import os
import urllib.request
import re
import importlib.metadata
import tiktoken

# get tokenizer from gpt2
tokenizer = tiktoken.get_encoding("gpt2")

# open the file read
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# encode the text
enc_text = tokenizer.encode(raw_text)
print(enc_text)
print(len(enc_text))
