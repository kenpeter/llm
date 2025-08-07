import tiktoken
from bpe_openai_gpt2 import get_encoder, download_vocab

# ======================== tiktoken =================
# get tokenizer
tik_tokenizer = tiktoken.get_encoding("gpt2")

# input text
text = "Hello, world. Is this-- a test?"

# text to token id
integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# print token id
# print(integers)

# token id back to words
strings = tik_tokenizer.decode(integers)

# Hello, world. Is this-- a test?
# print(strings)

# how many vocab in tiktoken
# print(tik_tokenizer.n_vocab)


# ======================== BPE openai =================

# download
# download_vocab()

# # openai tokenizer
# orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")

# integers = orig_tokenizer.encode(text)

# print(integers)

# strings = orig_tokenizer.decode(integers)

# print(strings)


# ================ hugging face BPE ====================

# from transfomer import gpt2 tokenizer
from transformers import GPT2Tokenizer

hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# need to use input_ids
print(hf_tokenizer(strings)["input_ids"])

# tokenizer fast
from transformers import GPT2TokenizerFast

hf_tokenizer_fast = GPT2TokenizerFast.from_pretrained("gpt2")
print(hf_tokenizer_fast(strings)["input_ids"])
