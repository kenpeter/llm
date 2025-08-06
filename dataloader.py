import os
import urllib.request
import re

# dl
if not os.path.exists("the-verdict.txt"):
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        + "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        + "the-verdict.txt"
    )
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

# read file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# trip spaces, this will create extra spaces.
preprocessed = re.split(r'([,.:;?_!"()]|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


# basically the dict form first, then we can use tokenizer
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])


# tokenizer
class SimpleTokenizerV1:
    # vocab is split, set, sorted, big arr
    def __init__(self, vocab):
        # vocab form dict (key, val)
        self.str_to_int = {token: integer for integer, token in enumerate(vocab)}
        # vocab form dict (val, key)
        self.int_to_str = {integer: token for integer, token in enumerate(vocab)}

    def encode(self, text):
        # income text will split
        preprocessed = re.split(r'([,.:;?_!"()]|--|\s)', text)

        # income text no space
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        # the s both in str_to_int and s in preprocess
        ids = [self.str_to_int[s] for s in preprocessed]
        # return id in array
        return ids

    # decode token id back to text
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations

        # when we form the vocab, using split, it has extra space injected. this remove those spaces
        text = re.sub(r'\s+([,.?!"()])', r"\1", text)

        return text


tokenizer = SimpleTokenizerV1(all_tokens)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

ids = tokenizer.encode(text)
text = tokenizer.decode(ids)
print(text)
