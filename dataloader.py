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

# trip spaces
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


# basically the dict form first, then we can use tokenizer
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)


# tokenizer
class SimpleTokenizerV1:
    def __init__(self, vocab):
        # form dict
        self.str_to_int = {token: integer for integer, token in enumerate(vocab)}
        # form dict
        self.int_to_str = {integer: token for integer, token in enumerate(vocab)}

    # encode text into token id
    def encode(self, text):
        # we decompose input text
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        # no white space for input text
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # now form the id
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # decode token id back to text
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


tokenizer = SimpleTokenizerV1(all_words)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
# [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]
ids = tokenizer.encode(text)
print(ids)
