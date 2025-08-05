
import os
import urllib.request
import re

# dl
if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/" + "LLMs-from-scratch/main/ch02/01_main-chapter-code/" + "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

# read file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# trip spaces
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]



# unique word, then sort
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

# vocab size
print(vocab_size)


# token key, integer value
vocab = {token:integer for integer,token in enumerate(all_words)}

# enum vocab and items
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break