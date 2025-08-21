# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.

# import time, tokenizer, torch, nn
import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    # opt: max_seq_len (model can handle) and window_size (slide window for kv cache)
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, max_seq_len=None, window_size=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # head_n * head_dim = d_out
        self.head_dim = d_out // num_heads 

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        ####################################################
        # opt, has limited win size

        # max model can handle
        # 1024, 1024
        self.max_seq_len = max_seq_len or context_length

        # max win for slide win
        # 1024
        self.window_size = window_size or self.max_seq_len

        # CHANGE: Removed global mask registration - computed dynamically now
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        # CHANGE: No ptr_current_pos here - using local ptr_cur instead
        ####################################################

    def forward(self, x, use_cache=False):
        # x: (b, token_n, d_out)
        b, num_tokens, d_in = x.shape

        # (b, token_n, d_out)
        keys_new = self.W_key(x)  
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # (b, token_n, d_out) -> (b, token_n, head_n, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # opt: do early
        # (b, token_n, d_out) -> (b, token_n, head_n, head_dim) -> (b, head_n, token_n, head_dim)
        keys_new = keys_new.transpose(1, 2)
        values_new = values_new.transpose(1, 2)
        queries = queries.transpose(1, 2)

        ####################################################
        # opt: build the large slide win buffer first
        if use_cache:
            # ptr_cur for slide win
            # self.ptr_current_pos track global token position

            # if chacke not there or batch size change
            if self.cache_k is None or self.cache_k.size(0) != b:
                # token_n === win size
                # (b, token_n, d_out) -> (b, token_n, head_n, head_dim) -> (b, head_n, token_n, head_dim) -> cache format
                self.cache_k = torch.zeros(b, self.num_heads, self.window_size, self.head_dim, device=x.device)

                # zeros (fresh) vs zeros like (copy)
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptr_cur = 0  # Initialize pointer to track next free slot in cache

            # opt: slide win with buffer above
            # if curr + len > win size
            if self.ptr_cur + num_tokens > self.window_size:
                # cal how many tokens we need to discard
                overflow = self.ptr_cur + num_tokens - self.window_size
                
                # :-1 means start from last but not include last (because ind on right)
                # e.g. [A, B, C, D, E] -> add F -> [:, :, :-1_overflow, :] -> [:, :, 0:4, :] -> [:, :, 1_overflow:, :] -> [:, :, 1:5_copy, :]
                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()
                # slide win pointer pointer
                # because point points to the future, -overflow
                self.ptr_cur -= overflow

            # keys_new assign to 3 dim only, others kept
            self.cache_k[:, :, self.ptr_cur:self.ptr_cur + num_tokens, :] = keys_new
            self.cache_v[:, :, self.ptr_cur:self.ptr_cur + num_tokens, :] = values_new

            # because pointer is pt to the future.
            self.ptr_cur += num_tokens

            # only use active portion of tokens, avoid most zeros to attention
            # [:, :, :self.ptr_cur, :] === [:, :, self.ptr_cur:self.ptr_cur + num_tokens, :]
            keys = self.cache_k[:, :, :self.ptr_cur, :]
            values = self.cache_v[:, :, :self.ptr_cur, :]
        else:
            # No caching: use only current tokens (standard attention)
            keys, values = keys_new, values_new
            self.ptr_cur = 0  # Reset pointer when switching between cache/no-cache modes


        ####################################################
        
        # query token size diff from value token size
        # q.shape = [b, head_n, token_n_q, head_dim]
        # k.shape = [b, head_n, token_n_k, head_dim]
        # [b, head_n, token_n_q, head_dim] @ [b, head_n, token_n_k, head_dim].T(2, 3) -> [b, head_n, token_n_q, token_n_k]
        
        # queries: what we ask
        # keys: what we find out
        # at the end, we want answer, attn_scores is related to k
        attn_scores = queries @ keys.transpose(2, 3)

        ####################################################
        # opt:
        # query token size diff from value token size
        # [b, head_n, token_n_q, token_n_k] -> size(-1) -> token_n_k
        K = attn_scores.size(-1)

        # token_n is newly processed token
        # K = cached + new
        if num_tokens == K:
            # no cache
            # row: token_n (new token)
            # col: K (everything)
            causal_mask = torch.triu(torch.ones(num_tokens, K, device=x.device, dtype=torch.bool), diagonal=1)
        else:
            
            # we have local sequence (row: new token; col: everything) and global sequence (everything) concept
            offset = K - num_tokens 
            """
                row_idx -> unsqueeze(1) -> (token_n, 1) -> [[0], [1]]
                col_idx -> unsqueeze(0) -> (1, K) -> [[0, 1, 2, 3, 4]]

                row_idx:
                [
                    [0, 0, 0, 0, 0], 
                    [1, 1, 1, 1, 1]
                ]

                offset = 2

                row_idx + offset -> [[0], [1]] -> [[3], [4]]

                compare -> diff shape -> boardcasting

                row_idx + offset:
                [
                    [3, 3, 3, 3, 3],
                    [4, 4, 4, 4, 4]
                ]  

                col_idx:
                [
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4]
                ]

                now we can compare

                [3, 3, 3, 3, 3] < [0, 1, 2, 3, 4]
                = [False, False, False, False, True]

                [4, 4, 4, 4, 4] < [0, 1, 2, 3, 4]
                = [False, False, False, False, False]
            """
            row_idx = torch.arange(num_tokens, device=x.device).unsqueeze(1)  # unsqueeze at index 1, (num_tokens, 1)
            col_idx = torch.arange(K, device=x.device).unsqueeze(0)           # unsqueeze at index 0, (1, K)
            causal_mask = row_idx + offset < col_idx                          # True where j > i+offset
        ####################################################

        # attn_scores: [batch, heads, token_n, K]  -> (1, 12, 2, 5)
        # causal_mask: [token_n, K] -> (2, 5)
        # doulbe unsqueeze(0) -> (1, 1, 2, 5)
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

    ####################################################
    # SIMPLIFICATION: Cleaner cache reset (no ptr_current_pos to track)
    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
    ####################################################


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            # opt: has limited win size
            window_size=cfg["kv_window_size"] if "kv_window_size" in cfg else cfg["context_length"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        # NEW

        # run the attention
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        # CHANGE: ModuleList for iteration (vs Sequential)
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # KEPT: Global position tracking for positional embeddings
        self.ptr_current_pos = 0
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        # opt: Proper positional encoding for cached generation
        # Original had complex masking logic, this handles position more cleanly
        if use_cache:
            # use cache
            # position arr
            # seq_len is current input len
            # win_size is limit
            pos_ids = torch.arange(self.ptr_current_pos, self.ptr_current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            # now we extend curr pos + seq_len
            self.ptr_current_pos += seq_len
        else:
            # no cache, entire postion arr, just from zero
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # CHANGE: Manual iteration for cache control (vs Sequential)
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    ####################################################
    # NEW
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.ptr_current_pos = 0
    ####################################################


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


####################################################
# NEW
def generate_text_simple_cached(model, idx, max_new_tokens, context_size=None, use_cache=True):
    # eval model
    model.eval()

    # context size; max seq len
    ctx_len = context_size or model.pos_emb.num_embeddings

    # no grade
    with torch.no_grad():
        # use cache
        if use_cache:
            # reset kv cache
            model.reset_kv_cache()
            # only get the range of token
            logits = model(idx[:, -ctx_len:], use_cache=True)

            # only able to output 200 max tokens
            for _ in range(max_new_tokens):
                # logits has the raw scores that possible output as next token
                # argmax will return the index, so next token id
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # stack new token next_idx with old ones
                idx = torch.cat([idx, next_idx], dim=1)
                # pass new token to model
                logits = model(next_idx, use_cache=True)
        else:
            # no cache very similar
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
####################################################


def main():
    # this is the config
    GPT_CONFIG_124M = {
        # vocab size 50257
        "vocab_size": 50257,
        # Context length 1024
        "context_length": 1024,  
        "emb_dim": 768,          # Embedding dimension
        # 12 heads
        "n_heads": 12,
        # 12 layers
        "n_layers": 12,
        # drop rate 0.1
        "drop_rate": 0.1,
        # qkv bias
        "qkv_bias": False,
        # op: kv win size 1-24
        "kv_window_size": 1024
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=encoded_tensor,
    #     max_new_tokens=200,
    #     context_size=GPT_CONFIG_124M["context_length"]
    # )

    ####################################################
    # NEW

    # gen text simple cache
    token_ids = generate_text_simple_cached(
        # model
        model=model,
        # token id
        idx=encoded_tensor,
        # max new token 200
        max_new_tokens=200,
    )
    ####################################################

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()