GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        # init
        super().__init__()
        # token_embed, [50257, 768]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # posi_embed, [1024, 768]
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # drop embed, 0.1
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # transformer block with 12 layers
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # final norm, 768
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        # out head, [768, 50257]
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):

        '''
            in_idx:
            tensor([[6109, 3626, 6100,  345],
                [6109, 1110, 6622,  257]])
        '''

        # batch size 2, token_n 4; 
        batch_size, seq_len = in_idx.shape
        # token embed; [2, 4, 768]
        tok_embeds = self.tok_emb(in_idx)

        # posi embed; [4, 768]; [token_n, embed_dim]
        # array-range: [0, 1, 2, 3]
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # combine token feature embed + posi embed
        # [2, 4, 768] + [4, 768] -> [2, 4, 768] + [1, 4, 768] = [2, 4, 768]
        x = tok_embeds + pos_embeds
        # drop rate
        x = self.drop_emb(x)
        # tranformer
        x = self.trf_blocks(x)
        # final norm
        x = self.final_norm(x)
        # output
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x

# ======  

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch size = 2
batch = torch.stack(batch, dim=0)


# ========

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)

logits = model(batch)




# =======

torch.manual_seed(123)

# create 2 training examples with 5 dimensions (features) each
batch_example = torch.randn(2, 5) 

layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)



class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        # because we normalize values, model need real value (scale)
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # because we normalize values, model need real value (shift)
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # mean
        mean = x.mean(dim=-1, keepdim=True)
        # variance
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # standard deviation
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)

# very close to zero
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)



# ======

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU()

# Some sample data
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
# plt.show()


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
    



ffn = FeedForward(GPT_CONFIG_124M)

# input shape: [batch_size, num_token, emb_size]
x = torch.rand(2, 3, 768) 
out = ffn(x)



# ===============


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        # super init
        super().__init__()
        # enable / disable shortcut
        self.use_shortcut = use_shortcut
        # Create 5 layers with GELU activation
        self.layers = nn.ModuleList([
            # [3, 3]
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),  # Layer 1: 3->3
            # [3, 3]
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),  # Layer 2: 3->3
            # [3, 3]
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),  # Layer 3: 3->3
            # [3, 3]
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),  # Layer 4: 3->3
            # [3, 1]
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())   # Layer 5: 3->1
        ])

    def forward(self, x):
        # Process each layer sequentially
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # 1. it is not completely skip still learn something
            # 2. x = x + layer_output, skip for only same shape
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                # 3. this means change shape to next layer
                x = layer_output
        return x


def print_gradients(model, x):
    # 1. pytorch model(x)
    # 2. tensorflow / scikit model(x).predict

    # this is model output
    output = model(x)
    # this is target
    target = torch.tensor([[0.]])

    # mean square error loss. how far to mean
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    # backward
    loss.backward()

    # Print gradient magnitudes for each weight matrix
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only examine weight parameters (skip biases)
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


# Network architecture: input_size=3, hidden_layers=4, output_size=1
layer_sizes = [3, 3, 3, 3, 3, 1]  


# (b, feature) -> (1, 3)
sample_input = torch.tensor([[1., 0., -1.]])

# Set random seed for reproducible results
torch.manual_seed(123)
# Create model without skip connections
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
# Analyze gradient flow in the network
print_gradients(model_without_shortcut, sample_input)





# ==============

# If the `previous_chapters.py` file is not available locally,
# you can import it from the `llms-from-scratch` PyPI package.
# For details, see: https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# E.g.,
# from llms_from_scratch.ch03 import MultiHeadAttention

from previous_chapters import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # different kind of attention
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x