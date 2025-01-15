import torch
from configs import ACTIVATIONS
from utils import attention



class CLIPEmbeddings(torch.nn.Module):
    def __init__(
        self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None
    ):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(
            vocab_size, embed_dim, dtype=dtype, device=device
        )
        self.position_embedding = torch.nn.Embedding(
            num_positions, embed_dim, dtype=dtype, device=device
        )

    def forward(self, input_tokens):
        return self.token_embedding(input_tokens) + self.position_embedding.weight


class CLIPEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        heads,
        intermediate_size,
        intermediate_activation,
        dtype,
        device,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                CLIPLayer(
                    embed_dim,
                    heads,
                    intermediate_size,
                    intermediate_activation,
                    dtype,
                    device,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, mask=None, intermediate_output=None):
        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output
        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device):
        super().__init__()
        self.heads = heads
        self.q_proj = torch.nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )
        self.k_proj = torch.nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )
        self.v_proj = torch.nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )
        self.out_proj = torch.nn.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = attention(q, k, v, self.heads, mask)
        return self.out_proj(out)


class Mlp(torch.nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=torch.nn.GELU,
        bias=True,
        dtype=None,
        device=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = torch.nn.Linear(
            in_features, hidden_features, bias=bias, dtype=dtype, device=device
        )
        self.act = act_layer
        self.fc2 = torch.nn.Linear(
            hidden_features, out_features, bias=bias, dtype=dtype, device=device
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CLIPLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        heads,
        intermediate_size,
        intermediate_activation,
        dtype,
        device,
    ):
        super().__init__()
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device)
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        # self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation, dtype, device)
        self.mlp = Mlp(
            embed_dim,
            intermediate_size,
            embed_dim,
            act_layer=ACTIVATIONS[intermediate_activation],
            dtype=dtype,
            device=device,
        )

    def forward(self, x, mask=None):
        x += self.self_attn(self.layer_norm1(x), mask)
        x += self.mlp(self.layer_norm2(x))
        return x