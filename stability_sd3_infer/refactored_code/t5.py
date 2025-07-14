import math
import torch
import torch.nn as nn
from utils import attention

class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device)
        )
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(device=x.device, dtype=x.dtype) * x


class T5DenseGatedActDense(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.wi_0 = nn.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wi_1 = nn.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = nn.Linear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        hidden_gelu = torch.nn.functional.gelu(self.wi_0(x), approximate="tanh")
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        x = self.wo(x)
        return x


class T5LayerFF(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(model_dim, ff_dim, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        x += forwarded_states
        return x


class T5Attention(torch.nn.Module):
    def __init__(
        self, model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device
    ):
        super().__init__()
        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.k = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.v = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.o = nn.Linear(inner_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.num_heads = num_heads
        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = torch.nn.Embedding(
                self.relative_attention_num_buckets, self.num_heads, device=device
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, device):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self, x, past_bias=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], x.device)
        if past_bias is not None:
            mask = past_bias
        out = attention(
            q, k * ((k.shape[-1] / self.num_heads) ** 0.5), v, self.num_heads, mask
        )
        return self.o(out), past_bias


class T5LayerSelfAttention(torch.nn.Module):
    def __init__(
        self,
        model_dim,
        inner_dim,
        ff_dim,
        num_heads,
        relative_attention_bias,
        dtype,
        device,
    ):
        super().__init__()
        self.SelfAttention = T5Attention(
            model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device
        )
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x, past_bias=None):
        output, past_bias = self.SelfAttention(self.layer_norm(x), past_bias=past_bias)
        x += output
        return x, past_bias


class T5Block(torch.nn.Module):
    def __init__(
        self,
        model_dim,
        inner_dim,
        ff_dim,
        num_heads,
        relative_attention_bias,
        dtype,
        device,
    ):
        super().__init__()
        self.layer = torch.nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                model_dim,
                inner_dim,
                ff_dim,
                num_heads,
                relative_attention_bias,
                dtype,
                device,
            )
        )
        self.layer.append(T5LayerFF(model_dim, ff_dim, dtype, device))

    def forward(self, x, past_bias=None):
        x, past_bias = self.layer[0](x, past_bias)
        x = self.layer[-1](x)
        return x, past_bias


class T5Stack(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        model_dim,
        inner_dim,
        ff_dim,
        num_heads,
        vocab_size,
        dtype,
        device,
    ):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, model_dim, device=device)
        self.block = torch.nn.ModuleList(
            [
                T5Block(
                    model_dim,
                    inner_dim,
                    ff_dim,
                    num_heads,
                    relative_attention_bias=(i == 0),
                    dtype=dtype,
                    device=device,
                )
                for i in range(num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(
        self, input_ids, intermediate_output=None, final_layer_norm_intermediate=True
    ):
        intermediate = None
        x = self.embed_tokens(input_ids)
        past_bias = None
        for i, l in enumerate(self.block):
            x, past_bias = l(x, past_bias)
            if i == intermediate_output:
                intermediate = x.clone()
        x = self.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)
        return x, intermediate


class T5(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict["num_layers"]
        self.encoder = T5Stack(
            self.num_layers,
            config_dict["d_model"],
            config_dict["d_model"],
            config_dict["d_ff"],
            config_dict["num_heads"],
            config_dict["vocab_size"],
            dtype,
            device,
        )
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.encoder.embed_tokens = embeddings

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)