import torch.nn as nn
from .attention import BasicTransformerBlockWithScores
from diffusers import Transformer2DModel

class Transformer2DModelWithAttScores(Transformer2DModel):
    def __init__(self, *args, **kwargs):
        kwargs.pop("cross_attention_dim")
        kwargs.pop("num_vector_embeds")
        kwargs.pop("only_cross_attention")
        kwargs.pop("use_linear_projection")
        super().__init__(*args, **kwargs)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlockWithScores(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=self.config.norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )
