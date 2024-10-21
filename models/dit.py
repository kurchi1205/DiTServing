import torch.nn as nn
from attention import BasicTransformerBlock
from diffusers import DiTTransformer2DModel

class DiTTransformer2DModelWithAttScores(DiTTransformer2DModel):
    def __init__(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=args.norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )
