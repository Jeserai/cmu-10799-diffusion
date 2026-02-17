"""
Time-dependent classifier for CelebA attributes.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    AttentionBlock,
    Downsample,
    GroupNorm32,
    ResBlock,
    TimestepEmbedding,
)


class TimeDependentClassifier(nn.Module):
    """
    Encoder-style classifier that conditions on timestep.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = (16, 8),
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
        time_scale: float = 1000.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        self.time_scale = float(time_scale)

        time_embed_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(time_embed_dim)

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        curr_channels = base_channels
        curr_resolution = 64
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResBlock(
                        curr_channels,
                        out_ch,
                        time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                self.down_attn.append(AttentionBlock(out_ch, num_heads=num_heads))
                curr_channels = out_ch
            if level != len(channel_mult) - 1:
                self.downsamples.append(Downsample(curr_channels))
                curr_resolution //= 2

        self.out_norm = GroupNorm32(32, curr_channels)
        self.out_linear = nn.Linear(curr_channels, num_classes)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t * self.time_scale
        time_emb = self.time_embed(t)

        h = self.input_conv(x)

        down_block_idx = 0
        downsample_idx = 0
        curr_resolution = x.shape[-1]
        for level, _ in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[down_block_idx](h, time_emb)
                if curr_resolution in self.attention_resolutions:
                    h = self.down_attn[down_block_idx](h)
                down_block_idx += 1
            if level != len(self.channel_mult) - 1:
                h = self.downsamples[downsample_idx](h)
                downsample_idx += 1
                curr_resolution //= 2

        h = self.out_norm(h)
        h = F.silu(h)
        h = h.mean(dim=(2, 3))
        logits = self.out_linear(h)
        return logits


def create_classifier_from_config(config: dict, num_classes: int) -> TimeDependentClassifier:
    model_config = config["model"]
    data_config = config["data"]
    fm_config = config.get("flow_matching", {})
    return TimeDependentClassifier(
        in_channels=data_config["channels"],
        num_classes=num_classes,
        base_channels=model_config["base_channels"],
        channel_mult=tuple(model_config["channel_mult"]),
        num_res_blocks=model_config["num_res_blocks"],
        attention_resolutions=model_config["attention_resolutions"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        use_scale_shift_norm=model_config["use_scale_shift_norm"],
        time_scale=fm_config.get("time_scale", 1000.0),
    )
