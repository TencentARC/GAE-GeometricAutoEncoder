from __future__ import annotations

import torch
from torch import nn


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv3d):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GroupNorm):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class VideoPatchDiscriminator3D(nn.Module):
    """3D PatchGAN for contiguous RGB clips in ``[B, C, T, H, W]`` format."""

    expects_video = True

    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        layers: list[nn.Module] = [
            nn.Conv3d(input_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = base_channels
        for layer_idx in range(1, num_layers):
            next_channels = base_channels * min(2**layer_idx, 8)
            temporal_stride = 2 if layer_idx == 1 else 1
            layers.extend(
                [
                    nn.Conv3d(
                        channels,
                        next_channels,
                        kernel_size=3,
                        stride=(temporal_stride, 2, 2),
                        padding=1,
                        bias=False,
                    ),
                    _group_norm(next_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            channels = next_channels

        next_channels = base_channels * min(2**num_layers, 8)
        layers.extend(
            [
                nn.Conv3d(channels, next_channels, kernel_size=3, stride=1, padding=1, bias=False),
                _group_norm(next_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
                nn.Conv3d(next_channels, 1, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.main = nn.Sequential(*layers)
        self.apply(_init_weights)

    def classify(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(
                "VideoPatchDiscriminator3D expects [B,C,T,H,W], "
                f"got shape={tuple(video.shape)}"
            )
        return self.main(video)

    def forward(
        self,
        fake: torch.Tensor,
        real: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits_fake = self.classify(fake)
        logits_real = self.classify(real) if real is not None else None
        return logits_fake, logits_real
