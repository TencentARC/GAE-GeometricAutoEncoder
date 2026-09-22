"""Positional-encoding helpers for baseline camera conditioning.

Only ``freq_encoding`` is used by the released baseline-PRoPE path
(``utils/camera/camera.py``); the broader research encodings were trimmed.
"""

import torch


def freq_encoding(rays_3d, embed_dim=16, camera_longest_side=None):
    # rays_3d: [b, n, 3]
    rays_3d = rays_3d.to(torch.float32)
    if camera_longest_side is not None:
        rescale_value = 512 / camera_longest_side
        rays_3d = rays_3d * rescale_value
    else:  # camera translation is normalized into [-1~1]
        rays_3d = rays_3d * 512

    pe = []
    freq_bands = 2. ** torch.linspace(0., embed_dim // 2 - 1, steps=embed_dim // 2)
    for freq in freq_bands:
        for p_fn in [torch.sin, torch.cos]:
            pe.append(p_fn(rays_3d * freq))

    pe = torch.cat(pe, dim=-1)

    return pe
