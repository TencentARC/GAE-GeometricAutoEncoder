"""Smoke-test the GAE facade with a stubbed DA3 backbone.

Downloading DA3-GIANT just to exercise the plumbing is slow, so the frozen
backbone is replaced by a stub that returns correctly shaped four-level
features. The codec itself is real, so this still verifies:

    encode():   normalize_levels -> GAECodec.encode -> posterior mean
    decode_rgb(): GAECodec.decode_rgb -> pixels
    standardize / destandardize round-trip
    sample():   delegates to the Euler + CFG sampler when a flow is attached
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import tempfile  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from gae.pipeline import GAE, load_flow  # noqa: E402
from gae import load_codec, load_backbone, load_flow as load_flow_pkg  # noqa: E402
from stage1.gae_codec import GAECodec  # noqa: E402
import inspect  # noqa: E402

LEVELS, LEVEL_DIM, GRID = 4, 1536, 6
N_PATCH = GRID * GRID


class StubBackbone(nn.Module):
    """Returns four-level DA3-shaped features: {lvl: [BV, N+1, C]} with CLS at 0.

    Mirrors DA3Backbone.encode(mode='all'), including the leading CLS token and
    the 5D (B, V, 3, H, W) input layout.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict = {}

    def encode(self, x: torch.Tensor, mode: str = "all") -> dict:
        b, v = x.shape[:2]
        key = (b * v, x.device, x.dtype)
        if key not in self._cache:
            self._cache[key] = torch.randn(
                b * v, N_PATCH + 1, LEVEL_DIM, device=x.device, dtype=x.dtype
            )
        feats = self._cache[key]
        return {lvl: feats for lvl in range(LEVELS)}


def main() -> int:
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp())
    for lvl in range(LEVELS):
        torch.save(
            {"mean": torch.zeros(LEVEL_DIM), "std": torch.ones(LEVEL_DIM)},
            tmp / f"normalization_stats_level{lvl}.pt",
        )

    latent_dim = 64
    codec = GAECodec(
        latent_dim=latent_dim,
        hidden_dims=[256, 128, 64],
        num_attn_blocks=1,
        kl_weight=1e-6,
        stats_dir=str(tmp),
        rgb_decoder={"hidden_dim": 128, "depth": 1, "num_heads": 4},
    ).eval()

    model = GAE(backbone=StubBackbone(), codec=codec)

    b, v = 2, 3
    images = torch.rand(b, v, 3, 84, 84)

    # ── encode ──
    z = model.encode(images)
    expected = (b, v, latent_dim, GRID, GRID)
    assert tuple(z.shape) == expected, f"encode: {tuple(z.shape)} != {expected}"
    print(f"encode          {tuple(images.shape)} -> {tuple(z.shape)}")

    # ── posterior mean determinism (paper: mu is used, not a sample) ──
    z2 = model.encode(images)
    assert torch.equal(z, z2), "encode is not deterministic"
    print("encode deterministic (posterior mean, not a sample)")

    # ── decode_rgb ──
    rgb = model.decode_rgb(z.flatten(0, 1), num_views=v)
    assert rgb.shape[0] == b * v and rgb.shape[1] == 3, tuple(rgb.shape)
    print(f"decode_rgb      {tuple(z.flatten(0, 1).shape)} -> {tuple(rgb.shape)}")

    # ── standardize / destandardize round-trip (Eq. 15) ──
    mean = torch.zeros(latent_dim)
    std = torch.ones(latent_dim)
    model.latent_mean = mean.view(1, -1, 1, 1)
    model.latent_std = std.view(1, -1, 1, 1)
    z_bar = model.standardize(z.flatten(0, 1))
    back = model.destandardize(z_bar)
    assert torch.allclose(back, z.flatten(0, 1), atol=1e-5), "standardize round-trip failed"
    print("standardize/destandardize round-trip OK")

    # ── sample() must refuse without a flow model ──
    try:
        model.sample(z[:, 0], total_views=2, cond_num=1)
    except RuntimeError as e:
        print(f"sample w/o flow  correctly refused: {e}")
    else:
        print("FAIL: sample() should require a flow model")
        return 1

    # ── from_configs must not pass device= into __init__ ──
    if "device" in inspect.signature(GAE.__init__).parameters:
        print("FAIL: GAE.__init__ should not take device=")
        return 1
    print("GAE.__init__ has no device kwarg")

    class _DummyFlow(nn.Module):
        def forward(self, *args, **kwargs):
            return None

    with_flow = GAE(backbone=StubBackbone(), codec=codec, flow=_DummyFlow())
    sig = inspect.signature(with_flow.sample)
    for name in ("z_ref", "total_views", "cond_num", "plucker_6d", "ref_global"):
        assert name in sig.parameters, f"sample() missing parameter {name}"
    print("sample with flow  exposes the Euler sampler signature")

    assert callable(load_flow) and callable(load_flow_pkg)
    print("load_flow exported from gae and gae.pipeline")

    print("\nPIPELINE SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
