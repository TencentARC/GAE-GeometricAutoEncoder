import torch

from .diffaug import DiffAug
from .discriminator import DinoDiscriminator
from .gan_loss import hinge_d_loss, vanilla_d_loss, vanilla_g_loss
from .lpips import LPIPS
from .video_discriminator import VideoPatchDiscriminator3D


def build_discriminator(
    config: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, DiffAug]:
    """Instantiate the configured image or video discriminator."""
    arch_cfg = config.get("arch", {})
    disc_type = str(arch_cfg.get("type", "dino2d")).lower()
    aug_cfg = config.get("augment", {})
    if disc_type in ("patchgan3d", "video_patchgan3d", "3d"):
        augment_prob = float(aug_cfg.get("prob", 0.0))
        if augment_prob > 0:
            raise ValueError(
                "3D discriminator requires sequence-consistent augmentation; "
                "set gan.disc.augment.prob=0."
            )
        disc = VideoPatchDiscriminator3D(
            input_channels=int(arch_cfg.get("input_channels", 3)),
            base_channels=int(arch_cfg.get("base_channels", 64)),
            num_layers=int(arch_cfg.get("num_layers", 4)),
            dropout=float(arch_cfg.get("dropout", 0.0)),
        ).to(device)
        return disc, DiffAug(prob=0.0, cutout=0.0)

    if disc_type not in ("dino", "dino2d"):
        raise ValueError(f"Unknown discriminator type: {disc_type!r}")
    ckpt_path = arch_cfg.get("dino_ckpt_path")
    if not ckpt_path:
        raise ValueError("DINO discriminator requires 'dino_ckpt_path' in gan.disc.arch.")
    disc = DinoDiscriminator(
        device=device,
        dino_ckpt_path=ckpt_path,
        ks=int(arch_cfg.get("ks", 3)),
        key_depths=tuple(arch_cfg.get("key_depths", (2, 5, 8, 11))),
        norm_type=arch_cfg.get("norm_type", "bn"),
        using_spec_norm=bool(arch_cfg.get("using_spec_norm", True)),
        norm_eps=float(arch_cfg.get("norm_eps", 1e-6)),
        recipe=arch_cfg.get("recipe", "S_8"),
    ).to(device)

    augment = DiffAug(
        prob=float(aug_cfg.get("prob", 1.0)),
        cutout=float(aug_cfg.get("cutout", 0.0)),
    )
    return disc, augment


__all__ = [
    "LPIPS",
    "DiffAug",
    "DinoDiscriminator",
    "VideoPatchDiscriminator3D",
    "hinge_d_loss",
    "vanilla_d_loss",
    "vanilla_g_loss",
    "build_discriminator",
]
