"""
DDTHead — Helios-style DDT backbone for GLD.

Key features:

  * Non-concat input: x is (BV, C, H, W) — ref views contain clean latent,
    tgt views contain noisy latent.  No channel-doubling.
  * Ref t=0 adaLN: model internally sets ref view timestep to 0, giving
    distinct adaLN modulations for clean refs vs noisy targets (analogous to
    Helios `zero_history_timestep`).
  * ref_key_scale: per-head learnable scale on ref-view K vectors in every
    attention layer, forcing the model to attend strongly to reference tokens
    (analogous to Helios `history_key_scale`).
  * Tgt-only loss: the model still outputs for all views, but only tgt views
    are supervised.  Loss handling is done in the training script.

Forward signature:
    forward(x, t, total_view, plucker_6d=None, ref_global=None,
            cond_num=..., **kwargs)

  - x: (BV, C, H, W) latent.  First cond_num views per sample are clean refs.
  - t: (BV,) timesteps (same value for all views; model overrides refs to 0).
  - total_view: int V.
  - plucker_6d: (B, V*Hs*Ws, 6).
  - ref_global: (B, T_ref, kdim) optional cross-attn context.
  - cond_num: int.
"""

from __future__ import annotations

from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Mlp

from .model_utils import (
    GaussianFourierEmbedding,
    RMSNorm,
    SwiGLUFFN,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
)
from .DDT import DDTFinalLayer, DDTGate, DDTModulate
from .plucker_attention import PluckerAttention


class TextProjection(nn.Module):
    """Project text encoder output to model hidden dim.

    Mirrors Helios PixArtAlphaTextProjection: two-layer MLP with GELU-tanh.
    """
    def __init__(self, text_embed_dim: int, hidden_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(text_embed_dim, hidden_dim, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


class LightningDDTBlockV3(nn.Module):
    """DiT block with PluckerAttention (Plücker PE + ref-K scale)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        # Plücker Flip PE
        plucker_init: str = "zero",
        plucker_init_scale: float = 0.01,
        plucker_mlp_hidden: int = 64,
        plucker_scale: float = 1.0,
        scale_gate_hidden: int = 0,
        log_scale_aug_prob: float = 0.0,
        log_scale_aug_range: tuple[float, float] = (-1.2, 1.6),
        # ref-K scale
        ref_key_scale: bool = True,
        ref_key_max_scale: float = 10.0,
        ref_key_init_bias: float = 0.0,
        # optional cross-attn
        use_cross_attn: bool = False,
        cross_attn_kdim: Optional[int] = None,
        cross_attn_drop: float = 0.0,
        # Helios-style: text cross-attn only on tgt views (not ref)
        guidance_cross_attn: bool = False,
        **block_kwargs,
    ):
        super().__init__()
        self.wo_shift = wo_shift
        self.use_cross_attn = use_cross_attn
        self.guidance_cross_attn = guidance_cross_attn

        norm_cls = (lambda d: RMSNorm(d)) if use_rmsnorm else (
            lambda d: nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        )
        self.norm1 = norm_cls(hidden_size)
        self.norm2 = norm_cls(hidden_size)

        self.attn = PluckerAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            plucker_init=plucker_init,
            plucker_init_scale=plucker_init_scale,
            plucker_mlp_hidden=plucker_mlp_hidden,
            plucker_scale=plucker_scale,
            scale_gate_hidden=scale_gate_hidden,
            log_scale_aug_prob=log_scale_aug_prob,
            log_scale_aug_range=log_scale_aug_range,
            ref_key_scale=ref_key_scale,
            ref_key_max_scale=ref_key_max_scale,
            ref_key_init_bias=ref_key_init_bias,
            **block_kwargs,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            def approx_gelu():
                return nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )

        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )

        if self.use_cross_attn:
            kdim = cross_attn_kdim if cross_attn_kdim is not None else hidden_size
            self.cross_attn_norm = norm_cls(hidden_size)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                kdim=kdim,
                vdim=kdim,
                dropout=cross_attn_drop,
                batch_first=True,
            )
            # NOTE: out_proj keeps default init (nonzero) so gradient flows through gate.
            # Gate=0 ensures zero output at init (warm-start safe from old checkpoint).
            self.cross_attn_gate = nn.Parameter(torch.full((1,), 0.01))

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        total_view: int,
        cond_num: int = 0,
        feat_rope=None,
        plucker_6d: Optional[torch.Tensor] = None,
        ref_global: Optional[torch.Tensor] = None,
        pag_mode: bool = False,
        num_prefix_tokens: int = 0,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        prope_image_size=None,
        patches_layout=None,
        temporal_frame_idx: Optional[torch.Tensor] = None,
        **_,
    ):
        if c.dim() < x.dim():
            c = c.unsqueeze(1)

        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(4, dim=-1)
            )
            shift_msa = shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=-1)
            )

        x = x + DDTGate(
            self.attn(
                DDTModulate(self.norm1(x), shift_msa, scale_msa),
                rope=feat_rope,
                total_view=total_view,
                plucker_6d=plucker_6d,
                cond_num=cond_num,
                pag_mode=pag_mode,
                num_prefix_tokens=num_prefix_tokens,
                viewmats=viewmats,
                Ks=Ks,
                prope_image_size=prope_image_size,
                patches_layout=patches_layout,
                temporal_frame_idx=temporal_frame_idx,
            ),
            gate_msa,
        )

        if self.use_cross_attn and ref_global is not None:
            BV = x.shape[0]
            B = ref_global.shape[0]
            V = BV // B

            if self.guidance_cross_attn and cond_num > 0:
                # Helios guidance_cross_attn: text only conditions tgt views.
                x_5d = rearrange(x, "(b v) n d -> b v n d", v=V)
                x_ref = x_5d[:, :cond_num]
                x_tgt = x_5d[:, cond_num:]
                x_tgt_flat = rearrange(x_tgt, "b v n d -> (b v) n d")

                tgt_v = V - cond_num
                ctx = ref_global.unsqueeze(1).expand(B, tgt_v, *ref_global.shape[1:])
                ctx = ctx.reshape(B * tgt_v, *ref_global.shape[1:])

                attn_out, _ = self.cross_attn(self.cross_attn_norm(x_tgt_flat), ctx, ctx)
                x_tgt_flat = x_tgt_flat + self.cross_attn_gate * attn_out

                x_tgt = rearrange(x_tgt_flat, "(b v) n d -> b v n d", v=tgt_v)
                x = rearrange(torch.cat([x_ref, x_tgt], dim=1), "b v n d -> (b v) n d")
            else:
                # Apply text to all views (or cond_num==0 so all are tgt).
                ctx = ref_global.unsqueeze(1).expand(B, V, *ref_global.shape[1:])
                ctx = ctx.reshape(BV, *ref_global.shape[1:])
                attn_out, _ = self.cross_attn(self.cross_attn_norm(x), ctx, ctx)
                x = x + self.cross_attn_gate * attn_out

        x = x + DDTGate(
            self.mlp(DDTModulate(self.norm2(x), shift_mlp, scale_mlp)),
            gate_mlp,
        )
        return x


class DDTHead(nn.Module):
    """DDT V3 backbone — Helios-style non-concat with ref t=0 + ref-K scale."""

    def __init__(
        self,
        input_size: int = 1,
        patch_size=1,
        in_channels: int = 384,
        hidden_size=(768, 2048),
        depth=(28, 6),
        num_heads=(16, 16),
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rope: bool = False,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        use_pos_embed: bool = False,
        # Plücker Flip PE
        plucker_init: str = "zero",
        plucker_init_scale: float = 0.01,
        plucker_mlp_hidden: int = 64,
        plucker_scale: float = 1.0,
        scale_gate_hidden: int = 0,
        log_scale_aug_prob: float = 0.0,
        log_scale_aug_range: tuple[float, float] = (-1.2, 1.6),
        # ref-K scale (Helios)
        ref_key_scale: bool = True,
        ref_key_max_scale: float = 10.0,
        ref_key_init_bias: float = 0.0,
        # camera conditioning mode
        camera_conditioning: str = "plucker_flip_pe",
        baseline_camera_mode: str = "plucker",
        cam_input_size: int = 252,
        cam_patch_size: int = 14,  # image→latent downscale (14 for DA3, 16 for DINO)
        # cross-attn + text conditioning
        use_cross_attn: bool = False,
        cross_attn_kdim: Optional[int] = None,
        text_embed_dim: Optional[int] = None,
        guidance_cross_attn: bool = False,
        # memory: 0.0=off, 1.0=all blocks, 0.5=50% blocks checkpointed
        gradient_checkpointing: float = 0.0,
        # compat hooks (unused in V3)
        predict_cls: bool = False,
        num_special_tokens: int = 0,
        source_condition_mode: Optional[str] = None,
        level: int = 3,
        **_unused,
    ):
        super().__init__()
        if camera_conditioning not in ("plucker_flip_pe", "baseline_prope", "none"):
            raise ValueError(
                "camera_conditioning must be 'plucker_flip_pe', "
                f"'baseline_prope', or 'none'; "
                f"got {camera_conditioning!r}"
            )
        self.camera_conditioning = camera_conditioning
        self.baseline_camera_mode = baseline_camera_mode
        self.cam_patch_size = cam_patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.use_rope = use_rope
        self.use_pos_embed = use_pos_embed
        self.use_cross_attn = use_cross_attn
        self.gc_frac = float(gradient_checkpointing)
        self.level = level

        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]
        self.num_heads = (
            list(num_heads) if not isinstance(num_heads, int) else [num_heads, num_heads]
        )
        self.num_encoder_blocks = depth[0]
        self.num_decoder_blocks = depth[1]
        self.num_blocks = depth[0] + depth[1]

        if isinstance(patch_size, (int, float)):
            patch_size = [int(patch_size), int(patch_size)]
        assert len(patch_size) == 2
        self.patch_size = patch_size
        self.s_patch_size = patch_size[0]
        self.x_patch_size = patch_size[1]

        # Pre-projection: C → embed_channels before PatchEmbed.
        # Keeps PatchEmbed capacity at 768 (same as V2 concat mode).
        # Separate ref/tgt projections encode identity ("I am clean" vs "I am noisy").
        embed_channels = in_channels * 2
        self.embed_channels = embed_channels

        self.s_proj_ref = nn.Conv2d(in_channels, embed_channels, 1, bias=True)
        self.s_proj_tgt = nn.Conv2d(in_channels, embed_channels, 1, bias=True)
        self.x_proj_ref = nn.Conv2d(in_channels, embed_channels, 1, bias=True)
        self.x_proj_tgt = nn.Conv2d(in_channels, embed_channels, 1, bias=True)

        # timm PatchEmbed 自己用 Conv2d(in_chans, embed_dim, kernel=stride=patch)
        # 完成 patchify，所以 in_chans 是输入的实际通道数 (embed_channels)，
        # 不能乘 patch_size²。patch_size=1 时乘不乘相等所以一直没暴露；
        # patch_size>1 时乘了会让 conv 期望 embed_channels·patch² 个通道而崩。
        self.x_embedder_ref = PatchEmbed(
            input_size, self.x_patch_size, embed_channels,
            self.decoder_hidden_size, bias=True, strict_img_size=False,
        )
        self.x_embedder_tgt = PatchEmbed(
            input_size, self.x_patch_size, embed_channels,
            self.decoder_hidden_size, bias=True, strict_img_size=False,
        )
        self.s_embedder_ref = PatchEmbed(
            input_size, self.s_patch_size, embed_channels,
            self.encoder_hidden_size, bias=True, strict_img_size=False,
        )
        self.s_embedder_tgt = PatchEmbed(
            input_size, self.s_patch_size, embed_channels,
            self.encoder_hidden_size, bias=True, strict_img_size=False,
        )

        self.x_channel_per_token = in_channels * self.x_patch_size * self.x_patch_size

        self.s_projector = (
            nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
            if self.encoder_hidden_size != self.decoder_hidden_size
            else nn.Identity()
        )
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)

        # Baseline GLD: spatial camera map → PatchEmbed → add to encoder tokens.
        # The camera map lives at IMAGE resolution (cam_input_size), so it must
        # be patchified with cam_patch_size (= image/latent ratio, 14 for DA3)
        # to land on the same Hs×Ws grid as the latent encoder tokens. Using the
        # latent s_patch_size (1) here would emit image_res² tokens (≫ Hs·Ws).
        self.camera_embedder = None
        if camera_conditioning == "baseline_prope":
            cam_ch = 7 if baseline_camera_mode == "plucker" else 4
            self.camera_embedder = PatchEmbed(
                cam_input_size, cam_patch_size, cam_ch,
                self.encoder_hidden_size, bias=True, strict_img_size=False,
            )
            nn.init.constant_(self.camera_embedder.proj.bias, 0)
            w = self.camera_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))

        # Text projection (Helios-style): project text encoder output to cross-attn kdim.
        self.text_proj = (
            TextProjection(text_embed_dim, cross_attn_kdim or self.encoder_hidden_size)
            if text_embed_dim
            else None
        )

        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size, 1, self.x_channel_per_token,
            use_rmsnorm=use_rmsnorm,
        )

        if use_pos_embed:
            num_patches = self.s_embedder_ref.num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.encoder_hidden_size),
                requires_grad=False,
            )
            self.x_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.decoder_hidden_size),
                requires_grad=False,
            )

        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]
        if self.use_rope:
            enc_half_head_dim = self.encoder_hidden_size // enc_num_heads // 2
            hw_seq_len = int(sqrt(self.s_embedder_ref.num_patches))
            self.enc_feat_rope = VisionRotaryEmbeddingFast(
                dim=enc_half_head_dim, pt_seq_len=hw_seq_len,
            )
            dec_half_head_dim = self.decoder_hidden_size // dec_num_heads // 2
            hw_seq_len = int(sqrt(self.x_embedder_ref.num_patches))
            self.dec_feat_rope = VisionRotaryEmbeddingFast(
                dim=dec_half_head_dim, pt_seq_len=hw_seq_len,
            )
        else:
            self.enc_feat_rope = None
            self.dec_feat_rope = None
        self._enc_rope_cache: dict = {}
        self._dec_rope_cache: dict = {}
        self.enc_half_head_dim = (
            self.encoder_hidden_size // enc_num_heads // 2 if self.use_rope else 0
        )
        self.dec_half_head_dim = (
            self.decoder_hidden_size // dec_num_heads // 2 if self.use_rope else 0
        )

        block_common = dict(
            mlp_ratio=mlp_ratio,
            use_qknorm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            use_swiglu=use_swiglu,
            wo_shift=wo_shift,
            plucker_init=plucker_init,
            plucker_init_scale=plucker_init_scale,
            plucker_mlp_hidden=plucker_mlp_hidden,
            plucker_scale=plucker_scale,
            scale_gate_hidden=scale_gate_hidden,
            log_scale_aug_prob=log_scale_aug_prob,
            log_scale_aug_range=log_scale_aug_range,
            ref_key_scale=ref_key_scale,
            ref_key_max_scale=ref_key_max_scale,
            ref_key_init_bias=ref_key_init_bias,
            use_baseline_prope=(camera_conditioning == "baseline_prope"),
            use_cross_attn=use_cross_attn,
            cross_attn_kdim=cross_attn_kdim,
            guidance_cross_attn=guidance_cross_attn,
        )
        self.blocks = nn.ModuleList([
            LightningDDTBlockV3(
                hidden_size=(
                    self.encoder_hidden_size if i < self.num_encoder_blocks
                    else self.decoder_hidden_size
                ),
                num_heads=(
                    enc_num_heads if i < self.num_encoder_blocks
                    else dec_num_heads
                ),
                **block_common,
            )
            for i in range(self.num_blocks)
        ])

        self._gc_set = self._build_gc_set(self.gc_frac)

        self.initialize_weights()

    def _build_gc_set(self, frac: float) -> set:
        """Pre-compute which block indices should be gradient-checkpointed.

        Encoder and decoder each independently select ceil(frac * depth) blocks,
        evenly spaced, so both stages are equally protected.
        """
        if frac <= 0.0:
            return set()

        def _pick(start: int, length: int) -> list:
            k = max(1, round(frac * length))
            if k >= length:
                return list(range(start, start + length))
            step = length / k
            return [start + int(i * step) for i in range(k)]

        enc = _pick(0, self.num_encoder_blocks)
        dec = _pick(self.num_encoder_blocks, self.num_decoder_blocks)
        return set(enc + dec)

    def initialize_weights(self):
        for proj in (self.s_proj_ref, self.s_proj_tgt,
                     self.x_proj_ref, self.x_proj_tgt):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.constant_(proj.bias, 0)

        for emb in (self.x_embedder_ref, self.x_embedder_tgt,
                    self.s_embedder_ref, self.s_embedder_tgt):
            w = emb.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(emb.proj.bias, 0)

        if self.use_pos_embed:
            grid_size = int(self.s_embedder_ref.num_patches ** 0.5)
            pe_s = get_2d_sincos_pos_embed(self.encoder_hidden_size, grid_size)
            self.pos_embed.data.copy_(torch.from_numpy(pe_s).float().unsqueeze(0))
            pe_x = get_2d_sincos_pos_embed(self.decoder_hidden_size, grid_size)
            self.x_pos_embed.data.copy_(torch.from_numpy(pe_x).float().unsqueeze(0))

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_sincos_pos(self, embed_dim, H, W, device, dtype):
        pe = get_2d_sincos_pos_embed(embed_dim, (H, W))
        return torch.from_numpy(pe).to(device=device, dtype=dtype).unsqueeze(0)

    def _get_rope(self, cache, dim, H, W, device, dtype):
        key = (device.type, getattr(device, "index", None), str(dtype), dim, H, W)
        rope = cache.get(key)
        if rope is None:
            rope = VisionRotaryEmbeddingFast(dim=dim, pt_seq_len=(H, W)).to(device=device)
            cache[key] = rope
        return rope

    def uses_baseline_camera(self) -> bool:
        return self.camera_conditioning == "baseline_prope"

    def _maybe_slice_baseline_camera(
        self, camera_embedding, viewmats, Ks, B, total_view, cond_num, z_ref_clean,
    ):
        """Drop the cond-extended prefix from baseline camera tensors when the
        encoder runs without cond tokens (z_ref_clean=None, e.g. CFG uncond /
        t2v). v4_cond_extend builds [cond(K) | views(V)] = K+V camera views; with
        no cond tokens the encoder only has V view tokens, so PRoPE (cameras=K+V
        vs seqlen=V*N) and the spatial camera add (V vs K+V) would both break."""
        if not self.uses_baseline_camera() or z_ref_clean is not None:
            return camera_embedding, viewmats, Ks
        if viewmats is not None and viewmats.shape[1] == total_view + cond_num:
            viewmats = viewmats[:, cond_num:]
            Ks = Ks[:, cond_num:] if Ks is not None else None
        if camera_embedding is not None:
            v_enc = camera_embedding.shape[0] // B
            if v_enc == total_view + cond_num:
                cam5 = rearrange(
                    camera_embedding, "(b v) c h w -> b v c h w", v=v_enc,
                )
                camera_embedding = rearrange(
                    cam5[:, cond_num:], "b v c h w -> (b v) c h w",
                )
        return camera_embedding, viewmats, Ks

    def _apply_baseline_camera_embed(
        self, s: torch.Tensor, camera_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if camera_embedding is None or self.camera_embedder is None:
            return s
        cam_emb = self.camera_embedder(camera_embedding)
        if cam_emb.shape[1] != s.shape[1]:
            raise ValueError(
                f"camera tokens {cam_emb.shape[1]} != encoder tokens {s.shape[1]}"
            )
        return s + cam_emb

    def _camera_attention_kwargs(
        self,
        *,
        plucker_6d: Optional[torch.Tensor],
        viewmats: Optional[torch.Tensor],
        Ks: Optional[torch.Tensor],
        Hs: int,
        Ws: int,
    ) -> dict:
        if self.uses_baseline_camera():
            # PRoPE uses two different scales:
            #   - patches_layout = latent grid (Hs×Ws) → RoPE spatial positions.
            #   - prope_image_size = IMAGE resolution (Hs·cam_patch_size) → used
            #     to normalize Ks (fx/image_w). Ks is in pixel units at image
            #     resolution, so this must be the image size, NOT the latent grid.
            return dict(
                plucker_6d=None,
                viewmats=viewmats,
                Ks=Ks,
                prope_image_size=(Hs * self.cam_patch_size, Ws * self.cam_patch_size),
                patches_layout=(Hs, Ws),
            )
        return dict(plucker_6d=plucker_6d)

    def unpatchify(self, x, h=None, w=None):
        c = self.in_channels
        p = self.x_patch_size
        if h is None or w is None:
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1]
        else:
            assert h * w == x.shape[1], f"token mismatch: {x.shape[1]} vs {h*w}"
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        total_view: int,
        plucker_6d: Optional[torch.Tensor] = None,
        ref_global: Optional[torch.Tensor] = None,
        camera_embedding: Optional[torch.Tensor] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        pag_mode: bool = False,
        pag_layer_idx: Optional[int] = None,
        **kwargs,
    ):
        """V3 forward.

        Args:
            x: (BV, C, H, W) latent.  First cond_num views per sample are
               clean ref, the rest are noisy tgt.
            t: (BV,) timesteps — model internally overrides ref views to t=0.
            total_view: V.
            plucker_6d: (B, V*Hs*Ws, 6).
            ref_global: (B, T_ref, kdim) cross-attn context.
            s: optional pre-computed encoder output.
            kwargs: cond_num (required), pag_mode, etc.

        Returns:
            (BV, C, H, W) predicted velocity for all views.
        """
        BV = x.shape[0]
        assert BV % total_view == 0
        B = BV // total_view

        cond_num = kwargs.get("cond_num", 1)
        x_patches = x

        Hs = x_patches.shape[2] // self.s_patch_size
        Ws = x_patches.shape[3] // self.s_patch_size
        Hx = x_patches.shape[2] // self.x_patch_size
        Wx = x_patches.shape[3] // self.x_patch_size

        # ── Ref t=0 adaLN (Helios zero_history_timestep) ─────────────
        t_v3 = rearrange(t, "(b v) -> b v", v=total_view).clone()
        t_v3[:, :cond_num] = 0.0
        t_flat = rearrange(t_v3, "b v -> (b v)")

        t_emb = self.t_embedder(t_flat)
        c = nn.functional.silu(t_emb)

        # ── Text projection (once, shared across all blocks) ──
        if self.text_proj is not None and ref_global is not None:
            ref_global = self.text_proj(ref_global)

        # ── Encoder ──────────────────────────────────────────────────
        if s is None:
            xp_5d = rearrange(x_patches, "(b v) c h w -> b v c h w", v=total_view)
            xp_tgt = xp_5d[:, cond_num:].reshape(-1, *xp_5d.shape[2:])
            s_tgt = self.s_embedder_tgt(self.s_proj_tgt(xp_tgt))
            if cond_num > 0:
                xp_ref = xp_5d[:, :cond_num].reshape(-1, *xp_5d.shape[2:])
                s_ref = self.s_embedder_ref(self.s_proj_ref(xp_ref))
                s = torch.cat([
                    rearrange(s_ref, "(b v) n d -> b v n d", v=cond_num),
                    rearrange(s_tgt, "(b v) n d -> b v n d", v=total_view - cond_num),
                ], dim=1)
            else:
                s = rearrange(s_tgt, "(b v) n d -> b v n d", v=total_view)
            s = rearrange(s, "b v n d -> (b v) n d")

            if self.use_pos_embed:
                pe_s = self._get_sincos_pos(
                    self.encoder_hidden_size, Hs, Ws, s.device, s.dtype,
                )
                s = s + pe_s

            s = self._apply_baseline_camera_embed(s, camera_embedding)

            enc_rope = None
            if self.use_rope:
                enc_rope = self._get_rope(
                    self._enc_rope_cache, self.enc_half_head_dim,
                    Hs, Ws, s.device, s.dtype,
                )

            cam_kwargs = self._camera_attention_kwargs(
                plucker_6d=plucker_6d, viewmats=viewmats, Ks=Ks, Hs=Hs, Ws=Ws,
            )
            for i in range(self.num_encoder_blocks):
                cur_pag = pag_mode if (pag_layer_idx is None or i == pag_layer_idx) else False
                block_kwargs = dict(
                    total_view=total_view,
                    cond_num=cond_num,
                    feat_rope=enc_rope,
                    ref_global=ref_global,
                    pag_mode=cur_pag,
                    **cam_kwargs,
                )
                if self.training and i in self._gc_set:
                    s = grad_ckpt(
                        self.blocks[i], s, c,
                        use_reentrant=False, **block_kwargs,
                    )
                else:
                    s = self.blocks[i](s, c, **block_kwargs)

            t_broadcast = t_emb.unsqueeze(1).expand(-1, s.shape[1], -1)
            s = nn.functional.silu(t_broadcast + s)

        s = self.s_projector(s)

        # ── Decoder ──────────────────────────────────────────────────
        xp_5d = rearrange(x_patches, "(b v) c h w -> b v c h w", v=total_view)
        xp_tgt = xp_5d[:, cond_num:].reshape(-1, *xp_5d.shape[2:])
        x_toks_tgt = self.x_embedder_tgt(self.x_proj_tgt(xp_tgt))
        if cond_num > 0:
            xp_ref = xp_5d[:, :cond_num].reshape(-1, *xp_5d.shape[2:])
            x_toks_ref = self.x_embedder_ref(self.x_proj_ref(xp_ref))
            x_toks = torch.cat([
                rearrange(x_toks_ref, "(b v) n d -> b v n d", v=cond_num),
                rearrange(x_toks_tgt, "(b v) n d -> b v n d", v=total_view - cond_num),
            ], dim=1)
        else:
            x_toks = rearrange(x_toks_tgt, "(b v) n d -> b v n d", v=total_view)
        x_toks = rearrange(x_toks, "b v n d -> (b v) n d")

        if self.use_pos_embed:
            pe_x = self._get_sincos_pos(
                self.decoder_hidden_size, Hx, Wx, x_toks.device, x_toks.dtype,
            )
            x_toks = x_toks + pe_x

        dec_rope = None
        if self.use_rope:
            dec_rope = self._get_rope(
                self._dec_rope_cache, self.dec_half_head_dim,
                Hx, Wx, x_toks.device, x_toks.dtype,
            )

        for i in range(self.num_encoder_blocks, self.num_blocks):
            cur_pag = pag_mode if (pag_layer_idx is None or i == pag_layer_idx) else False
            block_kwargs = dict(
                total_view=total_view,
                cond_num=cond_num,
                feat_rope=dec_rope,
                ref_global=ref_global,
                pag_mode=cur_pag,
                **self._camera_attention_kwargs(
                    plucker_6d=plucker_6d, viewmats=viewmats, Ks=Ks, Hs=Hx, Ws=Wx,
                ),
            )
            if self.training and i in self._gc_set:
                x_toks = grad_ckpt(
                    self.blocks[i], x_toks, s,
                    use_reentrant=False, **block_kwargs,
                )
            else:
                x_toks = self.blocks[i](x_toks, s, **block_kwargs)

        x_toks = self.final_layer(x_toks, s)
        return self.unpatchify(x_toks, Hx, Wx)

    # ── CFG inference ────────────────────────────────────────────────

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        total_view: int,
        plucker_6d: Optional[torch.Tensor] = None,
        ref_global: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        cfg_interval: tuple = (0.0, 1.0),
        uncond_mode: str = "zero_plucker",
        **kwargs,
    ):
        """CFG forward for V3.

        uncond_mode controls unconditional branch:
          - 'zero_plucker': zero out plucker + ref_global.  Ref views stay as-is
            (clean latent still provides basic conditioning).
          - 'ref_noise': replace ref views with noise in the unconditional half.
        """
        half_n = x.shape[0]
        combined_x = torch.cat([x, x], dim=0)
        combined_t = torch.cat([t, t], dim=0)

        if plucker_6d is not None:
            uncond_plucker = torch.zeros_like(plucker_6d)
            combined_plucker = torch.cat([plucker_6d, uncond_plucker], dim=0)
        else:
            combined_plucker = None

        if ref_global is not None:
            uncond_ref_global = torch.zeros_like(ref_global)
            combined_ref_global = torch.cat([ref_global, uncond_ref_global], dim=0)
        else:
            combined_ref_global = None

        cfg_kwargs = dict(kwargs)

        # ref_noise: replace ref views in unconditional half with noise
        if uncond_mode == "ref_noise":
            cond_num = cfg_kwargs.get("cond_num", 1)
            total_view = total_view
            B = half_n // total_view
            x_uncond = combined_x[half_n:].clone()
            x_uncond_5d = rearrange(x_uncond, "(b v) c h w -> b v c h w", v=total_view)
            x_uncond_5d[:, :cond_num] = torch.randn_like(x_uncond_5d[:, :cond_num])
            combined_x[half_n:] = rearrange(x_uncond_5d, "b v c h w -> (b v) c h w")

        out = self.forward(
            combined_x, combined_t, total_view,
            plucker_6d=combined_plucker,
            ref_global=combined_ref_global,
            **cfg_kwargs,
        )

        cond_out = out[:half_n]
        uncond_out = out[half_n:]

        t_min, t_max = cfg_interval
        in_window = (t >= t_min) & (t <= t_max)
        win = in_window.float().view(-1, 1, 1, 1)
        guided = uncond_out + cfg_scale * (cond_out - uncond_out)
        return win * guided + (1.0 - win) * cond_out
