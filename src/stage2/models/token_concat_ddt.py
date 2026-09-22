"""
TokenConcatDDT — Token-Concat Conditioning (Wan2.1-style).

Inherits from DDTHead and overrides forward to:
  1. Accept z_ref_clean as extra conditioning tokens
  2. Treat ALL V views equally in the ODE (no ref t=0 override)
  3. Concatenate cond tokens in the encoder token sequence
  4. Extract only view tokens for decoder output

Cond tokens use the SAME s_proj_ref / s_embedder_ref as DDTHead's ref views.
This means:
  - Zero new parameters (100% weight reuse from DDTHead pretrained)
  - Cond tokens produce the same representation as DDTHead ref tokens from day one
  - DDTHead pretrained attention patterns can immediately utilize cond tokens
  - All V view slots use s_proj_tgt / s_embedder_tgt (they're all "noisy")
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
from einops import rearrange

from .ddt_head import DDTHead
from .temporal_rope import build_temporal_frame_idx


class TokenConcatDDT(DDTHead):
    """DDT V4 — Token-concat conditioning, inheriting V3 architecture.

    Cond tokens reuse V3's s_proj_ref + s_embedder_ref (no new parameters).
    All V view tokens use s_proj_tgt + s_embedder_tgt.

    Key differences from V3 forward:
      1. z_ref_clean projected via s_proj_ref + s_embedder_ref as extra tokens
      2. All V views share the SAME timestep (no ref t=0 override)
      3. Cond tokens use t=0 embedding (they're always clean)
      4. Cond tokens get the SAME plucker rays as the ref views
      5. Output velocity is for ALL V views (no ref/tgt asymmetry)
    """

    def __init__(self, **kwargs):
        # Pop V4-only kwargs before passing to V3
        kwargs.pop("cond_token_mode", None)
        # ── REPA (opt-in): align an intermediate encoder layer to a frozen
        # semantic target (DA3 features). Parameter-free unless enabled. ──
        self.repa_enable = bool(kwargs.pop("repa_enable", False))
        self.repa_align_depth = int(kwargs.pop("repa_align_depth", 8))
        repa_target_dim = int(kwargs.pop("repa_target_dim", 2048))
        repa_proj_hidden = int(kwargs.pop("repa_proj_hidden", 2048))
        super().__init__(**kwargs)
        # No new parameters — cond tokens reuse s_proj_ref + s_embedder_ref

        # Buffer for the per-forward projected feature (read by the trainer).
        self._repa_pred: Optional[torch.Tensor] = None
        if self.repa_enable:
            assert 0 <= self.repa_align_depth < self.num_encoder_blocks, (
                f"repa_align_depth={self.repa_align_depth} out of encoder range "
                f"[0,{self.num_encoder_blocks})"
            )
            self.repa_projector = nn.Sequential(
                nn.Linear(self.encoder_hidden_size, repa_proj_hidden),
                nn.SiLU(),
                nn.Linear(repa_proj_hidden, repa_proj_hidden),
                nn.SiLU(),
                nn.Linear(repa_proj_hidden, repa_target_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        total_view: int,
        z_ref_clean: Optional[torch.Tensor] = None,
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
        """V4 forward — token-concat conditioning.

        Args:
            x: (BV, C, H, W) noisy latent for ALL V views.
            t: (BV,) timesteps — same t for all views (standard ODE).
            total_view: V.
            z_ref_clean: (B*cond_num, C, H, W) clean ref latent for conditioning.
                         If None, runs unconditional (for CFG).
            plucker_6d: (B, V*Hs*Ws, 6) per-token camera rays.
            ref_global: (B, T_ref, kdim) text cross-attn context.
            kwargs: cond_num (required).

        Returns:
            (BV, C, H, W) predicted velocity for ALL V views.
        """
        BV = x.shape[0]
        assert BV % total_view == 0
        B = BV // total_view
        cond_num = kwargs.get("cond_num", 1)
        view_frame_idx = kwargs.get("view_frame_idx", None)
        self._repa_pred = None  # reset; populated below if REPA is enabled

        # See ddt_head._maybe_slice_baseline_camera: drop cond prefix when running
        # without cond tokens (CFG uncond / t2v) so camera view-count matches.
        camera_embedding, viewmats, Ks = self._maybe_slice_baseline_camera(
            camera_embedding, viewmats, Ks, B, total_view, cond_num, z_ref_clean,
        )

        Hs = x.shape[2] // self.s_patch_size
        Ws = x.shape[3] // self.s_patch_size
        Hx = x.shape[2] // self.x_patch_size
        Wx = x.shape[3] // self.x_patch_size

        # ── Timestep embedding ──
        # V4: NO ref t=0 override — all views share the same t
        t_emb = self.t_embedder(t)  # (BV, encoder_hidden_size)
        c = nn.functional.silu(t_emb)

        # Cond tokens get t=0 (they represent clean features)
        if z_ref_clean is not None:
            t_cond = torch.zeros(B * cond_num, device=t.device, dtype=t.dtype)
            t_cond_emb = self.t_embedder(t_cond)
            c_cond = nn.functional.silu(t_cond_emb)  # (B*cond_num, D)

        # ── Text projection ──
        if self.text_proj is not None and ref_global is not None:
            ref_global = self.text_proj(ref_global)

        # ── Encoder ──────────────────────────────────────────────────
        if s is None:
            # All V view tokens use tgt projector (they're all noisy)
            x_5d = rearrange(x, "(b v) c h w -> b v c h w", v=total_view)
            x_flat = x_5d.reshape(-1, *x_5d.shape[2:])  # (BV, C, H, W)
            s_views = self.s_embedder_tgt(self.s_proj_tgt(x_flat))  # (BV, Ns, D)

            # Cond tokens use REF projector (same as V3 ref — pretrained!)
            if z_ref_clean is not None:
                s_cond = self.s_embedder_ref(self.s_proj_ref(z_ref_clean))  # (B*K, Ns, D)

                # Concat: [cond_tokens | view_tokens]
                s_views_5d = rearrange(s_views, "(b v) n d -> b v n d", v=total_view)
                s_cond_5d = rearrange(s_cond, "(b k) n d -> b k n d", k=cond_num)
                s_all = torch.cat([s_cond_5d, s_views_5d], dim=1)  # (B, K+V, Ns, D)
                s_all = rearrange(s_all, "b v n d -> (b v) n d")
                total_enc_views = cond_num + total_view

                # adaLN: cond uses c_cond (t=0), views use c (t=t_cur)
                c_views_5d = rearrange(c, "(b v) d -> b v d", v=total_view)
                c_cond_5d = rearrange(c_cond, "(b k) d -> b k d", k=cond_num)
                c_all = torch.cat([c_cond_5d, c_views_5d], dim=1)
                c_all = rearrange(c_all, "b v d -> (b v) d")
            else:
                s_all = s_views
                c_all = c
                total_enc_views = total_view

            # Positional embedding
            if self.use_pos_embed:
                pe_s = self._get_sincos_pos(
                    self.encoder_hidden_size, Hs, Ws, s_all.device, s_all.dtype)
                s_all = s_all + pe_s

            # RoPE
            enc_rope = None
            if self.use_rope:
                enc_rope = self._get_rope(
                    self._enc_rope_cache, self.enc_half_head_dim,
                    Hs, Ws, s_all.device, s_all.dtype)

            # Plucker rays: cond tokens get ref camera rays, views get all rays
            enc_plucker = None
            if plucker_6d is not None:
                if z_ref_clean is not None:
                    plucker_5d = rearrange(plucker_6d, "b (v n) d -> b v n d",
                                           v=total_view, n=Hs * Ws)
                    plucker_cond = plucker_5d[:, :cond_num]  # (B, K, Ns, 6)
                    plucker_all = torch.cat([plucker_cond, plucker_5d], dim=1)
                    enc_plucker = rearrange(plucker_all, "b v n d -> b (v n) d")
                else:
                    enc_plucker = plucker_6d

            s_all = self._apply_baseline_camera_embed(s_all, camera_embedding)

            # Temporal RoPE frame index: prepended cond views reuse the frame
            # index of the target view they replicate (cond_k ↔ tgt view_k).
            enc_frame_idx = None
            if getattr(self, "_temporal_cfg", None) is not None:
                enc_frame_idx = build_temporal_frame_idx(
                    total_view, cond_num, z_ref_clean is not None, s_all.device,
                    view_frame_idx=view_frame_idx,
                )

            cam_kwargs = self._camera_attention_kwargs(
                plucker_6d=enc_plucker, viewmats=viewmats, Ks=Ks, Hs=Hs, Ws=Ws,
            )
            # Encoder blocks
            for i in range(self.num_encoder_blocks):
                cur_pag = pag_mode if (pag_layer_idx is None or i == pag_layer_idx) else False
                block_kwargs = dict(
                    total_view=total_enc_views,
                    cond_num=cond_num,  # ref_key_scale amplifies cond tokens
                    feat_rope=enc_rope,
                    ref_global=ref_global,
                    pag_mode=cur_pag,
                    temporal_frame_idx=enc_frame_idx,
                    **cam_kwargs,
                )
                if self.training and i in self._gc_set:
                    s_all = grad_ckpt(
                        self.blocks[i], s_all, c_all,
                        use_reentrant=False, **block_kwargs,
                    )
                else:
                    s_all = self.blocks[i](s_all, c_all, **block_kwargs)

                # REPA: project the view tokens of this layer to the target space.
                if self.repa_enable and i == self.repa_align_depth:
                    if z_ref_clean is not None:
                        v_tokens = rearrange(
                            s_all, "(b v) n d -> b v n d", v=total_enc_views,
                        )[:, cond_num:]
                        v_tokens = rearrange(v_tokens, "b v n d -> (b v) n d")
                    else:
                        v_tokens = s_all
                    self._repa_pred = self.repa_projector(v_tokens)

            # Post-encoder: add timestep and activate
            t_broadcast = c_all.unsqueeze(1).expand(-1, s_all.shape[1], -1)
            s_all = nn.functional.silu(t_broadcast + s_all)

            # Extract only VIEW tokens (drop cond tokens)
            if z_ref_clean is not None:
                s_all_5d = rearrange(s_all, "(b v) n d -> b v n d", v=total_enc_views)
                s = rearrange(s_all_5d[:, cond_num:], "b v n d -> (b v) n d")
            else:
                s = s_all

        s = self.s_projector(s)

        # ── Decoder ──────────────────────────────────────────────────
        # All views use tgt projector (no ref/tgt split)
        x_flat = rearrange(x, "(b v) c h w -> (b v) c h w", v=total_view)
        x_toks = self.x_embedder_tgt(self.x_proj_tgt(x_flat))

        if self.use_pos_embed:
            pe_x = self._get_sincos_pos(
                self.decoder_hidden_size, Hx, Wx, x_toks.device, x_toks.dtype)
            x_toks = x_toks + pe_x

        dec_rope = None
        if self.use_rope:
            dec_rope = self._get_rope(
                self._dec_rope_cache, self.dec_half_head_dim,
                Hx, Wx, x_toks.device, x_toks.dtype)

        # Decoder operates on V view tokens only (cond tokens were dropped after
        # the encoder). In baseline_prope mode viewmats/Ks were cond-extended to
        # K+V for the encoder, so PRoPE would see cameras=K+V vs seqlen=V*N and
        # trip its `seqlen == cameras*patches` assert. Slice off the cond prefix.
        dec_viewmats, dec_Ks = viewmats, Ks
        if (
            self.uses_baseline_camera()
            and viewmats is not None
            and viewmats.shape[1] == total_view + cond_num
        ):
            dec_viewmats = viewmats[:, cond_num:]
            dec_Ks = Ks[:, cond_num:] if Ks is not None else None

        # Decoder sees only the V target views → frame index 0..V-1.
        dec_frame_idx = None
        if getattr(self, "_temporal_cfg", None) is not None:
            dec_frame_idx = build_temporal_frame_idx(
                total_view, 0, False, x_toks.device,
                view_frame_idx=view_frame_idx,
            )

        for i in range(self.num_encoder_blocks, self.num_blocks):
            cur_pag = pag_mode if (pag_layer_idx is None or i == pag_layer_idx) else False
            block_kwargs = dict(
                total_view=total_view,
                cond_num=0,
                feat_rope=dec_rope,
                ref_global=ref_global,
                pag_mode=cur_pag,
                temporal_frame_idx=dec_frame_idx,
                **self._camera_attention_kwargs(
                    plucker_6d=plucker_6d, viewmats=dec_viewmats, Ks=dec_Ks, Hs=Hx, Ws=Wx,
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
        out = self.unpatchify(x_toks, Hx, Wx)

        # REPA 头 (repa_projector) 只喂给 self._repa_pred 这个旁路，由 trainer 的
        # REPA loss 单独消费、不在本 forward 的返回值里。DDP(find_unused_parameters=
        # True) 在 forward 末尾从返回张量反向遍历判定 used 参数 → 够不到
        # repa_projector → 标记为 unused 先 mark ready；随后 repa_loss.backward 又
        # 触发它的 grad hook → "marked ready twice"。用零系数把 _repa_pred 接回
        # 输出，让 DDP 把它视作 used（数值/梯度均为 0，仅建立图依赖）。
        # 仅训练期生效；推理 (forward_with_cfg / 采样) 时 self.training=False 不触发。
        if self.training and self._repa_pred is not None:
            out = out + 0.0 * self._repa_pred.sum().to(out.dtype)
        return out

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        total_view: int,
        z_ref_clean: Optional[torch.Tensor] = None,
        plucker_6d: Optional[torch.Tensor] = None,
        ref_global: Optional[torch.Tensor] = None,
        camera_embedding: Optional[torch.Tensor] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        cfg_interval: tuple = (0.0, 1.0),
        **kwargs,
    ):
        """CFG forward for V4: unconditional = drop z_ref_clean + ref_global."""
        t_scalar = t[0].item() if t.numel() > 0 else 0.5

        if cfg_scale <= 1.0 or not (cfg_interval[0] <= t_scalar <= cfg_interval[1]):
            return self.forward(
                x, t, total_view,
                z_ref_clean=z_ref_clean,
                plucker_6d=plucker_6d,
                ref_global=ref_global,
                camera_embedding=camera_embedding,
                viewmats=viewmats,
                Ks=Ks,
                **kwargs,
            )

        cam_kw = dict(
            camera_embedding=camera_embedding, viewmats=viewmats, Ks=Ks,
        )
        vel_cond = self.forward(
            x, t, total_view,
            z_ref_clean=z_ref_clean,
            plucker_6d=plucker_6d,
            ref_global=ref_global,
            **cam_kw,
            **kwargs,
        )

        if self.uses_baseline_camera() and camera_embedding is not None:
            from .camera_baseline import make_uncond_baseline_camera
            u_cam, u_vm, u_Ks = make_uncond_baseline_camera(
                camera_embedding, viewmats, Ks,
            )
            cam_kw = dict(camera_embedding=u_cam, viewmats=u_vm, Ks=u_Ks)

        vel_uncond = self.forward(
            x, t, total_view,
            z_ref_clean=None,
            plucker_6d=plucker_6d,
            ref_global=None,
            **cam_kw,
            **kwargs,
        )

        return vel_uncond + cfg_scale * (vel_cond - vel_uncond)
