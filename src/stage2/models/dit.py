"""GAEFlow — the Stage 2 conditional flow model (paper Section 3.2-3.3).

A DDT-style transformer over the standardized GAE latent. It predicts the clean
latent from the noisy state (RAEv2-style x-prediction, paper Eq. 11) and is
converted to a velocity by :mod:`src.stage2.transport.flow`.

Three conditioning signals, all of which specify the state without evolving
with it:

* ``z_ref_clean`` — clean latent tokens of the *observed* reference views,
  prepended as conditioning tokens (paper Eq. 14, "evidence rather than
  state"). They attend but their outputs are discarded before the prediction
  head, so only generated views are integrated by the ODE.
* ``plucker_6d``  — per-token metric Plucker ray embedding injected into
  query/key self-attention (paper Eq. 13).
* ``ref_global``  — text, via cross-attention.

Inherits from :class:`TokenConcatDDT` (its token-concat conditioning is kept
intact). Adds:

  1. ``prediction_type`` flag (``"velocity"`` / ``"x"``). Just a metadata
     hint the trainer/transport reads; the network's final_layer still
     outputs raw values — interpretation is the transport's job.
  2. ``base_final_layer``: a zero-initialized DDT-style final head that
     reads view-token activations at encoder layer ``base_model_depth``,
     unpatchifies to ``(BV, C, H, W)``. Plays the "weak baseline" role for
     Internal Guidance (Eq. in RAEv2 §2.3 + Tab. 4).
  3. ``forward()`` returns a dict so the trainer can consume main + base +
     repa outputs without a tuple-vs-tensor branch. ``forward_with_cfg`` is
     reused from TokenConcatDDT unchanged (CFG remains available; IG/REPA-G are evaluated
     by a separate sampler path, not via ``forward_with_cfg``).

Why subclass TokenConcatDDT instead of a pure-T2I IG head: it already implements
plücker / ref_key_scale / cross-attn / token-concat conditioning that the
GLD I2V/T2V path needs. A pure-T2I IG model uses adaLN cond
tokens — incompatible with the rest of the GLD ecosystem. Token-level IG
hook on top of TokenConcatDDT keeps weight reuse straightforward.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint as grad_ckpt

from .DDT import DDTFinalLayer
from .token_concat_ddt import TokenConcatDDT
from .temporal_rope import build_temporal_frame_idx


class GAEFlow(TokenConcatDDT):
    """Conditional flow model over the GAE latent (x-prediction + IG baseline).

    All TokenConcatDDT args pass through unchanged; new ones:

    Args:
        prediction_type: ``"x"`` or ``"velocity"``. Metadata only — does NOT
            change the network. Transport reads it to decide
            ``convert_model_pred``.
        base_model_depth: Encoder layer index (0-based) where IG baseline
            forks. ``None`` disables the baseline head (then IG eval falls
            back to identity = main output).
    """

    def __init__(
        self,
        *,
        prediction_type: str = "x",
        base_model_depth: Optional[int] = 8,
        self_flow_enable: bool = False,
        self_flow_student_depth: int = 8,
        self_flow_teacher_depth: int = 20,
        self_flow_proj_hidden: int = 2048,
        **kwargs,
    ):
        if prediction_type not in ("x", "velocity"):
            raise ValueError(
                f"prediction_type must be 'x' or 'velocity'; got {prediction_type!r}"
            )
        super().__init__(**kwargs)
        self.prediction_type = prediction_type

        # Self-Flow: a lightweight student projection head predicts a deeper
        # EMA-teacher encoder representation.  It is deliberately independent
        # from REPA: REPA targets an external feature space whereas Self-Flow
        # targets this model's own encoder feature space.
        self.self_flow_enable = bool(self_flow_enable)
        self.self_flow_student_depth = int(self_flow_student_depth)
        self.self_flow_teacher_depth = int(self_flow_teacher_depth)
        self._self_flow_pred: Optional[torch.Tensor] = None
        if self.self_flow_enable:
            if not (1 <= self.self_flow_student_depth <= self.num_encoder_blocks):
                raise ValueError(
                    "self_flow_student_depth must be in [1, num_encoder_blocks], got "
                    f"{self.self_flow_student_depth}"
                )
            if not (1 <= self.self_flow_teacher_depth <= self.num_encoder_blocks):
                raise ValueError(
                    "self_flow_teacher_depth must be in [1, num_encoder_blocks], got "
                    f"{self.self_flow_teacher_depth}"
                )
            if self.self_flow_student_depth >= self.self_flow_teacher_depth:
                raise ValueError(
                    "Self-Flow requires a shallower student feature than teacher feature; "
                    f"got {self.self_flow_student_depth} >= {self.self_flow_teacher_depth}"
                )
            self.self_flow_projector = nn.Sequential(
                nn.Linear(self.encoder_hidden_size, self_flow_proj_hidden),
                nn.SiLU(),
                nn.Linear(self_flow_proj_hidden, self_flow_proj_hidden),
                nn.SiLU(),
                nn.Linear(self_flow_proj_hidden, self.encoder_hidden_size),
            )
        else:
            self.self_flow_projector = None

        self.base_model_depth = base_model_depth
        self._base_pred: Optional[torch.Tensor] = None  # populated each forward

        if base_model_depth is not None:
            if not (0 <= base_model_depth < self.num_encoder_blocks):
                raise ValueError(
                    f"base_model_depth={base_model_depth} out of encoder range "
                    f"[0,{self.num_encoder_blocks})"
                )
            # Encoder-side final head: hidden = encoder_hidden_size,
            # patch=s_patch_size, channels=in_channels (same x-pred space).
            # Inherit norm class from the existing decoder-side final_layer
            # so the two heads stay structurally consistent.
            inherits_rmsnorm = (
                self.final_layer.norm_final.__class__.__name__ == "RMSNorm"
            )
            self.base_final_layer = DDTFinalLayer(
                hidden_size=self.encoder_hidden_size,
                patch_size=self.s_patch_size,
                out_channels=self.in_channels,
                use_rmsnorm=inherits_rmsnorm,
            )
            # Zero-init so day-1 IG baseline ≈ 0 → does not perturb main loss.
            nn.init.constant_(self.base_final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.base_final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.base_final_layer.linear.weight, 0)
            nn.init.constant_(self.base_final_layer.linear.bias, 0)
        else:
            self.base_final_layer = None

    # ------------------------------------------------------------------
    # Encoder-side unpatchify mirrors DDTHead's main unpatchify but uses the
    # encoder patch size (s_patch_size). Handles the H≠W case.
    # ------------------------------------------------------------------
    def _unpatchify_enc(self, tokens: torch.Tensor, Hs: int, Ws: int) -> torch.Tensor:
        c = self.in_channels
        p = self.s_patch_size
        assert tokens.shape[1] == Hs * Ws, (
            f"_unpatchify_enc token mismatch: {tokens.shape[1]} vs {Hs*Ws}"
        )
        tokens = tokens.reshape(tokens.shape[0], Hs, Ws, p, p, c)
        tokens = torch.einsum("nhwpqc->nchpwq", tokens)
        return tokens.reshape(tokens.shape[0], c, Hs * p, Ws * p)

    # ------------------------------------------------------------------
    # Forward — duplicates TokenConcatDDT.forward almost verbatim with two additions:
    #   (a) at base_model_depth, snapshot view tokens for IG baseline head
    #   (b) returns a dict {"main", "base", "repa"} instead of bare tensor
    # ------------------------------------------------------------------
    def forward(  # type: ignore[override]
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
        return_dict: bool = False,
        self_flow_capture_depth: Optional[int] = None,
        return_self_flow_feature_only: bool = False,
        **kwargs,
    ):
        BV = x.shape[0]
        assert BV % total_view == 0
        B = BV // total_view
        cond_num = kwargs.get("cond_num", 1)
        view_frame_idx = kwargs.get("view_frame_idx", None)
        self._repa_pred = None
        self._base_pred = None
        self._self_flow_pred = None

        if self_flow_capture_depth is not None and not (
            1 <= int(self_flow_capture_depth) <= self.num_encoder_blocks
        ):
            raise ValueError(
                "self_flow_capture_depth must be in [1, num_encoder_blocks], got "
                f"{self_flow_capture_depth}"
            )

        # CFG uncond branch (z_ref_clean=None) collapses the encoder to V views,
        # but baseline_prope camera inputs were cond-extended to K+V. Slice the
        # cond prefix off all camera tensors so encoder/decoder both see V views.
        camera_embedding, viewmats, Ks = self._maybe_slice_baseline_camera(
            camera_embedding, viewmats, Ks, B, total_view, cond_num, z_ref_clean,
        )

        Hs = x.shape[2] // self.s_patch_size
        Ws = x.shape[3] // self.s_patch_size
        Hx = x.shape[2] // self.x_patch_size
        Wx = x.shape[3] // self.x_patch_size

        # Standard training supplies one timestep per view (BV,).  Self-Flow
        # supplies one timestep per spatial token (BV,H,W) / (BV,N).  The DDT
        # blocks already support per-token AdaLN conditioning, so only this
        # embedding path needs to distinguish the two forms.
        tokenwise_t = t.ndim > 1
        if tokenwise_t:
            t_tokens = t.reshape(BV, -1)
            if t_tokens.shape[1] != Hs * Ws:
                raise ValueError(
                    "Per-token timestep count must match encoder token count; got "
                    f"{t_tokens.shape[1]} vs {Hs * Ws}"
                )
            t_emb = self.t_embedder(t_tokens.reshape(-1)).reshape(
                BV, t_tokens.shape[1], self.encoder_hidden_size
            )
        else:
            t_emb = self.t_embedder(t)
        c = nn.functional.silu(t_emb)
        if z_ref_clean is not None:
            t_cond_shape = (
                (B * cond_num, Hs * Ws) if tokenwise_t else (B * cond_num,)
            )
            t_cond = torch.zeros(t_cond_shape, device=t.device, dtype=t.dtype)
            if tokenwise_t:
                t_cond_emb = self.t_embedder(t_cond.reshape(-1)).reshape(
                    B * cond_num, Hs * Ws, self.encoder_hidden_size
                )
            else:
                t_cond_emb = self.t_embedder(t_cond)
            c_cond = nn.functional.silu(t_cond_emb)

        if self.text_proj is not None and ref_global is not None:
            ref_global = self.text_proj(ref_global)

        if s is None:
            x_5d = rearrange(x, "(b v) c h w -> b v c h w", v=total_view)
            x_flat = x_5d.reshape(-1, *x_5d.shape[2:])
            s_views = self.s_embedder_tgt(self.s_proj_tgt(x_flat))

            if z_ref_clean is not None:
                s_cond = self.s_embedder_ref(self.s_proj_ref(z_ref_clean))
                s_views_5d = rearrange(s_views, "(b v) n d -> b v n d", v=total_view)
                s_cond_5d = rearrange(s_cond, "(b k) n d -> b k n d", k=cond_num)
                s_all = torch.cat([s_cond_5d, s_views_5d], dim=1)
                s_all = rearrange(s_all, "b v n d -> (b v) n d")
                total_enc_views = cond_num + total_view

                if tokenwise_t:
                    c_views_5d = rearrange(c, "(b v) n d -> b v n d", v=total_view)
                    c_cond_5d = rearrange(c_cond, "(b k) n d -> b k n d", k=cond_num)
                    c_all = torch.cat([c_cond_5d, c_views_5d], dim=1)
                    c_all = rearrange(c_all, "b v n d -> (b v) n d")
                else:
                    c_views_5d = rearrange(c, "(b v) d -> b v d", v=total_view)
                    c_cond_5d = rearrange(c_cond, "(b k) d -> b k d", k=cond_num)
                    c_all = torch.cat([c_cond_5d, c_views_5d], dim=1)
                    c_all = rearrange(c_all, "b v d -> (b v) d")
            else:
                s_all = s_views
                c_all = c
                total_enc_views = total_view

            if self.use_pos_embed:
                pe_s = self._get_sincos_pos(
                    self.encoder_hidden_size, Hs, Ws, s_all.device, s_all.dtype)
                s_all = s_all + pe_s

            enc_rope = None
            if self.use_rope:
                enc_rope = self._get_rope(
                    self._enc_rope_cache, self.enc_half_head_dim,
                    Hs, Ws, s_all.device, s_all.dtype)

            enc_plucker = None
            if plucker_6d is not None:
                if z_ref_clean is not None:
                    plucker_5d = rearrange(plucker_6d, "b (v n) d -> b v n d",
                                           v=total_view, n=Hs * Ws)
                    plucker_cond = plucker_5d[:, :cond_num]
                    plucker_all = torch.cat([plucker_cond, plucker_5d], dim=1)
                    enc_plucker = rearrange(plucker_all, "b v n d -> b (v n) d")
                else:
                    enc_plucker = plucker_6d

            s_all = self._apply_baseline_camera_embed(s_all, camera_embedding)

            # Temporal RoPE frame index: prepended cond views reuse the frame
            # index of the target view they replicate (cond_k ↔ tgt view_k), so
            # the same physical frame is not split across two ordinal phases.
            enc_frame_idx = None
            if getattr(self, "_temporal_cfg", None) is not None:
                enc_frame_idx = build_temporal_frame_idx(
                    total_view, cond_num, z_ref_clean is not None, s_all.device,
                    view_frame_idx=view_frame_idx,
                )

            cam_kwargs = self._camera_attention_kwargs(
                plucker_6d=enc_plucker, viewmats=viewmats, Ks=Ks, Hs=Hs, Ws=Ws,
            )
            for i in range(self.num_encoder_blocks):
                cur_pag = pag_mode if (pag_layer_idx is None or i == pag_layer_idx) else False
                block_kwargs = dict(
                    total_view=total_enc_views,
                    cond_num=cond_num,
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

                # REPA: project view tokens of this layer into target dim.
                if self.repa_enable and i == self.repa_align_depth:
                    if z_ref_clean is not None:
                        v_tokens = rearrange(
                            s_all, "(b v) n d -> b v n d", v=total_enc_views,
                        )[:, cond_num:]
                        v_tokens = rearrange(v_tokens, "b v n d -> (b v) n d")
                    else:
                        v_tokens = s_all
                    self._repa_pred = self.repa_projector(v_tokens)

                # Capture a view-only encoder feature after exactly N blocks.
                # The EMA teacher uses `return_self_flow_feature_only=True` to
                # stop here, avoiding a needless decoder pass.
                if self_flow_capture_depth is not None and (i + 1) == int(self_flow_capture_depth):
                    if z_ref_clean is not None:
                        sf_tokens = rearrange(
                            s_all, "(b v) n d -> b v n d", v=total_enc_views,
                        )[:, cond_num:]
                        sf_tokens = rearrange(sf_tokens, "b v n d -> (b v) n d")
                    else:
                        sf_tokens = s_all
                    if return_self_flow_feature_only:
                        return sf_tokens
                    if self.self_flow_projector is not None:
                        self._self_flow_pred = self.self_flow_projector(sf_tokens)

                # IG baseline: snapshot view tokens at base_model_depth,
                # run them through the encoder-side final head + adaLN(t).
                if (
                    self.base_final_layer is not None
                    and i == self.base_model_depth
                ):
                    if z_ref_clean is not None:
                        v_tokens_b = rearrange(
                            s_all, "(b v) n d -> b v n d", v=total_enc_views,
                        )[:, cond_num:]
                        v_tokens_b = rearrange(v_tokens_b, "b v n d -> (b v) n d")
                        if c_all.ndim == 3:
                            c_views = rearrange(
                                c_all, "(b v) n d -> b v n d", v=total_enc_views
                            )[:, cond_num:]
                            c_views = rearrange(c_views, "b v n d -> (b v) n d")
                        else:
                            c_views = rearrange(
                                c_all, "(b v) d -> b v d", v=total_enc_views
                            )[:, cond_num:]
                            c_views = rearrange(c_views, "b v d -> (b v) d")
                    else:
                        v_tokens_b = s_all
                        c_views = c_all
                    base_tokens = self.base_final_layer(v_tokens_b, c_views)
                    self._base_pred = self._unpatchify_enc(base_tokens, Hs, Ws)

            t_broadcast = (
                c_all
                if c_all.ndim == 3
                else c_all.unsqueeze(1).expand(-1, s_all.shape[1], -1)
            )
            s_all = nn.functional.silu(t_broadcast + s_all)

            if z_ref_clean is not None:
                s_all_5d = rearrange(s_all, "(b v) n d -> b v n d", v=total_enc_views)
                s = rearrange(s_all_5d[:, cond_num:], "b v n d -> (b v) n d")
            else:
                s = s_all

        s = self.s_projector(s)

        x_5d = rearrange(x, "(b v) c h w -> b v c h w", v=total_view)
        x_flat = rearrange(x_5d, "b v c h w -> (b v) c h w")
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

        # Decoder runs on V view tokens only (cond tokens dropped after encoder).
        # baseline_prope cond-extends viewmats/Ks to K+V for the encoder, so the
        # decoder must slice the cond prefix off or PRoPE trips its
        # `seqlen == cameras*patches` assert (seqlen=V*N vs cameras=K+V).
        dec_viewmats, dec_Ks = viewmats, Ks
        if (
            self.uses_baseline_camera()
            and viewmats is not None
            and viewmats.shape[1] == total_view + cond_num
        ):
            dec_viewmats = viewmats[:, cond_num:]
            dec_Ks = Ks[:, cond_num:] if Ks is not None else None

        # Decoder sees only the V target views (cond dropped) → frame index is
        # the plain view order 0..V-1.
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

        # DDP find_unused_parameters compat: route REPA + base preds through
        # the loss graph with zero coefficient so DDP marks their parameters
        # as used. Trainer consumes them via the dict (or attributes).
        if self.training:
            if self._repa_pred is not None:
                out = out + 0.0 * self._repa_pred.sum().to(out.dtype)
            if self._base_pred is not None:
                out = out + 0.0 * self._base_pred.sum().to(out.dtype)
            if self._self_flow_pred is not None:
                out = out + 0.0 * self._self_flow_pred.sum().to(out.dtype)

        if return_dict:
            return {
                "main": out,
                "base": self._base_pred,
                "repa": self._repa_pred,
                "self_flow": self._self_flow_pred,
            }
        return out
