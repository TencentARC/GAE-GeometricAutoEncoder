from math import sqrt
import torch
import torch.nn as nn
from einops import rearrange
import torch
from torch import nn
from .lightningDiT import PatchEmbed, Mlp, NormAttention
from timm.models.vision_transformer import PatchEmbed, Mlp
from .model_utils import VisionRotaryEmbeddingFast, RMSNorm, SwiGLUFFN, GaussianFourierEmbedding, LabelEmbedder, NormAttention, get_2d_sincos_pos_embed
import torch.nn.functional as F
from typing import *


def DDTModulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        shift: Tensor of shape (B, L, D)
        scale: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * (1 + scale) + shift, 
        with shift and scale repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * (1 + scale) + shift


def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        gate: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * gate, 
        with gate repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        # print(f"gate shape: {gate.shape}, x shape: {x.shape}")
        gate = gate.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * gate


class LightningDDTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_prope=False,
        **block_kwargs
    ):
        super().__init__()
        self.use_prope = use_prope

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, total_view, feat_rope=None, pag_mode=False, prope_image_size=None, patches_layout=None, num_prefix_tokens=0, **kwargs):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)  # (B, 1, C)
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(6, dim=-1)
        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x),
                        shift_msa, scale_msa), rope=feat_rope, total_view=total_view, use_prope=self.use_prope, viewmats=kwargs.get('viewmats'), Ks=kwargs.get('Ks'), pag_mode=pag_mode, prope_image_size=prope_image_size, patches_layout=patches_layout, num_prefix_tokens=num_prefix_tokens), gate_msa)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x),
                        shift_mlp, scale_mlp)), gate_mlp)
        return x


class DDTFinalLayer(nn.Module):
    """
    The final layer of DDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)  # no gate
        x = self.linear(x)
        return x


class DiTwDDTHead(nn.Module):
    def __init__(
            self,
            input_size: int = 1,
            patch_size: Union[list, int] = 1,
            in_channels: int = 768,
            hidden_size=[1152, 2048],
            depth=[28, 2],
            num_heads: Union[list[int], int] = [16, 16],
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False,
            use_pos_embed: bool = True,
            use_prope: bool = False,
            # Camera embedder parameters (for different resolutions)
            cam_input_size: int = 256,  # 256 for DINO, 518 for DA3
            cam_patch_size: int = 16,   # 16 for DINO, 14 for DA3
            cam_in_channels: int = 7,   # 7 for plucker (1+6), 4 for camray (1+3)
            level: int = 3, # Feature level (0, 1, 2, 3)
            predict_cls: bool = False,
            is_concat_mode: bool = False,  # If True, input has 2*in_channels (condition + noisy)
            source_condition_mode: str = None,  # NEW: "l1_to_x0" | "l1_as_cond" | None
            architecture_mode: str = "new", # "new" (split embedders) or "old" (single embedders)
            cfg_mode: str = "new",          # "new" (drop pose/intrinsics) or "old" (keep pose/intrinsics)
            num_special_tokens: int = 0,    # number of special tokens (e.g. 5 for VGGT camera+register)
    ):
        super().__init__()
        self.level = level
        self.use_prope = use_prope
        self.predict_cls = predict_cls
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.is_concat_mode = is_concat_mode
        self.source_condition_mode = source_condition_mode
        self.architecture_mode = architecture_mode
        self.cfg_mode = cfg_mode
        self.num_special_tokens = num_special_tokens

        # In concat mode, input has 2*in_channels: [condition | noisy]
        embed_in_channels = in_channels * 2 if is_concat_mode else in_channels

        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]
        self.num_heads = [num_heads, num_heads] if isinstance(
            num_heads, int) else list(num_heads)
        self.num_decoder_blocks = depth[1]
        self.num_encoder_blocks = depth[0]
        self.num_blocks = depth[0] + depth[1]
        self.use_rope = use_rope
        # analyze patch size
        if isinstance(patch_size, int) or isinstance(patch_size, float):
            patch_size = [patch_size, patch_size]  # patch size for s , x embed
        assert len(
            patch_size) == 2, f"patch size should be a list of two numbers, but got {patch_size}"
        self.patch_size = patch_size
        self.s_patch_size = patch_size[0]
        self.x_patch_size = patch_size[1]
        # Use embed_in_channels (doubled in concat mode) for input embedders
        s_channel_per_token = embed_in_channels * self.s_patch_size * self.s_patch_size
        s_input_size = input_size
        s_patch_size = self.s_patch_size
        x_input_size = input_size
        x_patch_size = self.x_patch_size
        x_channel_per_token = embed_in_channels * self.x_patch_size * self.x_patch_size
        if self.architecture_mode == "new":
            # Separate Ref/Tgt embedders (NEW Architecture)
            self.x_embedder_ref = PatchEmbed(
                x_input_size, x_patch_size, x_channel_per_token, self.decoder_hidden_size, bias=True, strict_img_size=False)
            self.x_embedder_tgt = PatchEmbed(
                x_input_size, x_patch_size, x_channel_per_token, self.decoder_hidden_size, bias=True, strict_img_size=False)
            
            self.s_embedder_ref = PatchEmbed(
                s_input_size, s_patch_size, s_channel_per_token, self.encoder_hidden_size, bias=True, strict_img_size=False)
            self.s_embedder_tgt = PatchEmbed(
                s_input_size, s_patch_size, s_channel_per_token, self.encoder_hidden_size, bias=True, strict_img_size=False)
        else:
            # Single Embedders (OLD Architecture)
            self.x_embedder = PatchEmbed(
                x_input_size, x_patch_size, x_channel_per_token, self.decoder_hidden_size, bias=True, strict_img_size=False)
            self.s_embedder = PatchEmbed(
                s_input_size, s_patch_size, s_channel_per_token, self.encoder_hidden_size, bias=True, strict_img_size=False)
            
            # Aliases for consistent usage in init (e.g. num_patches)
            self.s_embedder_ref = self.s_embedder
            self.x_embedder_ref = self.x_embedder
            # Only alias if needed, but for safety in shared logic accessing ref/tgt:
            self.s_embedder_tgt = self.s_embedder
            self.x_embedder_tgt = self.x_embedder

        
        self.s_channel_per_token = s_channel_per_token
        # Output uses original in_channels (not doubled)
        self.x_channel_per_token = in_channels * self.x_patch_size * self.x_patch_size

        # CLS token embedders
        if self.predict_cls:
            self.cls_embedder = nn.Linear(in_channels, self.encoder_hidden_size)
            self.x_cls_embedder = nn.Linear(in_channels, self.decoder_hidden_size)
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, self.encoder_hidden_size))
            self.x_cls_pos_embed = nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_size))

        # Special token embedders (for VGGT camera + register tokens)
        K = num_special_tokens
        if K > 0:
            assert not predict_cls, "num_special_tokens and predict_cls are mutually exclusive"

            # Encoder: Linear embedder for special tokens (embed_in_channels → enc_hidden)
            self.special_s_embedder = nn.Linear(embed_in_channels, self.encoder_hidden_size)
            self.special_pos_embed_s = nn.Parameter(
                torch.zeros(1, K, self.encoder_hidden_size))

            # Decoder: Linear embedder for special tokens
            self.special_x_embedder = nn.Linear(embed_in_channels, self.decoder_hidden_size)
            self.special_pos_embed_x = nn.Parameter(
                torch.zeros(1, K, self.decoder_hidden_size))

            # Output projection: dec_hidden → C (feature space)
            self.special_output_proj = nn.Linear(self.decoder_hidden_size, in_channels)

        self.s_projector = nn.Linear(
            self.encoder_hidden_size, self.decoder_hidden_size) if self.encoder_hidden_size != self.decoder_hidden_size else nn.Identity()
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        # Camera embedder - configurable for different image resolutions and modes
        self.camera_embedder = PatchEmbed(
            cam_input_size, cam_patch_size, cam_in_channels, self.encoder_hidden_size, bias=True, strict_img_size=False)
        # Store image size for ProPE (avoid hardcoding in attention)
        # self.prope_image_size = cam_input_size
        # self.y_embedder = LabelEmbedder( 
        #     num_classes, self.encoder_hidden_size, class_dropout_prob) # minkyung removed y_embedder (laebl embedder)
        # print(f"x_channel_per_token: {x_channel_per_token}, s_channel_per_token: {s_channel_per_token}")
        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size, 1, self.x_channel_per_token, use_rmsnorm=use_rmsnorm)
        # Will use fixed sin-cos embedding:
        if use_pos_embed:
            num_patches = self.s_embedder_ref.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(
                1, num_patches, self.encoder_hidden_size), requires_grad=False)
            self.x_pos_embed = nn.Parameter(torch.zeros(
                1, num_patches, self.decoder_hidden_size), requires_grad=False)
        self.use_pos_embed = use_pos_embed
        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]
        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            enc_half_head_dim = self.encoder_hidden_size // enc_num_heads // 2
            hw_seq_len = int(sqrt(self.s_embedder_ref.num_patches))
            # print(f"enc_half_head_dim: {enc_half_head_dim}, hw_seq_len: {hw_seq_len}")
            self.enc_feat_rope = VisionRotaryEmbeddingFast(
                dim=enc_half_head_dim,
                pt_seq_len=hw_seq_len,
            )
            dec_half_head_dim = self.decoder_hidden_size // dec_num_heads // 2
            hw_seq_len = int(sqrt(self.x_embedder_ref.num_patches))
            # print(f"dec_half_head_dim: {dec_half_head_dim}, hw_seq_len: {hw_seq_len}")
            self.dec_feat_rope = VisionRotaryEmbeddingFast(
                dim=dec_half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.enc_feat_rope = None
            self.dec_feat_rope = None
            
        self._enc_rope_cache = {}
        self._dec_rope_cache = {}
        self.enc_half_head_dim = self.encoder_hidden_size // enc_num_heads // 2 if self.use_rope else 0
        self.dec_half_head_dim = self.decoder_hidden_size // dec_num_heads // 2 if self.use_rope else 0
        self.blocks = nn.ModuleList([
            LightningDDTBlock(self.encoder_hidden_size if i < self.num_encoder_blocks else self.decoder_hidden_size,
                              enc_num_heads if i < self.num_encoder_blocks else dec_num_heads,
                              mlp_ratio=mlp_ratio,
                              use_qknorm=use_qknorm,
                              use_rmsnorm=use_rmsnorm,
                              use_swiglu=use_swiglu,
                              wo_shift=wo_shift,
                              use_prope=use_prope,
                              ) for i in range(self.num_blocks)
        ])
        self.initialize_weights()

    def initialize_weights(self, xavier_uniform_init: bool = False):
        if xavier_uniform_init:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        if self.architecture_mode == "new":
            for embedder in [self.x_embedder_ref, self.x_embedder_tgt, self.s_embedder_ref, self.s_embedder_tgt]:
                w = embedder.proj.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                nn.init.constant_(embedder.proj.bias, 0)
        else:
            # Initialize old embedders
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)
            
            w = self.s_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.s_embedder.proj.bias, 0)
        
        # minkyung: added init for camera_embedder
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.camera_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.camera_embedder.proj.bias, 0)

        # minkyung: added init for CLS embedder
        if self.predict_cls:
            nn.init.xavier_uniform_(self.cls_embedder.weight)
            nn.init.constant_(self.cls_embedder.bias, 0)
            nn.init.xavier_uniform_(self.x_cls_embedder.weight)
            nn.init.constant_(self.x_cls_embedder.bias, 0)
            nn.init.normal_(self.cls_pos_embed, std=0.02)
            nn.init.normal_(self.x_cls_pos_embed, std=0.02)

        # Initialize special token embedders
        if self.num_special_tokens > 0:
            nn.init.xavier_uniform_(self.special_s_embedder.weight)
            nn.init.constant_(self.special_s_embedder.bias, 0)
            nn.init.xavier_uniform_(self.special_x_embedder.weight)
            nn.init.constant_(self.special_x_embedder.bias, 0)
            nn.init.normal_(self.special_pos_embed_s, std=0.02)
            nn.init.normal_(self.special_pos_embed_x, std=0.02)
            # Zero-init output projection for training stability
            nn.init.constant_(self.special_output_proj.weight, 0)
            nn.init.constant_(self.special_output_proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02) # minkyung: removed y_embedder
        if self.use_pos_embed:
            # Initialize (and freeze) pos_embed and x_pos_embed by sin-cos embedding:
            grid_size = int(self.s_embedder_ref.num_patches ** 0.5)
            
            pos_embed = get_2d_sincos_pos_embed(self.encoder_hidden_size, grid_size)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            
            x_pos_embed = get_2d_sincos_pos_embed(self.decoder_hidden_size, grid_size)
            self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _get_sincos_pos(self, embed_dim, H, W, device, dtype):
        # Generates PosEmbed with support for rectangular grids (H, W)
        pe = get_2d_sincos_pos_embed(embed_dim, (H, W))  # (L, D) numpy
        pe = torch.from_numpy(pe).to(device=device, dtype=dtype).unsqueeze(0)  # (1,L,D)
        return pe

    def _get_rope(self, cache, dim, H, W, device, dtype):
        # Checks cache for RoPE for this specific resolution (H, W)
        key = (device.type, device.index, str(dtype), dim, H, W)
        rope = cache.get(key, None)
        if rope is None:
            rope = VisionRotaryEmbeddingFast(dim=dim, pt_seq_len=(H, W))
            rope = rope.to(device=device)
            # Ensure buffers are cast to correct dtype if needed (buffers are usually float32)
            # But VisionRotaryEmbeddingFast registers buffers.
            cache[key] = rope
        return rope

    def unpatchify(self, x, h=None, w=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        # c = self.in_channels
        c = self.in_channels
        p = self.x_patch_size
        
        # Handle CLS token: if T is not square, assume first token is CLS
        if self.predict_cls:
            cls_out = x[:, 0]  # (N, C)
            x = x[:, 1:]       # (N, T-1, C)
            
        if h is None or w is None:
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1]
        else:
             assert h * w == x.shape[1], f"token count mismatch: {x.shape[1]} vs {h*w}"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        
        if self.predict_cls:
            return imgs, cls_out
        return imgs

    # def forward(self, x, t, y, s=None, mask=None):
    #     # x = self.x_embedder_ref(x) + self.pos_embed
    #     t = self.t_embedder(t)
    #     y = self.y_embedder(y, self.training)
    #     c = nn.functional.silu(t + y)
    #     if s is None:
    #         s = self.s_embedder_ref(x)
    #         if self.use_pos_embed:
    #             s = s + self.pos_embed
    #         # print(f"t shape: {t.shape}, y shape: {y.shape}, c shape: {c.shape}, s shape: {s.shape}, pos_embed shape: {self.pos_embed.shape}")
    #         for i in range(self.num_encoder_blocks):
    #             s = self.blocks[i](s, c, feat_rope=self.enc_feat_rope)
    #         # broadcast t to s
    #         t = t.unsqueeze(1).repeat(1, s.shape[1], 1)
    #         s = nn.functional.silu(t + s)
    #     s = self.s_projector(s)
    #     x = self.x_embedder_ref(x)
    #     if self.use_pos_embed and self.x_pos_embed is not None:
    #         x = x + self.x_pos_embed
    #     for i in range(self.num_encoder_blocks, self.num_blocks):
    #         x = self.blocks[i](x, s, feat_rope=self.dec_feat_rope)
    #     x = self.final_layer(x, s)
    #     x = self.unpatchify(x)
    #     return x

    def forward(self, x, t, total_view, camera_embedding, s=None, mask=None, pag_mode=False, pag_layer_idx=None, **kwargs):
        # Remove prope_image_size from kwargs if present to avoid duplicate argument error
        prope_image_size = kwargs.pop('prope_image_size', None)
        # Strict logic: 
        # 1. If old architecture, fallback to 252 (legacy behavior)
        # 2. If new architecture, prope_image_size MUST be provided (usually via eval script)
        if prope_image_size is None and self.architecture_mode == "old":
             prope_image_size = 252
        
        if self.use_prope and prope_image_size is None:
             raise ValueError("prope_image_size must be provided when use_prope is True in new architecture mode.")
        
        # Handle source_condition for Image 1 architecture (l1_as_cond mode)
        # source_condition contains L1 features used for encoder conditioning
        # May be packed (BV, C, K+N, 1) for VGGT or spatial (BV, C, H, W) for DA3
        source_condition = kwargs.pop('source_condition', None)
        
        BV = x.shape[0]
        
        # Determine expected number of patches from camera_embedding if available
        # This is more robust for dynamic resolutions (multi-resolution training)
        if camera_embedding is not None:
             # Assume patch_size is 14 for DA3 RAE; camera_embedder might have it
             cp = self.camera_embedder.patch_size if hasattr(self.camera_embedder, 'patch_size') else 14
             if isinstance(cp, (list, tuple)): cp = cp[0]
             
             H_grid = camera_embedding.shape[2] // cp
             W_grid = camera_embedding.shape[3] // cp
             n_expected = H_grid * W_grid
        else:
             n_expected = self.s_embedder_ref.num_patches
             H_grid = W_grid = int(n_expected**0.5)

        # Robustly detect if input has CLS token or special tokens
        # Expected sequence format: (BV, C, N_patches + 1, 1) or (BV, C, N_patches, 1)
        input_is_seq = (x.ndim == 4 and x.shape[3] == 1)
        has_cls = self.predict_cls and input_is_seq and (x.shape[2] == n_expected + 1)

        K = self.num_special_tokens
        has_special = (K > 0 and input_is_seq and x.shape[2] == n_expected + K)

        if has_special:
            # x: (BV, C_in, K+N, 1) where C_in = 2C in concat mode
            x_special = x[:, :, :K, 0].permute(0, 2, 1)   # (BV, K, C_in)
            x_patches_flat = x[:, :, K:, 0]                 # (BV, C_in, N)
            x_patches = x_patches_flat.reshape(BV, -1, H_grid, W_grid)
            x_cls = None
        elif has_cls:
            # x_patches: (BV, C, N_patches, 1) -> (BV, C, H, W)
            # x_cls: (BV, C, 1, 1) -> (BV, C)
            x_cls = x[:, :, 0, 0] # (BV, C)
            x_patches_flat = x[:, :, 1:, 0]     # (BV, C, N_patches)
            x_patches = x_patches_flat.reshape(BV, -1, H_grid, W_grid)
            x_special = None
        elif input_is_seq:
            # Even if no CLS, we need to reshape if it is sequence format
            x_patches = x[:, :, :, 0].reshape(BV, -1, H_grid, W_grid)
            x_cls = None
            x_special = None
        else:
            x_patches = x
            x_cls = None
            x_special = None

        # Calculate dynamic grid sizes
        Hs = x_patches.shape[2] // self.s_patch_size
        Ws = x_patches.shape[3] // self.s_patch_size
        
        Hx = x_patches.shape[2] // self.x_patch_size
        Wx = x_patches.shape[3] // self.x_patch_size
        
        t = self.t_embedder(t)
        c = nn.functional.silu(t)
        
        if s is None:
            # Handle source_condition for Image 1 architecture (l1_as_cond mode)
            # When source_condition is provided, it contains L1 features to be used as encoder input
            # This replaces the default behavior of using x_patches (which contains [cond | noise])
            if source_condition is not None and self.source_condition_mode == "l1_as_cond":
                # Image 1: Encoder sees [ref_cond | L1_features], Decoder sees [ref_cond | noise]
                C = self.in_channels
                cond_part = x_patches[:, :C]  # Reference condition (C channels)

                # Unpack source_condition if packed (VGGT special tokens)
                sc = source_condition
                if sc.ndim == 4 and sc.shape[3] == 1:
                    # Packed: (BV, C, K+N, 1) → strip special tokens → reshape
                    sc_flat = sc[:, :, K:, 0] if K > 0 else sc[:, :, :, 0]
                    sc = sc_flat.reshape(BV, -1, H_grid, W_grid)

                encoder_input = torch.cat([cond_part, sc], dim=1)
                
                # Separate Ref/Tgt embedding for source condition
                cond_num = kwargs.get("cond_num", total_view // 2) # Default if not provided
                ei_5d = rearrange(encoder_input, "(b v) c h w -> b v c h w", v=total_view)
                ei_ref = ei_5d[:, :cond_num].reshape(-1, *ei_5d.shape[2:])
                ei_tgt = ei_5d[:, cond_num:].reshape(-1, *ei_5d.shape[2:])
                
                s_ref = self.s_embedder_ref(ei_ref)
                s_tgt = self.s_embedder_tgt(ei_tgt)
                s = torch.cat([rearrange(s_ref, "(b v) n d -> b v n d", v=cond_num),
                               rearrange(s_tgt, "(b v) n d -> b v n d", v=total_view - cond_num)], dim=1)
                s = rearrange(s, "b v n d -> (b v) n d")
            elif self.architecture_mode == "new":
                cond_num = kwargs.get("cond_num", total_view // 2)
                xp_5d = rearrange(x_patches, "(b v) c h w -> b v c h w", v=total_view)
                xp_ref = xp_5d[:, :cond_num].reshape(-1, *xp_5d.shape[2:])
                xp_tgt = xp_5d[:, cond_num:].reshape(-1, *xp_5d.shape[2:])
                
                s_ref = self.s_embedder_ref(xp_ref)
                s_tgt = self.s_embedder_tgt(xp_tgt)
                s = torch.cat([rearrange(s_ref, "(b v) n d -> b v n d", v=cond_num),
                               rearrange(s_tgt, "(b v) n d -> b v n d", v=total_view - cond_num)], dim=1)
                s = rearrange(s, "b v n d -> (b v) n d")
            else:
                # OLD Architecture (Single Embedder)
                s = self.s_embedder(x_patches)

            
            # Dynamic Positional Embedding for s
            if self.use_pos_embed:
                pos_embed_s = self._get_sincos_pos(self.encoder_hidden_size, Hs, Ws, s.device, s.dtype)
            
            if has_cls:
                s_cls = self.cls_embedder(x_cls).unsqueeze(1)
                # CLS pos embed is ALWAYS learnable and used if predict_cls is True
                s_cls = s_cls + self.cls_pos_embed

                if self.use_pos_embed:
                    s_patches = s + pos_embed_s
                    s = torch.cat([s_cls, s_patches], dim=1)
                else:
                    s = torch.cat([s_cls, s], dim=1)
            elif self.use_pos_embed:
                s = s + pos_embed_s

            # Camera embedding is spatial, apply to patches only.
            cam_emb = self.camera_embedder(camera_embedding)

            # Verify camera token count
            # Note: special tokens are prepended AFTER this check (line ~671),
            # so only CLS (if present) is already in s at this point.
            Lc = cam_emb.shape[1]
            n_prefix_in_s = 1 if has_cls else 0  # special tokens not yet in s
            Ls_patches = s.shape[1] - n_prefix_in_s
            if Lc != Ls_patches:
                 raise ValueError(
                    f"camera tokens mismatch: Lc={Lc} vs Ls_patches={Ls_patches}. "
                    f"Camera embedding must match source patch tokens count. "
                    f"Currently assuming (Hc, Wc) == (Hs, Ws) after patch embedding."
                )

            if has_special:
                # Embed special tokens for encoder
                special_s = self.special_s_embedder(x_special)   # (BV, K, enc_hidden)
                special_s = special_s + self.special_pos_embed_s
                # Camera embedding → patches only (not special tokens)
                s = s + cam_emb
                # Prepend special tokens before patches
                s = torch.cat([special_s, s], dim=1)  # (BV, K+N_tok, enc_hidden)
            elif has_cls:
                # Add cam_emb to patches part of s
                # s: (BV, 1+N_patches, D)
                s[:, 1:] = s[:, 1:] + cam_emb
            else:
                s = s + cam_emb

            # Get dynamic RoPE for Encoder
            enc_rope = None
            if self.use_rope:
                enc_rope = self._get_rope(self._enc_rope_cache, self.enc_half_head_dim, Hs, Ws, s.device, s.dtype)

            for i in range(self.num_encoder_blocks):
                # Apply pag_mode only if current layer matches pag_layer_idx
                current_pag_mode = pag_mode if (pag_layer_idx is None or i == pag_layer_idx) else False
                s = self.blocks[i](s, c, feat_rope=enc_rope, total_view=total_view, pag_mode=current_pag_mode, prope_image_size=prope_image_size, patches_layout=(Hs, Ws), num_prefix_tokens=K if has_special else 0, **kwargs)
            # broadcast t to s
            t = t.unsqueeze(1).repeat(1, s.shape[1], 1)
            s = nn.functional.silu(t + s)
            
        s = self.s_projector(s)
        
        if self.architecture_mode == "new":
            # Separate Ref/Tgt embedding for x
            cond_num = kwargs.get("cond_num", total_view // 2)
            xp_5d = rearrange(x_patches, "(b v) c h w -> b v c h w", v=total_view)
            xp_ref = xp_5d[:, :cond_num].reshape(-1, *xp_5d.shape[2:])
            xp_tgt = xp_5d[:, cond_num:].reshape(-1, *xp_5d.shape[2:])

            x_toks_ref = self.x_embedder_ref(xp_ref)
            x_toks_tgt = self.x_embedder_tgt(xp_tgt)
            x_toks = torch.cat([rearrange(x_toks_ref, "(b v) n d -> b v n d", v=cond_num),
                                rearrange(x_toks_tgt, "(b v) n d -> b v n d", v=total_view - cond_num)], dim=1)
            x_toks = rearrange(x_toks, "b v n d -> (b v) n d")
        else:
            # OLD Architecture (Single Embedder)
            x_toks = self.x_embedder(x_patches)


        # Dynamic Positional Embedding for x
        if self.use_pos_embed:
             pos_embed_x = self._get_sincos_pos(self.decoder_hidden_size, Hx, Wx, x_toks.device, x_toks.dtype)

        if has_cls:
            x_cls_tok = self.x_cls_embedder(x_cls).unsqueeze(1)
            x_cls_tok = x_cls_tok + self.x_cls_pos_embed

            if self.use_pos_embed:
                # Use decoder-specific x_pos_embed (2048-dim)
                s_patches = x_toks + pos_embed_x
                x_toks = torch.cat([x_cls_tok, s_patches], dim=1)
            else:
                x_toks = torch.cat([x_cls_tok, x_toks], dim=1)
        elif self.use_pos_embed:
            # Use decoder-specific x_pos_embed (2048-dim) instead of pos_embed (768-dim)
            x_toks = x_toks + pos_embed_x

        # Prepend special tokens for decoder
        if has_special:
            special_x = self.special_x_embedder(x_special)   # (BV, K, dec_hidden)
            special_x = special_x + self.special_pos_embed_x
            x_toks = torch.cat([special_x, x_toks], dim=1)  # (BV, K+N_tok, dec_hidden)

        # Get dynamic RoPE for Decoder
        dec_rope = None
        if self.use_rope:
             dec_rope = self._get_rope(self._dec_rope_cache, self.dec_half_head_dim, Hx, Wx, x_toks.device, x_toks.dtype)

        for i in range(self.num_encoder_blocks, self.num_blocks):
            # Apply pag_mode only if current layer matches pag_layer_idx
            current_pag_mode = pag_mode if (pag_layer_idx is None or i == pag_layer_idx) else False
            x_toks = self.blocks[i](x_toks, s, feat_rope=dec_rope, total_view=total_view, pag_mode=current_pag_mode, prope_image_size=prope_image_size, patches_layout=(Hx, Wx), num_prefix_tokens=K if has_special else 0, **kwargs)

        x_toks = self.final_layer(x_toks, s)

        if has_special:
            # Split special tokens and patches from output
            special_out = x_toks[:, :K]    # (BV, K, dec_hidden)
            patches_out = x_toks[:, K:]    # (BV, N_tok, dec_hidden)

            # Project special tokens → feature space
            special_proj = self.special_output_proj(special_out)  # (BV, K, C)

            # Unpatchify patches → 2D → flatten
            patches_2d = self.unpatchify(patches_out, Hx, Wx)    # (BV, C, H, W)
            patches_flat = patches_2d.reshape(BV, -1, H_grid * W_grid)  # (BV, C, N)

            # Repack to sequence format: [special | patches]
            special_t = special_proj.permute(0, 2, 1)             # (BV, C, K)
            return torch.cat([special_t, patches_flat], dim=2).unsqueeze(-1)  # (BV, C, K+N, 1)

        out = self.unpatchify(x_toks, Hx, Wx)
        if has_cls:
            imgs, cls_out = out
            # Reshape back to (BV, C, N+1, 1) for transport
            BV, C, H, W = imgs.shape
            imgs_flat = imgs.reshape(BV, C, H*W, 1)
            cls_flat = cls_out.reshape(BV, C, 1, 1)
            return torch.cat([cls_flat, imgs_flat], dim=2)

        return out
    
    def forward_with_pag(self, x, t, total_view, camera_embedding, pag_scale=3.0, pag_layer_idx=None, **kwargs):
        """
        Forward pass with PAG (Perturbed Attention Guidance).
        Runs both normal and perturbed (identity attention) forward passes,
        then combines them with guidance scaling.
        
        Args:
            pag_scale: Guidance scale for PAG. Higher values = stronger guidance.
                       Typical range: 1.0 - 5.0
            pag_layer_idx: Index of the layer to apply PAG to. 
                          If None, defaults to the middle layer (num_blocks // 2).
                          Set to a negative value or a specific list to customize if needed.
        
        Returns:
            Guided output: perturbed + pag_scale * (normal - perturbed)
        """
        if pag_layer_idx is None:
            # Default to layer 22 as requested by user
            pag_layer_idx = 22

        # Normal forward pass
        normal_out = self.forward(x, t, total_view, camera_embedding, pag_mode=False, **kwargs)
        
        # Perturbed forward pass (identity attention applied only to pag_layer_idx)
        perturbed_out = self.forward(x, t, total_view, camera_embedding, pag_mode=True, pag_layer_idx=pag_layer_idx, **kwargs)
        
        # PAG guidance formula: perturbed + scale * (normal - perturbed)
        # This pushes the output towards the normal prediction
        guided_out = perturbed_out + pag_scale * (normal_out - perturbed_out)
        
        return guided_out
    
    def forward_with_cfg(self, x, t, total_view, camera_embedding, cfg_scale, cfg_interval=(0, 1), uncond_camera_embedding=None, use_camera_drop=True, uncond_mode='keep', **kwargs):
        """
        Forward pass of DiT with Classifier-Free Guidance.
        Batches the conditional and unconditional forward passes together.
        
        Args:
            x: Input latents (B*V, C, H, W)
            t: Timesteps (B*V,)
            total_view: Total views in multiview setup
            camera_embedding: Conditional camera embedding (B*V, 7, H, W)
            cfg_scale: Guidance scale (1.0 = no guidance)
            cfg_interval: Time interval (min, max) to apply CFG
            uncond_camera_embedding: Optional pre-computed null camera embedding.
            use_camera_drop: Whether to zero out camera embedding (channels 1-6). Default: True
            uncond_mode: Conditioning noise mode. Options:
                - 'keep': Keep ref_cond and source_condition as-is
                - 'ref_noise': Replace ref_cond with noise
                - 'source_noise': Replace source_condition with noise
                - 'dual_noise': Replace both ref_cond and source_condition with noise
        """
        # Batch size for half (conditional)
        half_n = x.shape[0]
        
        # Prepare combined inputs
        combined_x = torch.cat([x, x], dim=0)
        combined_t = torch.cat([t, t], dim=0)
        
        # 1. Prepare camera embedding for unconditional pass (독립 처리)
        if uncond_camera_embedding is None:
            uncond_camera_embedding = camera_embedding.clone()
            
            if use_camera_drop:
                # Camera drop: zero out channels 1-6 (plucker/pose info)
                uncond_camera_embedding[:, 1:] = 0.0
            # else: keep camera as-is (no drop)
            
        combined_camera = torch.cat([camera_embedding, uncond_camera_embedding], dim=0)
        
        # For ProPE in CFG: Unconditional uses identity viewmats (matching training)
        cfg_kwargs = dict(kwargs)
        if 'viewmats' in cfg_kwargs and cfg_kwargs['viewmats'] is not None:
            viewmats_cond = cfg_kwargs['viewmats']
            
            if self.cfg_mode == "new":
                # NEW Behavior: Use identity viewmats for unconditional
                uncond_viewmats = torch.eye(4, device=viewmats_cond.device, dtype=viewmats_cond.dtype)
                uncond_viewmats = uncond_viewmats.unsqueeze(0).unsqueeze(0).expand_as(viewmats_cond).clone()
            else:
                # OLD Behavior: Keep original viewmats (duplicate for uncond)
                uncond_viewmats = viewmats_cond.clone()

            cfg_kwargs['viewmats'] = torch.cat([viewmats_cond, uncond_viewmats], dim=0)

        if 'Ks' in cfg_kwargs and cfg_kwargs['Ks'] is not None:
            Ks_cond = cfg_kwargs['Ks']           # IMPORTANT: keep intrinsics identical for unconditional path.
            identity_K = torch.eye(3, device=Ks_cond.device, dtype=Ks_cond.dtype).unsqueeze(0).unsqueeze(0)
            identity_K=identity_K.expand_as(Ks_cond)
            # We only drop camera pose info via `uncond_camera_embedding` / identity `viewmats`,
            # but keep K the same to avoid changing intrinsics silently between cond/uncond.
            cfg_kwargs['Ks'] = torch.cat([Ks_cond, identity_K], dim=0)
        
        # 2. Handle ref_cond for CFG (concat mode)
        if 'ref_cond' in cfg_kwargs and cfg_kwargs['ref_cond'] is not None:
            ref_cond = cfg_kwargs['ref_cond']
            
            if uncond_mode in ['ref_noise', 'dual_noise']:
                # Replace ref_cond with noise
                uncond_ref_cond = torch.randn_like(ref_cond)
            else:  # 'keep' or 'source_noise'
                # Keep ref_cond as-is
                uncond_ref_cond = ref_cond
            
            cfg_kwargs['ref_cond'] = torch.cat([ref_cond, uncond_ref_cond], dim=0)
        
        # 3. Handle source_condition for CFG
        if 'source_condition' in cfg_kwargs and cfg_kwargs['source_condition'] is not None:
            source_cond = cfg_kwargs['source_condition']
            
            if uncond_mode in ['source_noise', 'dual_noise']:
                # Replace source_condition with noise
                uncond_source_cond = torch.randn_like(source_cond)
            else:  # 'keep' or 'ref_noise'
                # Keep source_condition as-is
                uncond_source_cond = source_cond
            
            cfg_kwargs['source_condition'] = torch.cat([source_cond, uncond_source_cond], dim=0)

        # Forward pass for both cond and uncond batches
        model_out = self.forward(combined_x, combined_t, total_view, combined_camera, **cfg_kwargs)

        # IMPORTANT (fail-loud): ODE drift requires output shape == input shape.
        # Therefore, in CFG we only return the drift channels (first in_channels)
        # and we DO NOT propagate auxiliary channels through the ODE.
        eps = model_out[:, :self.in_channels]
        cond_eps, uncond_eps = torch.split(eps, half_n, dim=0)
        
        # Apply CFG formula: uncond + scale * (cond - uncond)
        guid_t_min, guid_t_max = cfg_interval
        # Use t from first half
        half_t = t 

        scale_mask = ((half_t >= guid_t_min) & (half_t <= guid_t_max)).view(-1, *[1] * (len(cond_eps.shape) - 1))
        
        guided_eps = torch.where(
            scale_mask,
            uncond_eps + cfg_scale * (cond_eps - uncond_eps),
            cond_eps
        )
        
        return guided_eps

    def forward_with_autoguidance(self, x, t, y, cfg_scale, additional_model_forward, cfg_interval=(0, 1), **kwargs):
        """
        Forward pass of LightningDiT, but also contain the forward pass for the additional model
        """
        model_out = self.forward(x, t, y, **kwargs)
        ag_model_out = additional_model_forward(x, t, y, **kwargs)
        eps = model_out[:, :self.in_channels]
        ag_eps = ag_model_out[:, :self.in_channels]

        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(eps.shape) - 1)),
            ag_eps + cfg_scale * (eps - ag_eps), eps
        )

        return eps

    def forward_with_pag_and_cfg(self, x, t, total_view, camera_embedding, 
                                  cfg_scale, pag_scale=3.0, pag_layer_idx=None, 
                                  cfg_interval=(0, 1), uncond_camera_embedding=None, **kwargs):
        """
        Forward pass with both PAG (Perturbed Attention Guidance) and CFG (Classifier-Free Guidance).
        Requires 3 forward passes:
        1. Perturbed forward (identity attention at pag_layer_idx)
        2. Normal forward (conditional)
        3. Normal forward (unconditional)
        
        Final output: perturbed + pag_scale * (cfg_guided - perturbed)
        where cfg_guided = uncond + cfg_scale * (cond - uncond)
        
        Args:
            x: Input latents (B*V, C, H, W)
            t: Timesteps (B*V,)
            total_view: Total views in multiview setup
            camera_embedding: Conditional camera embedding (B*V, 7, H, W)
            cfg_scale: CFG guidance scale
            pag_scale: PAG guidance scale
            pag_layer_idx: Layer index for PAG
            cfg_interval: Time interval to apply CFG
            uncond_camera_embedding: Optional pre-computed null camera embedding
        """
        if pag_layer_idx is None:
            pag_layer_idx = 22
        
        half_n = x.shape[0]
        
        # Prepare unconditional camera embedding
        if uncond_camera_embedding is None:
            uncond_camera_embedding = camera_embedding.clone()
            uncond_camera_embedding[:, 1:] = 0.0
        
        # For ProPE: Unconditional uses identity viewmats (matching training)
        cfg_kwargs = dict(kwargs)
        if 'viewmats' in cfg_kwargs and cfg_kwargs['viewmats'] is not None:
            viewmats_cond = cfg_kwargs['viewmats']
            # Create identity viewmats for unconditional
            uncond_viewmats = torch.eye(4, device=viewmats_cond.device, dtype=viewmats_cond.dtype)
            uncond_viewmats = uncond_viewmats.unsqueeze(0).unsqueeze(0).expand_as(viewmats_cond).clone()
            cfg_kwargs['viewmats'] = torch.cat([viewmats_cond, uncond_viewmats], dim=0)
        if 'Ks' in cfg_kwargs and cfg_kwargs['Ks'] is not None:
            Ks_cond = cfg_kwargs['Ks']
            # Keep intrinsics identical for unconditional path (see forward_with_cfg).
            cfg_kwargs['Ks'] = torch.cat([Ks_cond, Ks_cond], dim=0)
        
        # 1. Perturbed forward (identity attention) - uses conditional camera
        perturbed_out = self.forward(x, t, total_view, camera_embedding, 
                                     pag_mode=True, pag_layer_idx=pag_layer_idx, **kwargs)
        
        # 2 & 3. Batched cond + uncond forward
        combined_x = torch.cat([x, x], dim=0)
        combined_t = torch.cat([t, t], dim=0)
        combined_camera = torch.cat([camera_embedding, uncond_camera_embedding], dim=0)
        
        combined_out = self.forward(combined_x, combined_t, total_view, combined_camera, **cfg_kwargs)
        
        # Split output
        eps = combined_out[:, :self.in_channels]
        cond_eps, uncond_eps = torch.split(eps, half_n, dim=0)
        perturbed_eps = perturbed_out[:, :self.in_channels]
        
        # Apply CFG: cfg_guided = uncond + cfg_scale * (cond - uncond)
        guid_t_min, guid_t_max = cfg_interval
        half_t = t
        
        scale_mask = ((half_t >= guid_t_min) & (half_t <= guid_t_max)).view(-1, *[1] * (len(cond_eps.shape) - 1))
        
        cfg_guided = torch.where(
            scale_mask,
            uncond_eps + cfg_scale * (cond_eps - uncond_eps),
            cond_eps
        )
        
        # Apply PAG: final = perturbed + pag_scale * (cfg_guided - perturbed)
        final_out = perturbed_eps + pag_scale * (cfg_guided - perturbed_eps)
        
        return final_out
