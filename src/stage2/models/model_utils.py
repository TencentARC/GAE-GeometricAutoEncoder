import numpy.random as random

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from typing import *
from math import pi
from timm.models.vision_transformer import Attention
from functools import partial
from einops import rearrange, repeat
from collections.abc import Callable
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf, ListConfig
from .prope import prope_dot_product_attention
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except Exception:
    xops = None  # type: ignore[assignment]
    XFORMERS_AVAILABLE = False



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
               ), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
    else:
        grid_h = np.arange(grid_size[0], dtype=np.float32)
        grid_w = np.arange(grid_size[1], dtype=np.float32)

    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h.size, grid_w.size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        # persistent=False: RoPE freq buffers are deterministic functions of
        # the model config (pt_seq_len / ft_seq_len / dim / theta) and are
        # rebuilt at __init__. Persisting them in state_dict creates resume
        # incompatibilities whenever any of those args changes between save
        # and load (see temporal_rope.py:52 for the same pattern).
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)

        # print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
            
        if isinstance(pt_seq_len, int):
            t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
            freqs = torch.einsum('..., f -> ... f', t, freqs)
            freqs = repeat(freqs, '... n -> ... (n r)', r=2)
            freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
        else:
            # Rectangular support: pt_seq_len is (H, W)
            H, W = pt_seq_len
            
            # For now, assuming ft_seq_len is also passed as (H, W) or we just use H, W directly
            # If ft_seq_len is None, we use pt_seq_len
            if ft_seq_len is None:
                ft_H, ft_W = H, W
            elif isinstance(ft_seq_len, int):
                # Fallback or error? Assuming usage where we just pass tuple
                ft_H, ft_W = ft_seq_len, ft_seq_len 
            else:
                ft_H, ft_W = ft_seq_len
            
            t_h = torch.arange(ft_H) / ft_H * H
            t_w = torch.arange(ft_W) / ft_W * W
            
            freqs_h = torch.einsum('..., f -> ... f', t_h, freqs)
            freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)
            
            freqs_w = torch.einsum('..., f -> ... f', t_w, freqs)
            freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)
            
            freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        # persistent=False: see VisionRotaryEmbedding above for rationale.
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # print('======== shape of rope freq', self.freqs_cos.shape,freqs_sin.shape, '========')

    def forward(self, t):
        # print('======== shape of t', t.shape, '========')
        _, _, Lt, _ = t.shape # B, num_heads, L, dim
        L, _ = self.freqs_cos.shape # L, dim
        
        # Handle CLS token: if Lt is not a multiple of L, assume first tokens are extra (CLS)
        if Lt % L != 0:
            num_extra = Lt % L
            t_extra = t[:, :, :num_extra, :]
            t_patches = t[:, :, num_extra:, :]
            
            # Apply RoPE to patches
            repeat = (Lt - num_extra) // L
            freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin
            if repeat != 1:
                freqs_cos = freqs_cos.repeat_interleave(repeat, dim=0)
                freqs_sin = freqs_sin.repeat_interleave(repeat, dim=0)
            
            t_patches = t_patches * freqs_cos + rotate_half(t_patches) * freqs_sin
            return torch.cat([t_extra, t_patches], dim=2)
            
        repeat = Lt // L
        freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin
        if repeat != 1:
            freqs_cos = freqs_cos.repeat_interleave(repeat, dim=0)
            freqs_sin = freqs_sin.repeat_interleave(repeat, dim=0)
        # apply repeated freqs
        return t * freqs_cos + rotate_half(t) * freqs_sin


class RelativePositionBias2D(nn.Module):
    """
    2D relative positional bias for full self-attention.
    Creates a learnable bias table of size (2*H-1) (2*W-1) per head,
    and a fixed index map to look up bias for any pair of token positions.
    """

    def __init__(self, height: int, width: int, num_heads: int):
        super().__init__()
        self.height = height
        self.width = width
        self.num_heads = num_heads

        # Create a bias table: one bias for every possible relative offset
        # in y ∈ [-(H-1)..(H-1)] and x ∈ [-(W-1)..(W-1)]
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * height - 1) * (2 * width - 1), num_heads)
        )
        # Precompute a (H*W)×(H*W) index matrix of which bias entry each pair (i,j) uses
        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        # meshgrid of absolute coords, shape (H*W, 2)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1).view(-1, 2)

        # Compute all pairwise relative coords
        relative_coords = coords[:, None, :] - coords[None, :, :]   # shape (HW, HW, 2)
        # shift to positive
        relative_coords[..., 0] += height - 1  # y
        relative_coords[..., 1] += width - 1   # x

        # flatten 2D index into single index: idx = y*(2W-1) + x
        relative_index = relative_coords[..., 0] * (2 * width - 1) + relative_coords[..., 1]
        # register as buffer so it’s on the right device / dtype
        self.register_buffer("relative_index", relative_index.long())

    def forward(self):
        """
        Returns:
           bias: Tensor of shape (1, num_heads, HW, HW)
        to be added to the raw attention logits before softmax.
        """
        # Lookup and reshape to (HW, HW, num_heads)
        bias = self.relative_bias_table[self.relative_index.view(-1)]  # (HW*HW, num_heads)
        bias = bias.view(self.height * self.width,
                         self.height * self.width,
                         self.num_heads)  # (HW, HW, heads)
        # permute to (heads, HW, HW) and add batch-dim
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, HW, HW)
        return bias


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class NormAttention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, total_view, rope=None, use_prope=False, viewmats=None, Ks=None, pag_mode=False, prope_image_size=None, patches_layout=None, num_prefix_tokens=0) -> torch.Tensor:
        '''
        minkyung: 3D inflated attention for all attention layer
        concatenate over views after q,k,v projection & rope embedding

        Args:
            pag_mode: If True, use identity attention (PAG - Perturbed Attention Guidance).
                      Each token only attends to itself, skipping cross-token attention.
            prope_image_size: Image size for ProPE. If None, defaults to 252 for backward compatibility.
            num_prefix_tokens: Number of non-spatial prefix tokens (e.g. special tokens) that should
                              skip RoPE/ProPE. These sit at the beginning of the sequence.
        '''
        BV, N, C = x.shape
        V = total_view
        assert BV % V == 0, "batch B is not dividable by view V"
        B = BV // V
        KP = num_prefix_tokens  # alias for clarity

        qkv = self.qkv(x).reshape(BV, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            # If using ProPE, we skip standard RoPE here because ProPE handles it internally
            if not use_prope:
                if KP > 0:
                    # Special tokens don't get RoPE (non-spatial)
                    q_prefix, q_spatial = q[:, :, :KP], q[:, :, KP:]
                    k_prefix, k_spatial = k[:, :, :KP], k[:, :, KP:]
                    q_spatial = rope(q_spatial)
                    k_spatial = rope(k_spatial)
                    q = torch.cat([q_prefix, q_spatial], dim=2)
                    k = torch.cat([k_prefix, k_spatial], dim=2)
                else:
                    q = rope(q)
                    k = rope(k)

        # Rearrange to (B, H, S, D) where S = V * N
        q = rearrange(q, "(b v) h n c -> b h (v n) c", v=V) 
        k = rearrange(k, "(b v) h n c -> b h (v n) c", v=V) 
        v = rearrange(v, "(b v) h n c -> b h (v n) c", v=V)
        
        if pag_mode:
            # PAG: Identity attention - each token only attends to itself
            # This is equivalent to skipping the attention computation entirely
            # and just using the values directly (with projection)
            x = v  # Shape: (B, H, S, D)
        elif self.fused_attn:
            q = q.to(v.dtype)
            k = k.to(v.dtype) # rope may change the q,k's dtype
            
            if use_prope:
                assert viewmats is not None and Ks is not None
                if Ks.dtype != q.dtype:
                    Ks = Ks.to(q.dtype)

                # patches_x and patches_y need to be derived from N (token length)
                # KP prefix tokens (special/CLS) are non-spatial and skip ProPE
                N_spatial = N - KP  # number of spatial (patch) tokens per view
                if patches_layout is not None:
                     patches_h, patches_w = patches_layout
                     if patches_h * patches_w == N_spatial:
                         patches_dim_h, patches_dim_w = patches_h, patches_w
                         N_patches = N_spatial
                     elif patches_h * patches_w == N - 1:
                         # Legacy CLS path (KP=0 but predict_cls=True adds 1 token)
                         patches_dim_h, patches_dim_w = patches_h, patches_w
                         N_patches = N - 1
                     else:
                         raise ValueError(f"patches_layout {patches_layout} does not match tokens {N} (KP={KP})")
                else:
                    patches_dim = int(N_spatial ** 0.5)
                    patches_dim_h = patches_dim
                    patches_dim_w = patches_dim

                has_prefix = (patches_dim_h * patches_dim_w != N)

                if has_prefix:
                    # Generalized prefix handling: KP prefix tokens (special or CLS) + N_patches spatial
                    # Effective prefix count — use KP if > 0, else fall back to 1 (legacy CLS)
                    eff_K = KP if KP > 0 else 1
                    N_patches = N - eff_K
                    if patches_layout is None:
                        patches_dim = int(N_patches ** 0.5)
                        assert patches_dim * patches_dim == N_patches, f"ProPE requires square patch grid, but got {N_patches} patches"
                        patches_dim_h = patches_dim_w = patches_dim
                    else:
                        patches_dim_h, patches_dim_w = patches_layout
                        assert patches_dim_h * patches_dim_w == N_patches

                    # 1. Prepare ProPE transformations for patches
                    from .prope import _prepare_apply_fns

                    image_size = prope_image_size if prope_image_size is not None else 252
                    if isinstance(image_size, (tuple, list)):
                        img_h, img_w = image_size
                    else:
                        img_h = img_w = image_size

                    apply_fn_q_p, apply_fn_kv_p, apply_fn_o_p = _prepare_apply_fns(
                        head_dim=self.head_dim,
                        viewmats=viewmats,
                        Ks=Ks,
                        patches_x=patches_dim_w,
                        patches_y=patches_dim_h,
                        image_width=img_w,
                        image_height=img_h,
                    )

                    # 2. Extract prefix and patches from (B, H, V*(eff_K+N_patches), D)
                    # Reshape to (B, H, V, eff_K+N_patches, D)
                    q_mv = q.reshape(B, self.num_heads, V, N, self.head_dim)
                    k_mv = k.reshape(B, self.num_heads, V, N, self.head_dim)
                    v_mv = v.reshape(B, self.num_heads, V, N, self.head_dim)

                    q_prefix = q_mv[:, :, :, :eff_K, :]
                    k_prefix = k_mv[:, :, :, :eff_K, :]
                    v_prefix = v_mv[:, :, :, :eff_K, :]

                    q_p = q_mv[:, :, :, eff_K:, :].reshape(B, self.num_heads, V * N_patches, self.head_dim)
                    k_p = k_mv[:, :, :, eff_K:, :].reshape(B, self.num_heads, V * N_patches, self.head_dim)
                    v_p = v_mv[:, :, :, eff_K:, :].reshape(B, self.num_heads, V * N_patches, self.head_dim)

                    # 3. Apply ProPE to patches only
                    q_p = apply_fn_q_p(q_p)
                    k_p = apply_fn_kv_p(k_p)
                    v_p = apply_fn_kv_p(v_p)

                    # 4. Re-combine prefix and patches
                    q_p = q_p.reshape(B, self.num_heads, V, N_patches, self.head_dim)
                    k_p = k_p.reshape(B, self.num_heads, V, N_patches, self.head_dim)
                    v_p = v_p.reshape(B, self.num_heads, V, N_patches, self.head_dim)

                    q_final = torch.cat([q_prefix, q_p], dim=3).reshape(B, self.num_heads, V * N, self.head_dim)
                    k_final = torch.cat([k_prefix, k_p], dim=3).reshape(B, self.num_heads, V * N, self.head_dim)
                    v_final = torch.cat([v_prefix, v_p], dim=3).reshape(B, self.num_heads, V * N, self.head_dim)

                    # 5. Attention
                    if XFORMERS_AVAILABLE:
                         # xformers requires (B, S, H, D)
                        q_final = q_final.permute(0, 2, 1, 3)
                        k_final = k_final.permute(0, 2, 1, 3)
                        v_final = v_final.permute(0, 2, 1, 3)

                        # Correctly handle B200 requirements:
                        # 1. Input must be bfloat16/fp16 for xformers
                        # 2. Output must be cast back to original dtype to match residual stream/next layers
                        orig_dtype = q_final.dtype
                        if orig_dtype == torch.float32:
                            q_final = q_final.to(torch.bfloat16)
                            k_final = k_final.to(torch.bfloat16)
                            v_final = v_final.to(torch.bfloat16)

                        x = xops.memory_efficient_attention(
                            q_final, k_final, v_final,
                            p=self.attn_drop.p if self.training else 0.,
                        )

                        # Restore dtype if we changed it
                        if x.dtype != orig_dtype:
                            x = x.to(orig_dtype)

                        # Back to (B, H, S, D)
                        x = x.permute(0, 2, 1, 3)
                    else:
                        x = F.scaled_dot_product_attention(
                            q_final, k_final, v_final,
                            dropout_p=self.attn_drop.p if self.training else 0.,
                        )

                    # 6. Inverse transformation for patches only (prefix tokens unchanged)
                    x_mv = x.reshape(B, self.num_heads, V, N, self.head_dim)
                    x_prefix = x_mv[:, :, :, :eff_K, :]
                    x_p = x_mv[:, :, :, eff_K:, :].reshape(B, self.num_heads, V * N_patches, self.head_dim)
                    x_p = apply_fn_o_p(x_p)
                    x_p = x_p.reshape(B, self.num_heads, V, N_patches, self.head_dim)

                    x = torch.cat([x_prefix, x_p], dim=3).reshape(B, self.num_heads, V * N, self.head_dim)
                else:
                    # No CLS, use prope_dot_product_attention directly
                    if isinstance(prope_image_size, (list, tuple, ListConfig)):
                        # If list/tuple, assume [h, w] (or [w, h]? usually [h, w] in config, but let's assume image_size in config is [H, W])
                        # Wait, DiT head uses [H, W] usually. Let's check logic.
                        # Actually standard convention is [H, W]. ProPE expects image_width, image_height.
                        image_h, image_w = prope_image_size[0], prope_image_size[1]
                    else:
                        size = prope_image_size if prope_image_size is not None else 252
                        image_h, image_w = size, size
                        
                    x = prope_dot_product_attention(
                        q, k, v,
                        viewmats=viewmats,
                        Ks=Ks,
                        patches_x=patches_dim_w,
                        patches_y=patches_dim_h,
                        image_width=image_w,
                        image_height=image_h,
                        dropout_p=self.attn_drop.p if self.training else 0.,
                    )
            else:
                if XFORMERS_AVAILABLE:
                    # xformers requires (B, S, H, D)
                    q = q.permute(0, 2, 1, 3)
                    k = k.permute(0, 2, 1, 3)
                    v = v.permute(0, 2, 1, 3)
                    
                    # Correctly handle B200 requirements
                    orig_dtype = q.dtype
                    if orig_dtype == torch.float32:
                        q = q.to(torch.bfloat16)
                        k = k.to(torch.bfloat16)
                        v = v.to(torch.bfloat16)
                        
                    x = xops.memory_efficient_attention(
                        q, k, v,
                        p=self.attn_drop.p if self.training else 0.,
                    )

                    # Restore dtype if we changed it
                    if x.dtype != orig_dtype:
                        x = x.to(orig_dtype)

                    # Back to (B, H, S, D)
                    x = x.permute(0, 2, 1, 3)
                else:
                    x = F.scaled_dot_product_attention(
                        q, k, v,
                        dropout_p=self.attn_drop.p if self.training else 0.,
                    )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.transpose(1, 2).reshape(B, V * N, C)
        x = rearrange(x, "b (v n) c -> (b v) n c", v=V)     # (BV, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    # def forward(self, x: torch.Tensor, total_view, rope=None,) -> torch.Tensor:
    #     '''
    #     minkyung: 3D inflated attention for all attention layer
    #     concatenate over views after q,k,v projection & rope embedding 
    #     '''
    #     x = rearrange(x, "(b v) n c -> b (v n) c", v=total_view)
    #     B, N, C = x.shape
    #     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv.unbind(0)
    #     q, k = self.q_norm(q), self.k_norm(k)
        
    #     if rope is not None:
    #         q = rope(q)
    #         k = rope(k)
        
    #     if self.fused_attn:
    #         q = q.to(v.dtype)
    #         k = k.to(v.dtype) # rope may change the q,k's dtype
    #         x = F.scaled_dot_product_attention(
    #             q, k, v,
    #             dropout_p=self.attn_drop.p if self.training else 0.,
    #         )
    #     else:
    #         q = q * self.scale
    #         attn = q @ k.transpose(-2, -1)
    #         attn = attn.softmax(dim=-1)
    #         attn = self.attn_drop(attn)
    #         x = attn @ v
    #     x = x.transpose(1, 2).reshape(B, N, C)
    #     x = rearrange(x, "b (v n) c -> (b v) n c", v=total_view)     # (BV, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
        
    #     return x

    
    
class GaussianFourierEmbedding(nn.Module):
    """
    Gaussian Fourier Embedding for timesteps. 
    """
    embedding_size: int = 256
    scale: float = 1.0
    def __init__(self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        self.W = nn.Parameter(torch.normal(0, self.scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    def forward(self, t):
        with torch.no_grad():
            W = self.W # stop gradient manually
        t = t[:, None] * W[None, :] * 2 * torch.pi
        # Concatenate sine and cosine transformations
        t_embed =  torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        t_embed = self.mlp(t_embed)
        return t_embed


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings