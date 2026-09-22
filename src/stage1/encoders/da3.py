"""DA3 (Depth Anything 3) encoder for multi-level feature extraction.

Unified encoder class that consolidates all forward logic into a single interface.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Union
from . import register_encoder

# Import DA3 components
from depth_anything_3.api import DepthAnything3


@register_encoder()  
class DA3EncoderDirect(nn.Module):
    """
    Direct DA3 encoder that uses the DinoV2 backbone directly.
    
    Provides unified interface for feature extraction with multiple modes:
    - 'single': Extract features at a single level (default, for training)
    - 'all': Extract features at ALL 4 levels in one pass (for stats computation)
    - 'from_layer': Continue forward from intermediate features (for propagation)
    
    Args:
        pretrained_path: HuggingFace path or local path to DA3 model
        level: Default feature level to extract (0,1,2,3 or -1,-2,-3,-4)
        normalize: Whether to apply layer normalization (unused, kept for compatibility)
    """
    
    # Default OUT_LAYERS for backward compat; overridden dynamically in __init__
    DEFAULT_OUT_LAYERS = [5, 7, 9, 11]
    PATCH_SIZE = 14

    def __init__(
        self,
        pretrained_path: str = "depth-anything/DA3-Base",
        level: int = -1,
        normalize: bool = True,  # Kept for compatibility
    ):
        super().__init__()

        assert level in [-1, -2, -3, -4, 0, 1, 2, 3], \
            f"Level must be 0, 1, 2, 3 or -1, -2, -3, -4, got {level}"

        self.level = level

        # Load DA3 and extract backbone. Keep the DPT / GS heads too: the
        # released demo often has no local ``model.safetensors``, and
        # ``DA3Backbone`` used to skip geometry decode in that case. The Hub
        # snapshot already contains the official DA3-GIANT head.
        da3 = DepthAnything3.from_pretrained(pretrained_path)
        self.backbone = da3.model.backbone
        self.backbone.eval()
        self.dpt_head = getattr(da3.model, "head", None)
        self.gs_head = getattr(da3.model, "gs_head", None)
        self.gs_adapter = getattr(da3.model, "gs_adapter", None)
        for module in (self.dpt_head, self.gs_head, self.gs_adapter):
            if module is None:
                continue
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Dynamically read architecture params from the loaded backbone
        # e.g. [5,7,9,11] for DA3-Base, [11,15,19,23] for DA3-Large
        self.OUT_LAYERS = list(self.backbone.out_layers)
        embed_dim = self.backbone.pretrained.embed_dim  # 768 for Base, 1024 for Large

        # Feature dimensions: [local, current] concatenation -> 2 * embed_dim
        self.hidden_size = embed_dim * 2  # 1536 for Base, 2048 for Large
        self.patch_size = self.PATCH_SIZE

        print(f"[DA3EncoderDirect] Loaded {pretrained_path}: "
              f"embed_dim={embed_dim}, hidden_size={self.hidden_size}, "
              f"OUT_LAYERS={self.OUT_LAYERS}, "
              f"num_blocks={len(self.backbone.pretrained.blocks)}")
    
    def forward(
        self, 
        x: torch.Tensor,
        mode: str = 'single',
        level: int = None,
        start_layer_idx: int = None,
        total_view: int = 1,
        grid_size: Tuple[int, int] = None,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Unified forward pass with multiple modes.
        
        Args:
            x: Input tensor
               - For 'single'/'all': Images (B,C,H,W) or (B,S,C,H,W)
               - For 'from_layer': Intermediate features (B*S,T,C) or (B,S,T,C)
            mode: 
               - 'single': Extract features at one level (default)
               - 'all': Extract features at all 4 levels
               - 'from_layer': Continue from intermediate features
            level: Override self.level for 'single' mode
            start_layer_idx: Required for 'from_layer' mode
            total_view: Number of views for reshaping in 'from_layer' mode
            
        Returns:
            - 'single': (B*S, N, 1536) raw features
            - 'all': Dict[level_idx -> (B*S, N, 1536)] raw features
            - 'from_layer': List[(cls, patches)] for each subsequent OUT_LAYER
        """
        if mode == 'single':
            return self._forward_single(x, level)
        elif mode == 'all':
            return self._forward_all(x)
        elif mode == 'from_layer':
            if start_layer_idx is None:
                raise ValueError("start_layer_idx required for 'from_layer' mode")
            return self._forward_from_layer(x, start_layer_idx, total_view, grid_size=grid_size)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'single', 'all', or 'from_layer'")
    
    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, int]:
        """Prepare input tensor and return dimensions."""
        if x.ndim == 4:
            x = x.unsqueeze(1)
        B, S, C, H, W = x.shape
        return x, B, S, H, W
    
    def _run_backbone_loop(
        self, 
        x: torch.Tensor, 
        B: int, 
        S: int, 
        H: int, 
        W: int,
        target_layers: List[int] = None,
        stop_at_first: bool = False,
    ) -> Dict[int, torch.Tensor]:
        """
        Core backbone forward loop.
        
        Args:
            x: Prepared tokens (B, S, C, H, W)
            B, S, H, W: Dimensions
            target_layers: List of layer indices to collect features from
            stop_at_first: If True, return immediately after first target layer
            
        Returns:
            Dict mapping layer_index -> raw features
        """
        trans = self.backbone.pretrained
        
        # Prepare tokens
        x = trans.prepare_tokens_with_masks(x)
        B_sq, S_tok, N, C_emb = x.shape
        
        # Prepare RoPE
        pos_all, pos_nodiff_all = trans._prepare_rope(B, S, H, W, x.device)
        
        current_x = x
        local_x = None
        results = {}
        
        if target_layers is None:
            target_layers = self.OUT_LAYERS
        
        for i, blk in enumerate(trans.blocks):
            # RoPE logic
            if i < trans.rope_start or trans.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff_all
                l_pos = pos_all

            # Camera Token Handling
            if trans.alt_start != -1 and i == trans.alt_start:
                ref_token = trans.camera_token[:, :1].expand(B, -1, -1)
                if S > 1:
                    src_token = trans.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                else:
                    cam_token = ref_token
                current_x[:, :, 0] = cam_token

            # Attention mechanics
            if trans.alt_start != -1 and i >= trans.alt_start and i % 2 == 1:
                attn_type = "global"
                pos_emb = g_pos
            else:
                attn_type = "local"
                pos_emb = l_pos
                
            current_x = trans.process_attention(current_x, blk, attn_type=attn_type, pos=pos_emb)
            
            if attn_type == "local":
                local_x = current_x
            
            # Collect features at target layers
            if i in target_layers:
                # Construct Raw Feature: [local_x, current_x]
                curr_sq = current_x.view(B * S, N, -1)
                loc_sq = local_x.view(B * S, N, -1)
                out_raw = torch.cat([loc_sq, curr_sq], dim=-1)
                results[i] = out_raw
                
                if stop_at_first:
                    return results
        
        return results
    
    def _forward_single(self, x: torch.Tensor, level: int = None) -> torch.Tensor:
        """Extract features at a single level."""
        x, B, S, H, W = self._prepare_input(x)
        
        level_idx = level if level is not None else self.level
        target_layer = self.OUT_LAYERS[level_idx]
        
        results = self._run_backbone_loop(
            x, B, S, H, W,
            target_layers=[target_layer],
            stop_at_first=True
        )
        
        return results[target_layer]
    
    def _forward_all(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Extract features at all 4 levels."""
        x, B, S, H, W = self._prepare_input(x)
        
        layer_results = self._run_backbone_loop(
            x, B, S, H, W,
            target_layers=self.OUT_LAYERS,
            stop_at_first=False
        )
        
        # Convert layer indices to level indices
        level_results = {}
        for layer_idx, feat in layer_results.items():
            level_idx = self.OUT_LAYERS.index(layer_idx)
            level_results[level_idx] = feat
        
        return level_results
    
    def _forward_from_layer(
        self,
        x: torch.Tensor,
        start_layer_idx: int,
        total_view: int = 1,
        grid_size: Tuple[int, int] = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass starting from intermediate features through remaining backbone blocks.

        Args:
            x: [raw] Intermediate features [local, current] (B, S, T, 1536) or (B*S, T, 1536)
            start_layer_idx: Layer index that produced x (e.g., 5, 7, 9, 11)
            total_view: Number of views for reshaping
            grid_size: Optional (h_lat, w_lat) of the feature map

        Returns:
            List of (cls_raw, patches_layer_norm) tuples for each OUT_LAYER >= start
            cls_raw:            [raw] (B*S, 1536) = [local, current] — no LayerNorm
            patches_layer_norm: [layer_norm] (B*S, N, 1536) = [local, trans.norm(current)]
        """
        # Handle input shape
        if x.ndim == 3:
            B_total, T, C = x.shape
            S = total_view
            B = B_total // S
            x = x.view(B, S, T, C)
        else:
            B, S, T, C = x.shape
        
        trans = self.backbone.pretrained
        embed_dim = trans.embed_dim  # 768
        
        # Recover current state (raw) - second half of features
        current_x = x[..., embed_dim:]  # (B, S, T, 768)
        local_x = x[..., :embed_dim]    # (B, S, T, 768)
        
        # Prepare RoPE - infer H, W from actual feature shape or provided grid_size
        # T includes CLS token, so num_patches = T - 1
        num_patches = T - 1
        
        if grid_size is not None:
             h_lat, w_lat = grid_size
             if h_lat * w_lat != num_patches:
                 raise ValueError(f"Provided grid_size {grid_size} (N={h_lat*w_lat}) does not match num_patches={num_patches}")
             H = h_lat * self.PATCH_SIZE
             W = w_lat * self.PATCH_SIZE
        else:
            # Fallback for square assumptions if grid_size not provided
            patches_per_side = int(num_patches ** 0.5)
            if patches_per_side * patches_per_side != num_patches:
                raise ValueError(f"Number of patches {num_patches} is not a perfect square. T={T}. Please provide grid_size explicitly.")
            H = W = patches_per_side * self.PATCH_SIZE
        pos_all, pos_nodiff_all = trans._prepare_rope(B, S, H, W, x.device)

        results = {}
        
        # Store start layer features (normalized for decoder)
        if start_layer_idx in self.OUT_LAYERS:
            x_flat = x.view(B * S, T, C)
            cls_raw = x_flat[:, 0]  # (B*S, 1536)
            
            local_part = x_flat[:, 1:, :embed_dim]
            curr_part = x_flat[:, 1:, embed_dim:]
            curr_norm = trans.norm(curr_part)  # raw → layer_norm (DINOv2 final LayerNorm)
            out_patches_norm = torch.cat([local_part, curr_norm], dim=-1)  # [layer_norm]

            results[start_layer_idx] = (cls_raw, out_patches_norm)  # (raw cls, layer_norm patches)
        
        # Continue through remaining blocks
        start_block = start_layer_idx + 1
        
        for i, blk in enumerate(trans.blocks):
            if i < start_block:
                continue
            
            # RoPE logic
            if i < trans.rope_start or trans.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff_all
                l_pos = pos_all

            # Camera Token Handling
            if trans.alt_start != -1 and i == trans.alt_start:
                ref_token = trans.camera_token[:, :1].expand(B, -1, -1)
                if S > 1:
                    src_token = trans.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                else:
                    cam_token = ref_token
                current_x[:, :, 0] = cam_token

            if trans.alt_start != -1 and i >= trans.alt_start and i % 2 == 1:
                attn_type = "global"
                pos_emb = g_pos
            else:
                attn_type = "local"
                pos_emb = l_pos
            
            current_x = trans.process_attention(current_x, blk, attn_type=attn_type, pos=pos_emb)
            
            if attn_type == "local":
                local_x = current_x
            
            if i in self.OUT_LAYERS:
                curr_sq = current_x.view(B * S, T, -1)
                loc_sq = local_x.view(B * S, T, -1)

                out_raw = torch.cat([loc_sq, curr_sq], dim=-1)       # [raw]
                curr_norm = trans.norm(curr_sq)                       # raw → layer_norm
                out_final = torch.cat([loc_sq, curr_norm], dim=-1)   # [layer_norm]

                cls_raw = out_raw[:, 0]           # [raw] cls
                patches_norm = out_final[:, 1:]   # [layer_norm] patches

                results[i] = (cls_raw, patches_norm)
        
        # Return ordered by OUT_LAYERS
        return [results[layer] for layer in self.OUT_LAYERS if layer in results]
    
    # Convenience aliases for backward compatibility
    def forward_all_levels(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Alias for forward(mode='all')."""
        return self.forward(x, mode='all')
    
    def forward_from_layer(
        self, 
        x: torch.Tensor, 
        start_layer_idx: int, 
        total_view: int = 1
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Alias for forward(mode='from_layer')."""
        return self.forward(x, mode='from_layer', start_layer_idx=start_layer_idx, total_view=total_view)


# Keep old name for backward compatibility (deprecated)
DA3Encoder = DA3EncoderDirect
