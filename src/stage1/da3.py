"""Frozen DA3 backbone for GAE — paper Section 3.1.

Wraps Depth Anything 3 (DA3-GIANT) as two frozen components:

* the encoder E, producing the four-level feature hierarchy
  F_v^{0:3} = {f_{v,l}}, l = 0..3   (the input to paper Eq. 5);
* the dense-prediction-transformer head H_DPT, which reads depth, rays and
  point maps back out of a reconstructed hierarchy (paper Eq. 4).

Both stay frozen throughout Stage 1 and Stage 2. Freezing H_DPT is what makes
"geometry reconstruction must remain readable by the original DA3 head" a real
constraint rather than a soft one.

``propagate_features`` is retained: the raw-DA3 L0 baseline (a controlled
comparison row in the paper) needs to run a shallow level forward through the
remaining backbone blocks before the DPT head.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Union
from math import sqrt
from copy import deepcopy

from .encoders import ARCHS
from transformers import AutoConfig
from utils.config_utils import parse_encoder_size

# Import DA3 for DPT decoder
from depth_anything_3.api import DepthAnything3
from safetensors.torch import load_file
from .decoders.decoder import GeneralDecoder_Variable
from transformers import AutoConfig

class DA3Backbone(nn.Module):
    """
    RAE using DA3 encoder for multi-level feature extraction.
    
    Provides unified interface for:
    - Training: encode(mode='single') for single-level feature extraction
    - Stats computation: encode(mode='all') for all 4 levels
    - Validation: decode() for RGB+Depth reconstruction
    - Propagation: propagate_features() for shallow→deep feature propagation
    
    Args:
        encoder_pretrained_path: Path to pretrained DA3 model
        level: Feature level (0,1,2,3 or -1,-2,-3,-4)
        encoder_input_size: Input image size (default: 518) or [H, W]
        encoder_type: Encoder class name
        dpt_decoder_path: Path to pretrained DPT decoder
        da3_weights_path: Path to DA3 backbone weights
        noise_tau: Noise level for training
        reshape_to_2d: Whether to reshape to (B,C,H,W)
        normalization_stat_path: Path to normalization statistics
    """
    
    # OUT_LAYERS and PATCH_SIZE are set dynamically in __init__ from encoder
    PATCH_SIZE = 14
    
    def __init__(
        self,
        encoder_pretrained_path: str = "depth-anything/DA3-Base",
        level: int = -1,
        encoder_input_size: Union[int, List[int], Tuple[int, int]] = 518,
        encoder_type: str = 'DA3EncoderDirect',
        dpt_decoder_path: Optional[str] = None,
        da3_weights_path: Optional[str] = None,
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
        dpt_model_type: str = 'da3-base',
        mae_weight: Optional[str] = None,
        # Legacy/unused params (kept for config compatibility)
        decoder_config_path: Optional[str] = None,
        decoder_patch_size: int = 14,
        pretrained_decoder_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.level = level
        self.encoder_h, self.encoder_w = parse_encoder_size(encoder_input_size)
        self.encoder_input_size = encoder_input_size # Keep original arg for reference
        self.reshape_to_2d = reshape_to_2d
        self.noise_tau = noise_tau
        self.eps = eps
        self.dpt_model_type = dpt_model_type
        
        # Initialize encoder
        encoder_cls = ARCHS[encoder_type]
        self.encoder = encoder_cls(
            pretrained_path=encoder_pretrained_path,
            level=level,
        )
        
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size

        # Dynamically read OUT_LAYERS from encoder (set by loaded backbone)
        self.OUT_LAYERS = self.encoder.OUT_LAYERS
        
        # Calculate patch counts for potentially rectangular inputs
        h_patches = self.encoder_h // self.encoder_patch_size
        w_patches = self.encoder_w // self.encoder_patch_size
        self.num_patches = h_patches * w_patches
        self.num_patches_per_side = (h_patches, w_patches) # Store as tuple if rectangular
        
        # ImageNet normalization constants
        self.register_buffer('encoder_mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('encoder_std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        
        # Initialize DPT decoder for validation / point clouds.
        # Prefer a local safetensors if provided. If it is missing,
        # reuse the DPT head already loaded by ``DA3EncoderDirect.from_pretrained``
        # so Hub-only demos still decode depth/ray and write ``.ply``.
        self.rae_cl_decoder = None
        self.da3_backbone = None
        # GS branch (only present for da3-giant): kept frozen for GS distillation
        # loss. gs_adapter is loaded for completeness but unused by raw_gs distill.
        self.gs_head = None
        self.gs_adapter = None
        weight_file = self._resolve_da3_weight_file(
            da3_weights_path, encoder_pretrained_path)
        if weight_file:
            self._init_dpt_decoder(weight_file, dpt_decoder_path)
        elif getattr(self.encoder, "dpt_head", None) is not None:
            self._attach_dpt_from_encoder(dpt_decoder_path)
        else:
            print("[DA3Backbone] no DPT head — geometry / point clouds disabled")
            
        # Initialize MAE decoder if weight is provided
        self.mae_decoder = None
        if mae_weight is not None:
            if decoder_config_path is None:
                decoder_config_path = 'configs/decoder/ViTXL'
            
            print(f"Initializing MAE decoder from {mae_weight} using config {decoder_config_path}")
            decoder_config = AutoConfig.from_pretrained(decoder_config_path)
  
            decoder_config.hidden_size = self.latent_dim * 4  # 4 levels concatenated
            decoder_config.patch_size = decoder_patch_size
            # RAE_MAE: decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches))
            # We use base_image_size=(504,504) in GeneralDecoder_Variable init.
            
            # Note: GeneralDecoder_Variable expects base_image_size.
            # We assume 504x504 as standard for now, or use encoder input size if 504
            base_size = (504, 504)
            
            self.mae_decoder = GeneralDecoder_Variable(decoder_config, base_image_size=base_size)
            
            # Load weights
            ckpt = torch.load(mae_weight, map_location='cpu')
            if 'ema' in ckpt:
                state_dict = ckpt['ema']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt # Assume direct state dict
                
            # Filter for decoder keys only
            decoder_state = {}
            for k, v in state_dict.items():
                if k.startswith('decoder.'):
                    decoder_state[k.replace('decoder.', '')] = v
            
            missing, unexpected = self.mae_decoder.load_state_dict(decoder_state, strict=False)
            if len(missing) > 0:
                print(f"MAE Decoder Missing keys: {missing[:5]} ...")
            
            self.mae_decoder.eval()
            for p in self.mae_decoder.parameters():
                p.requires_grad = False
                
        # # Load normalization statistics
        self._init_normalization(normalization_stat_path)
    
    @staticmethod
    def _usable_decoder_path(path) -> str | None:
        if path is None:
            return None
        path = str(path).strip()
        if path.lower() in ("", "none", "null"):
            return None
        return path if os.path.isfile(path) else None

    @staticmethod
    def _resolve_da3_weight_file(da3_weights_path, encoder_pretrained_path) -> str | None:
        """Return a readable DA3 ``model.safetensors``, or None.

        The yaml default ``pretrained_models/da3_giant/model.safetensors`` is
        present on the internal mount and absent in the public demo. In the
        latter case we fall back to the Hub snapshot next to the encoder.
        """
        candidates = []
        if da3_weights_path:
            candidates.append(da3_weights_path)
        if encoder_pretrained_path and os.path.isdir(str(encoder_pretrained_path)):
            candidates.append(os.path.join(encoder_pretrained_path, "model.safetensors"))
        try:
            from huggingface_hub import try_to_load_from_cache
            repo = encoder_pretrained_path
            if isinstance(repo, str) and "/" in repo and not os.path.isdir(repo):
                cached = try_to_load_from_cache(repo, "model.safetensors")
                if cached:
                    candidates.append(cached)
        except Exception:
            pass
        seen = set()
        for raw in candidates:
            if not raw or raw in seen:
                continue
            seen.add(raw)
            path = raw if os.path.isabs(raw) else os.path.abspath(raw)
            if os.path.isfile(path):
                return path
        return None

    def _attach_dpt_from_encoder(self, dpt_decoder_path):
        """Reuse the DPT/GS heads already sitting on the Hub-loaded encoder."""
        print("[DA3Backbone] DPT head from encoder Hub snapshot (no local safetensors)")
        self.da3_backbone = self.encoder.backbone
        self.rae_cl_decoder = self.encoder.dpt_head
        self.gs_head = getattr(self.encoder, "gs_head", None)
        self.gs_adapter = getattr(self.encoder, "gs_adapter", None)
        if self.gs_head is not None:
            self.gs_head.eval()
            for p in self.gs_head.parameters():
                p.requires_grad = False
            print("[GS head] attached from encoder and frozen")
        self._finalize_dpt_head(dpt_decoder_path)

    def _finalize_dpt_head(self, dpt_decoder_path):
        overlay = self._usable_decoder_path(dpt_decoder_path)
        if overlay is not None:
            ckpt = torch.load(overlay, map_location="cpu")
            if "model" in ckpt:
                decoder_state = {}
                for k, v in ckpt["model"].items():
                    if k.startswith("decoder."):
                        decoder_state[k.replace("decoder.", "")] = v
                    elif k.startswith("pred."):
                        decoder_state[k.replace("pred.", "")] = v
                if decoder_state:
                    current_state = self.rae_cl_decoder.state_dict()
                    compatible_state = {}
                    skipped = []
                    for k, v in decoder_state.items():
                        if k in current_state and v.shape == current_state[k].shape:
                            compatible_state[k] = v
                        else:
                            skipped.append(k)
                    if skipped:
                        print(f"[DPT decoder] Skipped {len(skipped)} incompatible keys "
                              f"(shape mismatch, e.g. {skipped[0]})")
                    if compatible_state:
                        self.rae_cl_decoder.load_state_dict(compatible_state, strict=False)
                        print(f"[DPT decoder] Loaded {len(compatible_state)} compatible keys")
        self.rae_cl_decoder.eval()
        for p in self.rae_cl_decoder.parameters():
            p.requires_grad = False
        if hasattr(self.rae_cl_decoder, "inference"):
            self.rae_cl_decoder.inference = True
            print(f"Enabled inference mode for DPT decoder (inference={self.rae_cl_decoder.inference})")

    @staticmethod
    def _cache_to_local(path: str) -> str:
        """Copy a file to /local-ssd if available, returning the (possibly cached) path.

        Multi-process safe: writes to a pid-tagged temp file then atomically
        renames into place. Required because precompute scripts launch N shards
        per node that all hit this codepath simultaneously on cold cache; a
        naive ``if not isfile: copy`` lets every process race to write the same
        ``dst`` and any concurrent reader sees a half-written safetensors file
        ("Error while deserializing header: header too small").
        """
        ssd = "/local-ssd"
        if not os.path.isdir(ssd) or path.startswith(ssd):
            return path
        import hashlib, shutil
        h = hashlib.md5(path.encode()).hexdigest()[:12]
        cache_dir = os.path.join(ssd, "da3_cache")
        dst = os.path.join(cache_dir, f"{h}_{os.path.basename(path)}")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(path):
                return dst
        except OSError:
            pass
        tmp = os.path.join(cache_dir, f"_staging_{os.getpid()}_{h}_{os.path.basename(path)}")
        print(f"[DA3Backbone] caching {path} → {dst}")
        shutil.copy2(path, tmp)
        try:
            os.replace(tmp, dst)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return dst

    def _init_dpt_decoder(self, da3_weights_path: str, dpt_decoder_path: str):
        """Initialize DPT decoder for RGB+Depth reconstruction."""
        # Reuse encoder backbone instead of loading duplicate
        self.da3_backbone = self.encoder.backbone
        
        # Load a DA3 model just to get the head architecture
        print(f"Loading DepthAnything3 with config: {self.dpt_model_type}")
        model = DepthAnything3(self.dpt_model_type)
        da3_weights_path = self._cache_to_local(da3_weights_path)
        state_dict = load_file(da3_weights_path)
        model.load_state_dict(state_dict, strict=False)
        
        # Extract only the head (decoder)
        self.rae_cl_decoder = model.model.head

        # Keep the frozen GS branch if available (da3-giant) for GS distillation.
        # del model below only drops the local name; submodules referenced here
        # by self stay alive.
        self.gs_head = getattr(model.model, "gs_head", None)
        self.gs_adapter = getattr(model.model, "gs_adapter", None)
        if self.gs_head is not None:
            self.gs_head.eval()
            for p in self.gs_head.parameters():
                p.requires_grad = False
            print("[GS head] loaded and frozen for GS distillation")
        
        self._finalize_dpt_head(dpt_decoder_path)
        # Clean up the temporary model (only kept the head)
        del model
    
    def _init_normalization(self, stat_path: Optional[str]):
        """Load normalization statistics."""
        if stat_path is not None:
            stats = torch.load(stat_path, map_location='cpu')
            
            self.register_buffer('latent_mean', stats.get('mean', None))
            self.register_buffer('latent_var', stats.get('var', None))
            self.do_normalization = True
            print(f"Loaded normalization stats from {stat_path}")
        else:
            self.latent_mean = None
            self.latent_var = None
            self.do_normalization = False
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Prepare input tensor: handle 4D/5D. No longer forces resizing.
        
        Returns:
            (prepared_tensor, batch_size, num_views)
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)
        
        b, v, c, h, w = x.shape
        
        # We no longer force resizing to self.encoder_h/w.
        # Encoders (DA3) handle dynamic sizes but usually require divisibility by patch size (14).
        # We assume the user/dataloader handles divisibility at the level of gt_inp.
        
        return x, b, v
    
    def _normalize(
        self,
        z: torch.Tensor,
        cls: torch.Tensor = None,
        dim: int = None,   # channel size (e.g., 1536 for Base, 2048 for Large). None → self.latent_dim
    ):
        """raw → latent_norm: zero-mean, unit-var normalization using per-level stats."""
        if dim is None:
            dim = self.latent_dim
        if not self.do_normalization:
            return (z, cls) if cls is not None else z

        mean = self.latent_mean.to(z.device, z.dtype).reshape(-1)
        var  = self.latent_var .to(z.device, z.dtype).reshape(-1)
        std  = torch.sqrt(var + self.eps)

        C = dim
        if mean.numel() != C:
            raise ValueError(f"latent_mean len={mean.numel()} != dim={C}")

        def bcast(x, stat):
            ax = [i for i, s in enumerate(x.shape) if s == C]
            if len(ax) != 1:
                raise ValueError(f"Cannot uniquely infer channel axis for C={C} in shape={tuple(x.shape)}")
            shape = [1] * x.dim()
            shape[ax[0]] = C
            return stat.view(*shape)

        z = (z - bcast(z, mean)) / bcast(z, std)

        if cls is not None:
            cls = (cls - bcast(cls, mean)) / bcast(cls, std)
            return z, cls

        return z

    def _denormalize(
        self,
        z: torch.Tensor,
        cls: torch.Tensor = None,
        dim: int = None,   # channel size (e.g., 1536 for Base, 2048 for Large). None → self.latent_dim
    ):
        """latent_norm → raw: reverse latent statistics normalization."""
        if dim is None:
            dim = self.latent_dim
        if not self.do_normalization:
            return (z, cls) if cls is not None else z

        mean = self.latent_mean.to(z.device, z.dtype).reshape(-1)
        var  = self.latent_var .to(z.device, z.dtype).reshape(-1)
        std  = torch.sqrt(var + self.eps)

        C = dim
        if mean.numel() != C:
            raise ValueError(f"latent_mean len={mean.numel()} != dim={C}")

        def bcast(x, stat):
            ax = [i for i, s in enumerate(x.shape) if s == C]
            if len(ax) != 1:
                raise ValueError(f"Cannot uniquely infer channel axis for C={C} in shape={tuple(x.shape)}")
            shape = [1] * x.dim()
            shape[ax[0]] = C
            return stat.view(*shape)

        z = z * bcast(z, std) + bcast(z, mean)

        if cls is not None:
            cls = cls * bcast(cls, std) + bcast(cls, mean)
            return z, cls

        return z

    
    def _separate_cls(self, z: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate CLS token from patch features."""
        bv, n, c = z.shape
        
        # 1. If grid_size is provided, use it to check for CLS
        if grid_size is not None:
            expected_patches = grid_size[0] * grid_size[1]
            if n == expected_patches + 1:
                return z[:, 1:], z[:, 0]
            elif n == expected_patches:
                return z, None

        # 2. Fallback to initialization parameters
        if n == self.num_patches + 1:
            return z[:, 1:], z[:, 0]
        elif n == self.num_patches:
            return z, None
        
        # 3. Last resort: check if square perfect
        side = int(sqrt(n))
        if side * side == n - 1:
             return z[:, 1:], z[:, 0]
        
        return z, None
    
    def _reshape_to_2d(self, z: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Reshape (B*V, N, C) -> (B*V, C, H, W)"""
        bv, n, c = z.shape
        
        if grid_size is not None:
             h, w = grid_size
             if n != h * w:
                 # Grid size mismatch, likely due to CLS not being separated yet?
                 # Should not happen if _separate_cls was called.
                 # Fallback to square if possible
                 h = w = int(sqrt(n))
        elif n == self.num_patches:
             if isinstance(self.num_patches_per_side, tuple):
                  h, w = self.num_patches_per_side
             else:
                  h = w = int(sqrt(n))
        else:
             h = w = int(sqrt(n))
                 
        return z.transpose(1, 2).reshape(bv, c, h, w)
    
    def _reshape_to_seq(self, z: torch.Tensor) -> torch.Tensor:
        """Reshape (B*V, C, H, W) -> (B*V, N, C)"""
        bv, c, h, w = z.shape
        return z.reshape(bv, c, h * w).transpose(1, 2)
    
    # =========================================================================
    # Main Encoding Methods
    # =========================================================================
    

    @property
    def backbone_norm(self) -> nn.Module:
        """ViT LayerNorm on the global half of each DA3 level (DPT input)."""
        return self.encoder.backbone.pretrained.norm

    def encode(
        self,
        x: torch.Tensor,
        mode: str = 'single',
        return_cls: bool = False,
        level: int = None,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Unified encoding interface.

        Norm flow: Image → DA3 backbone → raw → _normalize → latent_norm

        Args:
            x: Input images [0,1] range. 4D (B,C,H,W) or 5D (B,V,C,H,W)
            mode:
                'single' - Extract features at self.level (training)
                'all' - Extract features at all 4 levels (stats/decode)
            return_cls: If True, also return CLS token (only for mode='single')
            level: Override self.level for mode='single'

        Returns:  [latent_norm]
            mode='single': (B*V, C, H, W) latent_norm
                          With return_cls=True: (latent_norm features, latent_norm cls)
            mode='all': Dict[level_idx -> (B*V, N, C)]  (raw, not latent_norm)
        """
        x, b, v = self._prepare_input(x)
        
        if mode == 'all':
            with torch.no_grad():
                return self.encoder.forward(x, mode='all')
        
        # mode == 'single'
        z = self.encoder.forward(x, mode='single', level=level)
        
        # Handle encoder output shape
        if z.ndim == 4:
            b_out, v_out, n, c_out = z.shape
            z = z.reshape(b_out * v_out, n, c_out)
        
        # Add noise during training
        if self.training and self.noise_tau > 0:
            noise_sigma = self.noise_tau * torch.rand(
                (z.size(0),) + (1,) * (z.dim() - 1), device=z.device
            )
            z = z + noise_sigma * torch.randn_like(z)
        
        # Separate CLS and reshape
        cls = None
        if self.reshape_to_2d:
            h_p, w_p = x.shape[-2] // 14, x.shape[-1] // 14
            z, cls = self._separate_cls(z, grid_size=(h_p, w_p))
            z = self._reshape_to_2d(z, grid_size=(h_p, w_p))
        
        # Normalize
        if cls is not None:
            z, cls = self._normalize(z, cls)
        else:
            z = self._normalize(z)
        
        if return_cls:
            return z, cls
        return z
    
    def encode_with_cls(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience alias for encode(return_cls=True)."""
        return self.encode(x, return_cls=True)
    
    def encode_all_levels(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Convenience alias for encode(mode='all')."""
        return self.encode(x, mode='all')
    
    # =========================================================================
    # Decoding Methods
    # =========================================================================
    
    def decode(
        self,
        feats: List[Tuple[torch.Tensor, ...]],
        H: int = 518,
        W: int = 518
    ) -> dict:
        """
        Decode multi-level features to RGB+Depth using DPT head.

        Args:
            feats: [layer_norm patches, any cls]
                   List of (patches, cls) tuples, one per level (L0..L3).
                   patches: (B, V, N, C) — must be layer_norm
                   cls: (B, V, C) — can be raw or dummy zeros
            H, W: Output image size

        Returns:
            dict with 'rgb' (ImageNet denormalized), 'depth', 'depth_conf'
        """
        rgb = None
        depth = None
        depth_conf = None

        # 1. Run DPT decoder if available (produces Depth + initial RGB)
        if self.rae_cl_decoder is not None:
            with torch.no_grad():
                with torch.autocast(device_type=feats[0][0].device.type, enabled=False):
                    output = self.rae_cl_decoder(feats, H, W, patch_start_idx=0) # call tridpt_
            rgb = output['rgb']
            depth = output.get('depth', None)
            depth_conf = output.get('depth_conf', None)
            ray = output.get('ray', None)
            ray_conf = output.get('ray_conf', None)

        # 2. If MAE decoder is present, overwrite RGB
        if self.mae_decoder is not None:
            # Prepare input for MAE: Concat all 4 levels (patches only)
            # feats is list of (patches, cls). We need only patches.
            # Assuming feats is [L0, L1, L2, L3] order or similar.
            # RAE_MAE cats: [z[i][0] for i in range(4)].
            
            # Check if we have 4 levels
            if len(feats) != 4:
                # If propagation wasn't full 4 levels, we might have issues.
                # But typical decode usage in validation supplies 4 levels.
                pass 
                
            mae_feats = []
            for (patches, cls_token) in feats:
                # patches: (B, V, N, C)
                mae_feats.append(patches)
                
            # cat dim=-1 => (B, V, N, C*4)
            z_cat = torch.cat(mae_feats, dim=-1)
            
            # Reshape to (B*V, N, C_total)
            b, v, n, c_tot = z_cat.shape
            z_cat = z_cat.reshape(b*v, n, c_tot)
            
            # Run MAE decoder
            with torch.no_grad():
                with torch.autocast(device_type=z_cat.device.type, enabled=True, dtype=torch.bfloat16):
                    # MAE decoder forward
                    # forward(hidden_states, input_size, drop_cls_token=False)
                    mae_out_logits = self.mae_decoder(z_cat, input_size=(H, W), drop_cls_token=False).logits
                    
                    # Unpatchify
                    x_rec = self.mae_decoder.unpatchify(mae_out_logits, (H, W)) # (B*V, 3, H, W)
                    
                    # Reshape to (B, V, 3, H, W) to match DPT format for consistency in denorm block below
                    x_rec = x_rec.reshape(b, v, 3, H, W)
                    
                    rgb = x_rec
        
        if not hasattr(rgb, 'device'):
            raise RuntimeError(f"'rgb' is not a tensor: {type(rgb)}")
        
        # Squeeze stereo dimension if present (DPT output might be 5D, MAE is matched above)
        if rgb.ndim == 5 and rgb.shape[1] == 1:
            rgb = rgb.squeeze(1)
        if depth is not None and depth.ndim == 5 and depth.shape[1] == 1:
            depth = depth.squeeze(1)
        if depth_conf is not None and depth_conf.ndim == 5 and depth_conf.shape[1] == 1:
            depth_conf = depth_conf.squeeze(1)
        
        # Denormalize RGB - handle both 4D and 5D shapes
        if rgb.ndim == 5:
            # (B, V, 3, H, W) - need to flatten for broadcasting
            b, v, c, h, w = rgb.shape
            rgb = rgb.reshape(b * v, c, h, w)
            rgb = rgb * self.encoder_std + self.encoder_mean
            rgb = rgb.reshape(b, v, c, h, w)
        else:
            # (B*V, 3, H, W) - standard case
            rgb = rgb * self.encoder_std + self.encoder_mean
        
        return {'rgb': rgb, 'depth': depth, 'depth_conf': depth_conf, 'ray': ray, 'ray_conf': ray_conf}
    
    def propagate_features(
        self,
        features: torch.Tensor,
        from_level: int,
        total_view: int = 1,
        cls_token: torch.Tensor = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Propagate shallow features to deeper levels via DA3 backbone forward.

        Norm flow: latent_norm → _denormalize → raw → backbone forward → layer_norm patches + raw cls

        Args:
            features: [latent_norm] Shallow level features (B*V, C, H, W)
            from_level: Starting level (0,1,2,3)
            total_view: Number of views for multi-view reshape
            cls_token: [latent_norm] Optional CLS token (B*V, C). If None, uses zeros.

        Returns:  [layer_norm patches, raw cls]
            List of (patches_4d, cls_3d) tuples for each level >= from_level
            patches_4d: (B, V, N, C)  — layer_norm (= [local, LayerNorm(current)])
            cls_3d: (B, V, C)         — raw (= [local, current], no LayerNorm)
        """
        layer_idx = self.OUT_LAYERS[from_level]
        
        # Handle shape: convert to (B*V, N+1, C) with CLS prepended
        b_tot, c, h, w = features.shape
        B = b_tot // total_view
        V = total_view
        # import pdb; pdb.set_trace()
        features = self._reshape_to_seq(features)  # (B*V, N, C)
        
        if cls_token is None:
            cls_token = torch.zeros(b_tot, c, device=features.device, dtype=features.dtype)

            
        features = torch.cat([cls_token.unsqueeze(1), features], dim=1).reshape(B, V, -1, c)
        features = self._denormalize(features)

        # Run encoder propagation
        raw_results = self.encoder.forward(
            features, 
            mode='from_layer', 
            start_layer_idx=layer_idx, 
            total_view=total_view,
            grid_size=(h, w) # Pass spatial dimensions to handle rectangular grids
        )
        
        # Reshape outputs for DPT decoder
        final_results = []
        for (cls_raw, patches_norm) in raw_results:
            b_tot = cls_raw.shape[0]
            batch_size = b_tot // total_view
            
            n, c = patches_norm.shape[1], patches_norm.shape[2]
            patches_4d = patches_norm.reshape(batch_size, total_view, n, c)
            cls_3d = cls_raw.reshape(batch_size, total_view, -1)
            
            final_results.append((patches_4d, cls_3d))
        
        # DEBUG: Log output stats (these are in DA3 backbone's native format)
        
        return final_results
    
