"""
Config Utilities for Dynamic Parameter Computation.

This module provides utility functions to automatically compute derived config
parameters from base settings, reducing redundancy and potential inconsistencies.
"""
from typing import Tuple, Union, Optional
from omegaconf import OmegaConf, DictConfig, ListConfig
import math


def parse_encoder_size(size_config) -> Tuple[int, int]:
    """
    Parse encoder input size from config, supporting both scalar and 2D (H, W) formats.
    
    Args:
        size_config: Either an int (square), a list/tuple [H, W], or a dict {height: H, width: W}
        
    Returns:
        Tuple[int, int]: (height, width)
        
    Examples:
        >>> parse_encoder_size(252)
        (252, 252)
        >>> parse_encoder_size([504, 336])
        (504, 336)
        >>> parse_encoder_size({'height': 504, 'width': 336})
        (504, 336)
    """
    if size_config is None:
        return (252, 252)  # Default for DA3
    
    if isinstance(size_config, (int, float)):
        size = int(size_config)
        return (size, size)
    
    if isinstance(size_config, (list, tuple, ListConfig)):
        if len(size_config) == 1:
            return (int(size_config[0]), int(size_config[0]))
        if len(size_config) >= 2:
            return (int(size_config[0]), int(size_config[1]))
    
    if isinstance(size_config, dict) or (hasattr(size_config, 'get')):
        h = size_config.get('height', size_config.get('h', 252))
        w = size_config.get('width', size_config.get('w', h))
        return (int(h), int(w))
    
    raise ValueError(f"Cannot parse encoder_input_size: {size_config}")


def init_config_defaults(
    rae_config: DictConfig,
    model_config: DictConfig,
    misc_config: Optional[DictConfig] = None,
    patch_size: int = 14,
    is_da3: bool = True,
) -> Tuple[int, int]:
    """
    Initialize derived config parameters from base settings.
    
    This function computes and sets default values for parameters that can be
    derived from `encoder_input_size`, reducing config redundancy.
    
    Args:
        rae_config: Stage 1 (RAE) configuration
        model_config: Stage 2 (DiT/DDT) configuration  
        misc_config: Miscellaneous configuration (optional)
        patch_size: Patch size for ViT (14 for DA3/DINO)
        is_da3: Whether this is DA3 (affects default in_channels)
        
    Returns:
        Tuple[int, int]: (encoder_height, encoder_width) for use elsewhere
        
    Side Effects:
        Modifies model_config and misc_config in-place with computed defaults.
    """
    # Parse encoder input size (supports scalar or 2D)
    rae_params = rae_config.get('params', {})
    size_config = rae_params.get('encoder_input_size', 252 if is_da3 else 224)
    encoder_h, encoder_w = parse_encoder_size(size_config)
    
    # Validate divisibility by patch_size
    if encoder_h % patch_size != 0:
        raise ValueError(
            f"encoder_input_size height ({encoder_h}) must be divisible by patch_size ({patch_size})"
        )
    if encoder_w % patch_size != 0:
        raise ValueError(
            f"encoder_input_size width ({encoder_w}) must be divisible by patch_size ({patch_size})"
        )
    
    # Compute latent dimensions
    latent_h = encoder_h // patch_size
    latent_w = encoder_w // patch_size
    
    # Get in_channels from model config
    model_params = model_config.get('params', {})
    default_in_channels = 1536 if is_da3 else 768
    in_channels = model_params.get('in_channels', default_in_channels)
    
    # -------------------------------------------------------------------------
    # Set Stage 2 defaults if not specified
    # -------------------------------------------------------------------------
    if 'params' not in model_config:
        model_config['params'] = {}
    
    # input_size: For square latents, use single value; for non-square, use tuple
    if 'input_size' not in model_config.params:
        if latent_h == latent_w:
            model_config.params['input_size'] = latent_h
        else:
            model_config.params['input_size'] = [latent_h, latent_w]
    
    # cam_input_size: Same as encoder_input_size
    if 'cam_input_size' not in model_config.params:
        if encoder_h == encoder_w:
            model_config.params['cam_input_size'] = encoder_h
        else:
            model_config.params['cam_input_size'] = [encoder_h, encoder_w]
    
    # cam_patch_size: Always patch_size (14 for DA3/DINO)
    if 'cam_patch_size' not in model_config.params:
        model_config.params['cam_patch_size'] = patch_size
    
    # -------------------------------------------------------------------------
    # Set Misc defaults if not specified
    # -------------------------------------------------------------------------
    if misc_config is not None:
        # latent_size: [C, H, W]
        if 'latent_size' not in misc_config:
            misc_config['latent_size'] = [in_channels, latent_h, latent_w]
        
        # time_dist_shift_dim: C * H * W
        if 'time_dist_shift_dim' not in misc_config:
            misc_config['time_dist_shift_dim'] = in_channels * latent_h * latent_w
        
        # time_dist_shift_base: Default 4096
        if 'time_dist_shift_base' not in misc_config:
            misc_config['time_dist_shift_base'] = 4096
    
    return (encoder_h, encoder_w)


def get_image_size_from_config(rae_config: DictConfig, default: int = 252) -> Tuple[int, int]:
    """
    Extract default image size from RAE config.
    
    Warning: This returns the *model's* expected encoder size.
    Dataloaders strictly require 'dataset.image_size' to be set explicitly.
    Do not use this for dataloader configuration in strict mode.
    
    Args:
        rae_config: Stage 1 (RAE) configuration
        default: Default size if not specified
        
    Returns:
        Tuple[int, int]: (height, width)
    """
    rae_params = rae_config.get('params', {})
    size_config = rae_params.get('encoder_input_size', default)
    return parse_encoder_size(size_config)
