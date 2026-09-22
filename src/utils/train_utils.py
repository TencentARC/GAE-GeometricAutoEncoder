from omegaconf import OmegaConf, DictConfig
from typing import List, Tuple



def parse_configs(config_path: str) -> Tuple[DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:
    """
    Load a config file and return component sections as DictConfigs.
    
    Returns:
        Tuple of (rae_config, stage2_config, transport_config, sampler_config,
                  guidance_config, pag_config, misc_config, training_config,
                  validation_config, multiview_config, dataset_config)
    """
    config = OmegaConf.load(config_path)
    rae_config = config.get("stage_1", None)
    stage2_config = config.get("stage_2", None)
    transport_config = config.get("transport", None)
    sampler_config = config.get("sampler", None)
    # Note: guidance and pag are now moved inside validation for new configs, 
    # but we keep top-level access for backward compatibility if needed.
    guidance_config = config.get("guidance", None)
    pag_config = config.get("pag", None)
    misc = config.get("misc", None)
    training_config = config.get("training", None)
    validation_config = config.get("validation", None)
    multiview_config = config.get("multiview", None)
    dataset_config = config.get("dataset", None)
    return rae_config, stage2_config, transport_config, sampler_config, guidance_config, pag_config, misc, training_config, validation_config, multiview_config, dataset_config

def none_or_str(value):
    if value == 'None':
        return None
    return value


################################################################################
#                           Common Training Utilities                           #
################################################################################

import torch
import torch.distributed as dist
import logging
import numpy as np
from PIL import Image
from collections import OrderedDict


def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    time_dist_type="uniform",
    time_dist_shift=1.0,
):
    """
    Create a Transport object for diffusion training.
    
    Args:
        path_type: 'Linear', 'GVP', or 'VP'
        prediction: 'velocity', 'noise', or 'score'
        loss_weight: 'velocity', 'likelihood', or None
        train_eps: Training epsilon (auto-set based on path_type if None)
        sample_eps: Sampling epsilon (auto-set based on path_type if None)
        time_dist_type: Time distribution type
        time_dist_shift: Time distribution shift
    
    Returns:
        Transport object
    """
    from stage2.transport.transport import Transport, ModelType, WeightType, PathType
    
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }
    path_type = path_choice[path_type]

    if path_type in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:
        train_eps = 0
        sample_eps = 0
    
    transport = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        time_dist_type=time_dist_type,
        time_dist_shift=time_dist_shift,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )
    
    return transport


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    
    Args:
        ema_model: The EMA model to update
        model: The source model
        decay: EMA decay factor
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name.startswith("_orig_mod."):
            name = name[10:]
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    
    Args:
        model: PyTorch model
        flag: Whether to require gradients
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """End DDP training by destroying the process group."""
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    
    Args:
        logging_dir: Directory to write log file (None for no file output)
    
    Returns:
        Logger instance
    """
    if logging_dir is not None and dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(), 
                logging.FileHandler(f"{logging_dir}/log.txt")
            ]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    
    Args:
        pil_image: PIL Image to crop
        image_size: Target size for both dimensions
    
    Returns:
        Cropped PIL Image
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])