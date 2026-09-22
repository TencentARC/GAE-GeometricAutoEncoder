"""Stage 1: the GAE codec.

Paper Section 3.1. Trains a compact per-view latent between the frozen DA3
encoder and the frozen DA3 geometry head, then freezes it for Stage 2.

    frozen DA3 encoder  ->  level-wise normalize + concat  ->  GAECodec
    GAECodec latent     ->  rebuild 4 levels -> frozen DPT head -> depth/ray/pointmap
                        ->  learned RGBHead                     -> RGB
"""

from .da3 import DA3Backbone
from .gae_codec import GAECodec

__all__ = ["DA3Backbone", "GAECodec"]
