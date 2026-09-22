"""Stage 2: conditional flow matching in the frozen GAE latent.

Paper Sections 3.2-3.3. The codec from Stage 1 is frozen, its posterior mean is
standardized per channel, and a single conditional flow model is trained over
that state. Only generated views evolve as ODE states; reference views, camera
rays and text are controls.
"""

from .models.dit import GAEFlow
from .transport.flow import convert_x_to_v, training_losses_rae

__all__ = ["GAEFlow", "convert_x_to_v", "training_losses_rae"]
