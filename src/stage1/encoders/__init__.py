"""Encoder registry for Stage 1 backbones.

Only the DA3 encoder is shipped: GAE is defined over the Depth Anything 3
hierarchy. (The research tree also carried a VGGT encoder for a separate
baseline; it is not part of GAE and is omitted here.)
"""

from typing import Callable, Dict, Optional, Type, Union

ARCHS: Dict[str, Type] = {}
__all__ = ["ARCHS", "register_encoder"]


def _add_to_registry(name: str, cls: Type) -> Type:
    if name in ARCHS and ARCHS[name] is not cls:
        raise ValueError(f"Encoder '{name}' is already registered.")
    ARCHS[name] = cls
    return cls


def register_encoder(
    cls: Optional[Type] = None, *, name: Optional[str] = None
) -> Union[Callable[[Type], Type], Type]:
    """Register an encoder class in ``ARCHS``."""

    def decorator(inner_cls: Type) -> Type:
        encoder_name = name or inner_cls.__name__
        return _add_to_registry(encoder_name, inner_cls)

    if cls is None:
        return decorator

    return decorator(cls)


from . import da3  # noqa: E402  performs registration on import
