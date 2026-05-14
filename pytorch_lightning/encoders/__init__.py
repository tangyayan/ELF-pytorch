"""Encoder registry. Future LLaMA / GPT-2-large adapters register here."""

from typing import Dict, Type

from .base import EncoderInterface
from .t5 import T5Encoder


_REGISTRY: Dict[str, Type[EncoderInterface]] = {
    "t5-small": T5Encoder,
    "t5-base": T5Encoder,
    "t5-large": T5Encoder,
    # "llama":      LlamaEncoder,       # TODO: add adapter, parity-test, register
    # "gpt2-large": GPT2LargeEncoder,   # TODO: add adapter, parity-test, register
}


def build_encoder(name: str, **kwargs) -> EncoderInterface:
    """Factory: resolve `name` to a registered encoder class and construct it."""
    if name in _REGISTRY:
        cls = _REGISTRY[name]
    elif name.startswith("google-t5/"):
        cls = T5Encoder
    else:
        raise ValueError(
            f"No encoder registered for {name!r}. "
            f"Registered: {sorted(_REGISTRY)}. "
            f"To add a new pretrained encoder, implement EncoderInterface in "
            f"pytorch_lightning/encoders/<name>.py and register it here."
        )
    return cls(name, **kwargs)


__all__ = ["EncoderInterface", "T5Encoder", "build_encoder"]
