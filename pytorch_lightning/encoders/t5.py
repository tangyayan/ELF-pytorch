"""T5 encoder adapter.

Wraps `transformers.T5EncoderModel` so the ELF model can talk to it through
`EncoderInterface`. We deliberately do NOT touch any internal op of the T5
encoder — no FlashAttention, no layernorm fusion, no precision change inside
it. The reason is that the encoder is a frozen, externally-defined component
and its output must reproduce the original pretrained model bit-for-bit.

The JAX original loads from a `.pkl` (`embedded-language-flows/t5_small_encoder_jax/...`)
which is a re-pack of `google-t5/t5-small`. We use the upstream HF weights
directly — same numbers, simpler dependency.
"""

import torch
from transformers import T5EncoderModel

from .base import EncoderInterface


_T5_ALIASES = {
    "t5-small": "google-t5/t5-small",
    "t5-base": "google-t5/t5-base",
    "t5-large": "google-t5/t5-large",
}


class T5Encoder(EncoderInterface):
    """Frozen T5 encoder. NO internal op is changed — exact reproduction of HF T5.

    Notes:
      * Runs in fp32 (matches ELF JAX). Do not override unless you also pass a
        parity check against the fp32 reference.
      * Supports 2D or 3D `attention_mask` — HF's `get_extended_attention_mask`
        handles both (the collator uses 3D for the cond/target attention layout).
    """

    def __init__(self, model_name: str, dtype: torch.dtype = torch.float32):
        super().__init__(model_name=model_name)
        resolved = _T5_ALIASES.get(model_name, model_name)
        # EXACT REPRODUCTION: load the upstream HF weights as-is.
        self.model = T5EncoderModel.from_pretrained(resolved, torch_dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.d_model = self.model.config.d_model

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # NO FlashAttention, no fused norm, no precision change here.
        # See the docstring above: exact pretrained-encoder reproduction.
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask).last_hidden_state
