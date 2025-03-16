"""
Loader for MIM-Refiner compatible with the evaluation scrpts.

Original repo and available models: https://github.com/ml-jku/MIM-Refiner.
"""

from typing import Any
import torch
from jaxtyping import Float
from torch import Tensor, nn


class MimRefinerWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self._wrapped = module

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._wrapped, name)

    def forward(
        self,
        x: Float[Tensor, "b c h w"],
    ) -> tuple[
        Float[Tensor, "b d"],
        None,
        Float[Tensor, "b ih iw d"],
    ]:
        b, rgb, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"Input shape [{h}, {w}] must be divisible by patch size {self.patch_size}")
        ih = h // self.patch_size
        iw = w // self.patch_size
        out = self._wrapped(x)  # [b, 1 + ih * iw, d]
        cls_token, patch_tokens = torch.split_with_sizes(out, [1, ih * iw], dim=1)
        return cls_token.squeeze(1), None, patch_tokens.unflatten(1, (ih, iw))


def __model_loader__(model_name: str, device: str = "cuda") -> nn.Module:
    model = torch.hub.load("ml-jku/MIM-Refiner", model_name)
    model = MimRefinerWrapper(model)
    model.eval()
    model.to(device)
    model.requires_grad_(False)
    return model
