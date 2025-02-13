# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import logging
from urllib.parse import urlsplit

import timm
import torch
from jaxtyping import Float
from torch import Tensor, nn

logger = logging.getLogger(__file__)


class TimmAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        # I don't understand if there is a general way to access global vector + feature map in timm
        # So for now we limit ourselves to vit
        self._model: timm.models.VisionTransformer = model
        self.patch_size = model.patch_embed.patch_size[0]  # no support for rectangular patch embed rn
        self.embed_dim = model.embed_dim

    def forward(  # type: ignore
        self,
        x: Float[Tensor, "b c h w"],
    ) -> tuple[
        Float[Tensor, "b d"],
        Float[Tensor, "b k d"] | None,
        Float[Tensor, "b ih iw d"],
    ]:
        B, _, h, w = x.shape
        x = self._model.forward_features(x)
        patches = x[:, self._model.num_prefix_tokens :].reshape(B, h // self.patch_size, w // self.patch_size, -1)
        return self._model.forward_head(x, pre_logits=True), None, patches


tested_models = [
    "vit_small_patch8_224.dino",
    "vit_small_patch16_224.dino",
    "vit_base_patch8_224.dino",
    "vit_base_patch16_224.dino",
]


def __model_loader__(
    model_name: str,
    pretrained_weights: str | None = None,
    device: str = "cuda",
    pretrained_weights_key: str | None = None,
    **timm_kwargs,
) -> nn.Module:
    model: nn.Module = timm.models._factory.create_model(model_name, **timm_kwargs)
    if isinstance(pretrained_weights, str) and pretrained_weights != "":
        # load ckpt, local or remote, into memory
        parsed = urlsplit(pretrained_weights)
        loader = torch.load if parsed.scheme == "" else torch.hub.load_state_dict_from_url
        sd = loader(pretrained_weights, map_location="cpu", weights_only=True)
        if pretrained_weights_key is not None:
            sd = sd[pretrained_weights_key]
        # load into model
        msg = model.load_state_dict(sd, strict=False)
        logger.info(f"Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}")
    model = TimmAdapter(model)
    model.eval()
    model.to(device=device)
    return model
