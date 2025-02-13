# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
from types import MethodType
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn


def custom_fwd(
    self: nn.Module,
    x: Float[Tensor, "b c h w"],
) -> tuple[
    Float[Tensor, "b d"],
    Float[Tensor, "b k d"] | None,
    Float[Tensor, "b ih iw d"],
]:
    bs, _, h, w = x.shape
    x = self.prepare_tokens_with_masks(x, None)
    for blk in self.blocks:
        x = blk(x)
    x_norm = self.norm(x)
    cls_tok = x_norm[:, 0]
    registers = x_norm[:, 1 : 1 + self.num_register_tokens]
    patches = x_norm[:, 1 + self.num_register_tokens :]
    patches = patches.reshape(bs, h // self.patch_size, w // self.patch_size, -1)
    return cls_tok, registers, patches


dinov2_name_type = Literal[
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
    "dinov2_vits14_reg",
    "dinov2_vitb14_reg",
    "dinov2_vitl14_reg",
    "dinov2_vitg14_reg",
]


def __model_loader__(model_name: dinov2_name_type, device: str = "cuda") -> nn.Module:
    model: nn.Module = torch.hub.load("facebookresearch/dinov2:main", model_name, pretrained=True)  # pyright:ignore[reportAssignmentType]
    model.forward = MethodType(custom_fwd, model)
    model.eval()
    model.to(device=device)
    return model
