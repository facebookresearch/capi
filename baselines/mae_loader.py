# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
from functools import partial

import timm.models.vision_transformer
import torch
from jaxtyping import Float
from torch import Tensor, nn


# Class from https://github.com/facebookresearch/mae/blob/main/models_vit.py
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = self.patch_embed.patch_size[0]

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    # custom fwd
    def forward(  # type: ignore
        self,
        x: Float[Tensor, "b c h w"],
    ) -> tuple[
        Float[Tensor, "b d"],
        Float[Tensor, "b k d"] | None,
        Float[Tensor, "b ih iw d"],
    ]:
        bs, _, h, w = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        patch_h, patch_w = self.patch_embed.patch_size
        return outcome, None, x[:, 1:].reshape(bs, h // patch_h, w // patch_w, -1)


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        num_classes=0,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        num_classes=0,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        num_classes=0,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


model_dict = {
    "vit_base_patch16": (vit_base_patch16, "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"),
    "vit_large_patch16": (vit_large_patch16, "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth"),
    "vit_huge_patch14": (vit_huge_patch14, "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth"),
}


def __model_loader__(model_name: str, device: str = "cuda") -> nn.Module:
    model_fn, url = model_dict[model_name]
    model = model_fn(dynamic_img_size=True)
    model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location="cpu")["model"])
    model.eval()
    model.to(device=device)
    return model
