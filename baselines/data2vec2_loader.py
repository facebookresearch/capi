# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
from functools import partial
from types import MethodType
from urllib.parse import quote_plus

import timm
import torch
from jaxtyping import Float
from timm.layers.patch_embed import PatchEmbed
from timm.models.vision_transformer import ResPostBlock, VisionTransformer
from torch import Tensor, nn


class WeirdD2V2Block(ResPostBlock):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.mlp(x))
        return x


def patched_forward(
    self: nn.Module,
    x: Float[Tensor, "b c h w"],
) -> tuple[
    Float[Tensor, "b d"],
    Float[Tensor, "b k d"] | None,
    Float[Tensor, "b ih iw d"],
]:
    bs, _, h, w = x.shape
    patches, cls_tok = self.forward_intermediates(
        x,
        1,
        return_prefix_tokens=True,
        norm=True,
        output_fmt="NLC",
        intermediates_only=True,
    )[0]
    patches = patches.reshape(bs, h // self.patch_size, w // self.patch_size, -1)
    return cls_tok[:, 0], None, patches


def patch_embed_with_bias(*args, **kwargs):
    if "bias" in kwargs:
        del kwargs["bias"]
    return PatchEmbed(*args, **kwargs, bias=True)


def create_d2v2(model_name):
    model: VisionTransformer = timm.create_model(  # type: ignore
        model_name,
        num_classes=0,
        block_fn=WeirdD2V2Block,
        embed_layer=patch_embed_with_bias,
        no_embed_class=True,
        weight_init="skip",
        pre_norm=True,
        final_norm=False,
        dynamic_img_size=True,
    )
    model.forward = MethodType(patched_forward, model)
    model.patch_size = model.patch_embed.patch_size[0]
    return model


def d2v2_converter(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    sd["cls_token"] = sd.pop("modality_encoders.IMAGE.extra_tokens")
    sd["pos_embed"] = sd.pop("modality_encoders.IMAGE.fixed_positional_encoder.positions")
    sd["patch_embed.proj.weight"] = sd.pop("modality_encoders.IMAGE.local_encoder.proj.weight")
    sd["patch_embed.proj.bias"] = sd.pop("modality_encoders.IMAGE.local_encoder.proj.bias")
    sd["norm_pre.weight"] = sd.pop("modality_encoders.IMAGE.context_encoder.norm.weight")
    sd["norm_pre.bias"] = sd.pop("modality_encoders.IMAGE.context_encoder.norm.bias")
    for k in list(sd.keys()):
        if "modality_encoders.IMAGE.decoder" in k:
            sd.pop(k)
    if "_ema" in sd:
        sd.pop("_ema")
    return sd


model_dict = {
    "data2vec2_vitb16": (
        partial(
            create_d2v2,
            model_name="vit_base_patch16_224",
        ),
        "https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_imagenet.pt",
        "model",
        "",
    ),
    "data2vec2_vitl16": (
        partial(
            create_d2v2,
            model_name="vit_large_patch16_224",
        ),
        "https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_imagenet.pt",
        "model",
        "",
    ),
    "data2vec2_vith14": (
        partial(
            create_d2v2,
            model_name="vit_huge_patch14_224",
        ),
        "https://dl.fbaipublicfiles.com/fairseq/data2vec2/huge_imagenet.pt",
        "model",
        "",
    ),
    "data2vec2_vitb16_ema": (
        partial(
            create_d2v2,
            model_name="vit_base_patch16_224",
        ),
        "https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_imagenet.pt",
        "model",
        "_ema",
        "",
    ),
    "data2vec2_vitl16_ema": (
        partial(
            create_d2v2,
            model_name="vit_large_patch16_224",
        ),
        "https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_imagenet.pt",
        "model",
        "_ema",
        "",
    ),
    "data2vec2_vith14_ema": (
        partial(
            create_d2v2,
            model_name="vit_huge_patch14_224",
        ),
        "https://dl.fbaipublicfiles.com/fairseq/data2vec2/huge_imagenet.pt",
        "model",
        "_ema",
        "",
    ),
}


def __model_loader__(model_name: str, device: str = "cuda") -> nn.Module:
    model_fn, url, *keys, prefix = model_dict[model_name]
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", file_name=quote_plus(url))
    for k in keys:
        state_dict = state_dict[k]
    state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
    state_dict = d2v2_converter(state_dict)
    model = model_fn()
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device=device)
    return model
