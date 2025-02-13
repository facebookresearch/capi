# type: ignore
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
# https://github.com/apple/ml-aim/tree/main/aim-v1/aim/v1/torch
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


class SinCosPosEmbed(nn.Module):
    def __init__(self, cls_token: bool = False):
        super().__init__()
        self.cls_token = cls_token

    def forward(self, h: int, w: int, embed_dim: int) -> torch.Tensor:
        assert embed_dim % 2 == 0, embed_dim

        grid_h = torch.arange(h).float()
        grid_w = torch.arange(w).float()
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([2, 1, h, w])

        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        pos_embed = torch.concatenate([emb_h, emb_w], dim=1)  # (H*W, D)
        if self.cls_token:
            pos_embed = torch.concatenate([torch.zeros([1, embed_dim]), pos_embed], dim=0)
        return pos_embed

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        omega = torch.arange(embed_dim // 2).float()
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = pos[:, None] * omega[None, :]  # (M, D/2), outer product

        emb_sin = torch.sin(out)  # (M, D/2)
        emb_cos = torch.cos(out)  # (M, D/2)

        emb = torch.concatenate([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)

        self.img_size, self.embed_dim = img_size, embed_dim
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class ViTPreprocessor(nn.Module):
    def __init__(
        self,
        patchifier: PatchEmbed,
        drop_patches: bool = False,
        cls_token: bool = False,
        pos_embed_type: Literal["sincos", "absolute"] = "sincos",
    ):
        super().__init__()
        self.patchifier = patchifier
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.patchifier.embed_dim)) if cls_token else None
        if pos_embed_type == "sincos":
            self.pos_embed = SinCosPosEmbed(cls_token)
        else:
            shape = (
                1,
                self.patchifier.num_patches + cls_token,
                self.patchifier.embed_dim,
            )
            self.pos_embed = nn.Parameter(torch.zeros(shape))
        self.drop_patches = drop_patches

        self.initialize_weights()

    def initialize_weights(self) -> None:
        if not isinstance(self.pos_embed, SinCosPosEmbed):
            torch.nn.init.normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d) following MAE
        if hasattr(self.patchifier, "proj"):
            w = self.patchifier.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, _, H, W = x.shape
        tokens = self.patchifier(x)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)
        B, N, D = tokens.shape

        if callable(self.pos_embed):
            p_h, p_w = self.patchifier.patch_size
            pos_embed = self.pos_embed(H // p_h, W // p_w, D).unsqueeze(0)
        else:
            pos_embed = self.pos_embed
        pos_embed = pos_embed.to(tokens.device)

        tokens = tokens + pos_embed[:, :N]

        if self.drop_patches and mask is not None:
            if self.cls_token is not None:
                cls_token, tokens = tokens[:, :1], tokens[:, 1:]
            tokens = tokens[~mask].reshape(B, -1, D)
            if self.cls_token is not None:
                tokens = torch.cat([cls_token, tokens], dim=1)

        return tokens


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_bias: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.is_causal = is_causal

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, **_: Any) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        use_bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=use_bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=use_bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_target: Callable[[bool], nn.Module],
        norm_layer: Callable[[int], nn.Module],
        ffn_target: Callable[..., nn.Module] = MLP,
        mlp_hidden_dim: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        ffn_dropout_rate: float = 0.0,
        use_bias: bool = True,
    ):
        assert not isinstance(
            attn_target,
            nn.Module,
        ), "attn_target should be a callable. Otherwise attn_target is shared across blocks!"
        assert not isinstance(
            ffn_target,
            nn.Module,
        ), "ffn_target should be a callable. Otherwise ffn_target is shared across blocks!"

        super().__init__()
        self.attn = attn_target(use_bias)

        self.norm_1 = norm_layer(dim)
        self.mlp = ffn_target(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
            use_bias=use_bias,
        )
        self.norm_2 = norm_layer(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # pre-norm
        x = x + self.attn(self.norm_1(x), mask=mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class AverageLayers(nn.Module):
    def __init__(self, layers: Sequence[int], reduce: bool = False):
        super().__init__()
        self.layers = layers
        self.reduce = reduce

    def forward(self, _: torch.Tensor, layer_features: list[torch.Tensor]) -> torch.Tensor:
        layer_features = [layer_features[layer_id] for layer_id in self.layers]
        feats = torch.stack(layer_features, dim=-1).mean(dim=-1)

        return feats.mean(dim=1) if self.reduce else feats

    @property
    def max_block_id(self) -> int:
        return max(self.layers)


class Transformer(nn.Module):
    def __init__(
        self,
        attn_target: Callable[[bool], nn.Module],
        embed_dim: int,
        num_blocks: int,
        norm_layer: Callable[[int], nn.Module],
        ffn_target: Callable[..., nn.Module] = MLP,
        post_transformer_layer: nn.Module | None = None,
        mlp_ratio: int = 4,
        mlp_hidden_dim: int | None = None,
        ffn_dropout_rate: float = 0.0,
        use_bias: bool = False,
        post_trunk_norm: bool = True,
    ):
        super().__init__()
        if mlp_hidden_dim is None:
            mlp_hidden_dim = int(mlp_ratio * embed_dim)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    ffn_target=ffn_target,
                    mlp_hidden_dim=mlp_hidden_dim,
                    norm_layer=norm_layer,
                    ffn_dropout_rate=ffn_dropout_rate,
                    use_bias=use_bias,
                )
                for _ in range(num_blocks)
            ],
        )
        self.post_trunk_norm = norm_layer(embed_dim) if post_trunk_norm else None
        self.post_transformer_layer = post_transformer_layer

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        max_block_id: int | None = -1,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | list[torch.Tensor]:
        # only evaluate up to the max block id
        if max_block_id is None:
            assert self.post_transformer_layer is not None, "Unable to determine the max block id."
            max_block_id = self.post_transformer_layer.max_block_id

        features = []
        for blk_id, blk in enumerate(self.blocks):
            tokens = blk(tokens, mask=mask)
            features.append(tokens)

            if blk_id == max_block_id:
                break

        if return_features:
            return features

        if self.post_trunk_norm is not None:
            tokens = self.post_trunk_norm(tokens)

        if self.post_transformer_layer is not None:
            tokens = self.post_transformer_layer(tokens, layer_features=features)

        return tokens, features


class AIM(nn.Module):
    def __init__(self, preprocessor: nn.Module, trunk: nn.Module, head: nn.Module, layer_to_use: int = 18):
        super().__init__()
        self.preprocessor = preprocessor
        self.trunk = trunk
        self.head = head
        self._layer_to_use = layer_to_use
        self.patch_size = self.preprocessor.patchifier.patch_size[0]

    def forward(
        self,
        x: Float[Tensor, "b c h w"],
    ) -> tuple[
        Float[Tensor, "b d"],
        Float[Tensor, "b k d"] | None,
        Float[Tensor, "b ih iw d"],
    ]:
        bs, _, h, w = x.shape
        x = self.preprocessor(x, mask=None)
        feats = self.trunk(x, mask=None, max_block_id=-1, return_features=True)
        features = feats[self._layer_to_use]
        patch_h, patch_w = self.preprocessor.patchifier.patch_size
        return features.mean(axis=1), None, features.reshape(bs, h // patch_h, w // patch_w, -1)


def _get_attention_target(dim: int, num_heads: int) -> Callable[[bool], nn.Module]:
    def callback(use_bias: bool) -> nn.Module:
        return Attention(dim=dim, num_heads=num_heads, use_bias=use_bias)

    return callback


def _aim(
    img_size: int | tuple[int, int],
    patch_size: int | tuple[int, int],
    embed_dim: int,
    num_blocks: int,
    num_heads: int,
    num_channels: int = 3,
    probe_layers: int | tuple[int, ...] = 6,
    **kwargs: Any,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    # preprocessor
    patchifier = PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=num_channels,
        embed_dim=embed_dim,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    preprocessor = ViTPreprocessor(patchifier, drop_patches=False, cls_token=False)

    # trunk
    if isinstance(probe_layers, int):
        probe_layers = tuple(range(num_blocks - probe_layers, num_blocks))
    assert all(layer >= 0 for layer in probe_layers), probe_layers

    attn_target = _get_attention_target(dim=embed_dim, num_heads=num_heads)
    post_transform_layer = AverageLayers(probe_layers, reduce=False)
    trunk = Transformer(
        attn_target,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        post_transformer_layer=post_transform_layer,
        **kwargs,
    )

    return preprocessor, trunk, None


def aim_600M(img_size: int | tuple[int, int] = 224, **kwargs: Any) -> AIM:
    preprocessor, trunk, head = _aim(
        img_size=img_size,
        patch_size=14,
        embed_dim=1536,
        num_blocks=24,
        num_heads=12,
        **kwargs,
    )
    return AIM(preprocessor, trunk, head)


def aim_1B(img_size: int | tuple[int, int] = 224, **kwargs: Any) -> AIM:
    preprocessor, trunk, head = _aim(
        img_size=img_size,
        patch_size=14,
        embed_dim=2048,
        num_blocks=24,
        num_heads=16,
        **kwargs,
    )
    return AIM(preprocessor, trunk, head)


def aim_3B(img_size: int | tuple[int, int] = 224, patch_size: int = 14, **kwargs: Any) -> AIM:
    preprocessor, trunk, head = _aim(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=3072,
        num_blocks=24,
        num_heads=24,
        **kwargs,
    )
    return AIM(preprocessor, trunk, head)


def aim_7B(img_size: int | tuple[int, int] = 224, patch_size: int = 14, **kwargs: Any) -> AIM:
    preprocessor, trunk, head = _aim(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=4096,
        num_blocks=32,
        num_heads=32,
        **kwargs,
    )
    return AIM(preprocessor, trunk, head)


model_dict = {
    "aim_600M": (
        aim_600M,
        "https://huggingface.co/apple/AIM/resolve/main/aim_600m_2bimgs_attnprobe_backbone.pth",
        "",
    ),
    "aim_1B": (
        aim_1B,
        "https://huggingface.co/apple/AIM/resolve/main/aim_1b_5bimgs_attnprobe_backbone.pth",
        "",
    ),
    "aim_3B": (
        aim_3B,
        "https://huggingface.co/apple/AIM/resolve/main/aim_3b_5bimgs_attnprobe_backbone.pth",
        "",
    ),
    "aim_7B": (
        aim_7B,
        "https://huggingface.co/apple/AIM/resolve/main/aim_7b_5bimgs_attnprobe_backbone.pth",
        "",
    ),
}


def __model_loader__(model_name: str, device: str = "cuda") -> nn.Module:
    model_fn, url, *keys, prefix = model_dict[model_name]
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    for k in keys:
        state_dict = state_dict[k]
    state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
    model = model_fn()
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device=device)
    return model
