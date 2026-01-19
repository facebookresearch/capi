# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import logging
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

from utils import weight_norm

logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        mlp_ratio: int | float | None = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_features is None:
            assert mlp_ratio is not None
            hidden_features = int(in_features * mlp_ratio)
        else:
            assert mlp_ratio is None
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: Float[Tensor, "*b d"]) -> Float[Tensor, "*b d"]:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# from xformers
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rope(nn.Module):
    def __init__(
        self,
        dim: int,
        max_freq: float | int = 7,
        min_freq: float | int = 7e-4,
    ):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.freqs = nn.Parameter(torch.empty(2, self.dim))

    def _device_weight_init(self):
        # Create freqs in 1d
        freqs_1d = self.max_freq * (self.max_freq / self.min_freq) ** torch.linspace(0, -1, self.dim // 4)
        # duplicate freqs for rotation pairs of channels
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        # First half of channels do x, second half do y
        freqs_2d = torch.zeros(2, self.dim)
        freqs_2d[0, : self.dim // 2] = freqs_1d
        freqs_2d[1, -self.dim // 2 :] = freqs_1d
        # it's an angular freq here
        self.freqs.data.copy_(freqs_2d * 2 * torch.pi)

    def forward(self, x: Float[Tensor, "*b d"], coords: Float[Tensor, "*b 2"]) -> Float[Tensor, "*b d"]:
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        context_dim: int | None = None,
        rope_kwargs: Mapping = {},
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        context_dim = context_dim or dim

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = Rope(dim=head_dim, **rope_kwargs)

    def forward(
        self,
        x: Float[Tensor, "b n_q d"],
        coords: Float[Tensor, "b n_q 2"],
        context: Float[Tensor, "b n_k d_k"] | None = None,
        context_coords: Float[Tensor, "b n_k 2"] | None = None,
    ) -> Float[Tensor, "b n_q d"]:
        if context is None or context_coords is None:
            context = x
            context_coords = coords
        b, n_q, d = x.shape
        b, n_k, _ = context.shape
        h = self.num_heads

        q = self.q_proj(x).reshape(b, n_q, h, d // h).transpose(1, 2)
        k = self.k_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)
        v = self.v_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)
        q = self.rope(q, coords[:, None, :, :])
        k = self.rope(k, context_coords[:, None, :, :])
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape([b, n_q, d])
        x = self.proj(x)
        return x


class NaiveResidual(nn.Module):
    def __init__(self, drop_prob: float | int, norm: nn.Module, fn: nn.Module):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.keep_prob = 1 - drop_prob

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        **kwargs: Float[Tensor, "b ..."] | None,
    ) -> Float[Tensor, "b n d"]:
        fn_out = self.fn(self.norm(x), **kwargs)
        if self.keep_prob == 1.0 or not self.training:
            return x + fn_out
        mask = fn_out.new_empty(x.shape[0]).bernoulli_(self.keep_prob)[:, None, None]
        return x + fn_out * mask / self.keep_prob


class EfficientResidual(NaiveResidual):
    def forward(
        self,
        x: Float[Tensor, "b n d"],
        **kwargs: Float[Tensor, "b ..."] | None,
    ) -> Float[Tensor, "b n d"]:
        if self.keep_prob == 1.0 or not self.training:
            return x + self.fn(self.norm(x), **kwargs)
        b, _, _ = x.shape
        n_keep = max(int(b * self.keep_prob), 1)
        indices = torch.randperm(b, device=x.device)[:n_keep]
        for k, v in kwargs.items():
            if v is not None:
                kwargs[k] = v[indices]
        return torch.index_add(
            x,
            dim=0,
            source=self.fn(self.norm(x[indices]), **kwargs),
            index=indices,
            alpha=b / n_keep,
        )


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float | int,
        norm_layer: Callable[[int], nn.Module],
        context_dim: int | None,
        drop_path_type: str = "efficient",
        attn_kwargs: Mapping = {},
    ) -> None:
        super().__init__()
        residual_module = {
            "naive": NaiveResidual,
            "efficient": EfficientResidual,
        }[drop_path_type]
        self.residual1 = residual_module(
            drop_path,
            norm_layer(dim),
            Attention(
                dim,
                context_dim=context_dim,
                **attn_kwargs,
            ),
        )
        self.residual2 = residual_module(
            drop_path,
            norm_layer(dim),
            Mlp(in_features=dim),
        )

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        context: Float[Tensor, "b n_k d_k"] | None = None,
        coords: Float[Tensor, "b n 2"] | None = None,
        context_coords: Float[Tensor, "b n_k 2"] | None = None,
    ) -> Float[Tensor, "b n d"]:
        x = self.residual1(
            x,
            context=context,
            coords=coords,
            context_coords=context_coords,
        )
        x = self.residual2(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        norm_layer: Callable[[int], nn.Module],
        depth: int,
        drop_path_rate: float | int,
        context_dim: int | None = None,
        block_kwargs: Mapping[str, Any] = {},
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_blocks = depth

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    context_dim=context_dim,
                    **block_kwargs,
                )
                for i in range(depth)
            ],
        )

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        return_layers: set[int],
        contexts: list[Float[Tensor, "b n_k d_k"]] | None = None,
        coords: Float[Tensor, "b n 2"] | None = None,
        context_coords: Float[Tensor, "b n_k 2"] | None = None,
    ) -> dict[int, Float[Tensor, "b n d"]]:
        outputs = {}
        if 0 in return_layers:
            outputs[0] = x
        for blk_idx, blk in enumerate(self.blocks):
            context = contexts[blk_idx] if contexts is not None else None
            x = blk(
                x,
                context=context,
                coords=coords,
                context_coords=context_coords,
            )
            if blk_idx + 1 in return_layers:
                outputs[blk_idx + 1] = x
        return outputs


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        in_chans: int = 3,
        norm_layer_type: str = "RMSNorm",
        n_registers: int = 16,
        register_coordinates: str = "edge",
        scale_reg_to_visible: bool = True,
        transformers_kwargs: Mapping[str, Any] = {},
        encoder_kwargs: Mapping[str, Any] = {},
        decoder_kwargs: Mapping[str, Any] = {},
        norm_layer_kwargs: Mapping[str, Any] = {"eps": 1e-5},
        final_norm_kwargs: Mapping[str, Any] = {"elementwise_affine": False},
        out_layer: int = -1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_prefix_tokens = self.n_registers = n_registers
        self.register_coordinates = register_coordinates
        self.scale_reg_to_visible = scale_reg_to_visible

        norm_layer: Callable[[int], nn.Module] = partial(getattr(torch.nn, norm_layer_type), **norm_layer_kwargs)
        self.encoder = Transformer(
            **transformers_kwargs,
            **encoder_kwargs,
            norm_layer=norm_layer,
        )
        self.decoder = Transformer(
            **transformers_kwargs,
            **decoder_kwargs,
            context_dim=self.encoder.embed_dim,
            norm_layer=norm_layer,
        )
        self.embed_dim = self.encoder.embed_dim
        self.pred_dim = self.decoder.embed_dim
        self.n_blocks = len(self.encoder.blocks)
        self.out_layer = out_layer % (len(self.encoder.blocks) + 1)

        self.mask_token = nn.Parameter(torch.empty(1, self.decoder.embed_dim))
        self.registers = nn.Parameter(torch.empty(1, n_registers, self.encoder.embed_dim))
        self.patch_embed = nn.Conv2d(in_chans, self.encoder.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.enc_norm = norm_layer(self.embed_dim, **final_norm_kwargs)
        self.dec_norm = norm_layer(self.decoder.embed_dim, **final_norm_kwargs)
        logger.debug(f"Created {type(self).__name__}: {self}")

    def init_weights(self):
        self.patch_embed.reset_parameters()
        w = self.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.registers, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)
        return self

    def prepare_tokens_and_drop(
        self,
        x: Float[Tensor, "b c h w"],
        visible_indices: Int[Tensor, " n_visible_total"] | None,
    ) -> tuple[
        Float[Tensor, "b n_visible d"],
        Float[Tensor, "b n_visible 2"],
        Float[Tensor, "b n_tokens 2"],
    ]:
        b, _, h, w = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        # compute coord for ropes
        coord_x = torch.linspace(0, 1, h // self.patch_size, device=x.device, dtype=x.dtype)
        coord_y = torch.linspace(0, 1, w // self.patch_size, device=x.device, dtype=x.dtype)
        coords_all = torch.cartesian_prod(coord_x, coord_y)
        # Mask out
        coords_all = coords_all[None].expand(b, -1, -1)
        if visible_indices is not None:
            coords = coords_all.flatten(0, 1)[visible_indices].reshape(b, -1, 2)
            x = x.flatten(0, 1)[visible_indices].reshape(b, -1, self.embed_dim)
        else:
            coords = coords_all
        # add reg
        if self.register_coordinates == "zeros":
            reg_coords = torch.zeros(b, self.n_registers, 2, device=x.device, dtype=x.dtype)
        elif self.register_coordinates == "edge":
            reg_coords = self.get_edge_coordinates(self.n_registers, x.dtype, x.device).expand(b, -1, -1)
        else:
            raise ValueError(self.register_coordinates)
        if self.scale_reg_to_visible:
            mi, ma = coords.min(dim=1, keepdim=True).values, coords.max(dim=1, keepdim=True).values
            reg_coords = mi + reg_coords * (ma - mi)
        x = torch.cat([self.registers.expand(b, -1, -1), x], dim=1)
        coords = torch.cat([reg_coords, coords], dim=1)
        return x, coords, coords_all

    def get_edge_coordinates(self, n: int, dtype: torch.dtype, device: torch.device) -> Float[Tensor, "1 n 2"]:
        side = n // 4
        assert n == 4 * side
        reg_coords = torch.zeros(1, n, 2, dtype=dtype, device=device)
        c = torch.arange(side, dtype=dtype, device=device) / side
        reg_coords[:, 0 * side : 1 * side, 0] = c
        reg_coords[:, 0 * side : 1 * side, 1] = 0
        reg_coords[:, 1 * side : 2 * side, 0] = 1
        reg_coords[:, 1 * side : 2 * side, 1] = c
        reg_coords[:, 2 * side : 3 * side, 0] = 1 - c
        reg_coords[:, 2 * side : 3 * side, 1] = 1
        reg_coords[:, 3 * side : 4 * side, 0] = 0
        reg_coords[:, 3 * side : 4 * side, 1] = 1 - c
        return reg_coords

    def forward_features(
        self,
        x: Float[Tensor, "b c h w"],
        visible_indices: Int[Tensor, " n_visible_total"] | None,
        predict_indices: Int[Tensor, " n_predict_total"] | None,
        enc_layer: int,
        dec_layer: int | None,
    ) -> tuple[Float[Tensor, "b n_visible d"], Float[Tensor, "b n_predict d"] | None]:
        b, _, _, _ = x.shape
        # prepare and drop
        x, coords_enc, coords_all = self.prepare_tokens_and_drop(x, visible_indices)
        # these are the layers we need
        enc_layers = {enc_layer}
        if dec_layer is not None:
            enc_layers.add(len(self.encoder.blocks))
        # encoder fwd
        encoder_outputs = self.encoder(x, coords=coords_enc, return_layers=enc_layers)
        encoder_outputs = {k: self.enc_norm(v) for k, v in encoder_outputs.items()}
        # decoder fwd
        if dec_layer is not None:
            # if predict_indices is None, we use all positions
            coords_dec = coords_all.flatten(0, 1)[predict_indices].reshape(b, -1, 2)
            decoder_outputs = self.decoder(
                self.mask_token[None].expand(*coords_dec.shape[:2], -1),
                contexts=[encoder_outputs[len(self.encoder.blocks)]] * self.decoder.n_blocks,
                coords=coords_dec,
                context_coords=coords_enc,
                return_layers={dec_layer},
            )
            dec_out = self.dec_norm(decoder_outputs[dec_layer])
        else:
            dec_out = None
        logger.debug(f"Taking layer {enc_layer}")
        enc_out = encoder_outputs[enc_layer]
        return (enc_out, dec_out)

    def forward_pretrain(
        self,
        x: Float[Tensor, "b c h w"],
        visible_indices: Int[Tensor, " n_visible_total"] | None = None,
        predict_indices: Int[Tensor, " n_predict_total"] | None = None,
        do_prediction: bool = False,
    ) -> tuple[Float[Tensor, "b n_visible d"], Float[Tensor, "n_predict_total d"] | None]:
        """Used at train time"""
        encoder_output, decoder_output = self.forward_features(
            x,
            visible_indices,
            predict_indices,
            enc_layer=self.out_layer,
            dec_layer=len(self.decoder.blocks) if do_prediction else None,
        )
        if decoder_output is not None:
            decoder_output = decoder_output.flatten(0, 1)
        return encoder_output[:, self.num_prefix_tokens :], decoder_output

    def forward(
        self,
        x: Float[Tensor, "b c h w"],
    ) -> tuple[Float[Tensor, "b d"], Float[Tensor, "b k d"], Float[Tensor, "b ih iw d"]]:
        """Used at eval time"""
        b, _, h, w = x.shape
        # uses `indices=None` to get all positions
        enc_out, dec_out = self.forward_features(x, None, None, enc_layer=self.out_layer, dec_layer=1)
        global_repr = dec_out.mean(dim=1)  # pyright:ignore[reportOptionalMemberAccess]
        registers = enc_out[:, : self.num_prefix_tokens]
        feature_map = enc_out[:, self.num_prefix_tokens :].reshape(
            b,
            h // self.patch_size,
            w // self.patch_size,
            self.embed_dim,
        )
        return (global_repr, registers, feature_map)


# from https://github.com/facebookresearch/mae/blob/main/models_mae.py
def _init_weights(m: nn.Module, xavier_gain=1) -> None:
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight, gain=xavier_gain)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm | nn.RMSNorm) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if hasattr(m, "_device_weight_init"):
        m._device_weight_init()


class L2NormLinear(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        *,
        do_weight_norm=True,
    ):
        super().__init__()
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
        trunc_normal_(self.last_layer.weight, std=0.02)
        if do_weight_norm:
            self.last_layer = weight_norm(self.last_layer)
            self.last_layer.weight_g.data.fill_(1)
            # ...
            # So, do_weight_norm is True apparently
            # I have much to apologize for

    def forward(self, x: Float[Tensor, "*b in_dim"]) -> Float[Tensor, "*b out_dim"]:
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, eps=eps)
        return self.last_layer(x)


exp_max_values = {
    torch.float16: 0,
    torch.float32: 50,
    torch.float64: 50,
    torch.bfloat16: 50,
}


def stable_exp(M: Float[Tensor, "*b n p"]) -> Float[Tensor, "*b n p"]:
    shift = M.max(dim=-2, keepdim=True).values
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(shift, torch.distributed.ReduceOp.MAX)
    M += exp_max_values[M.dtype] - shift
    return M.exp()


def reduced_sum(*args, **kwargs):
    summed = torch.sum(*args, **kwargs)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(summed)
    return summed


@torch.no_grad()
def sinkhorn_knopp(
    M: Float[Tensor, "*b n p"],
    n_iterations: int,
    eps: float | int = 1e-8,
) -> Float[Tensor, "*b n p"]:
    M = stable_exp(M)
    for _ in range(n_iterations):
        M /= reduced_sum(M, dim=-2, keepdim=True) + eps
        M /= torch.sum(M, dim=-1, keepdim=True) + eps
    return M


class OnlineClustering(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool,  # god why is this bias still here
        n_sk_iter: int,
        target_temp: float | int,
        pred_temp: float | int,
        positionwise_sk: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.n_sk_iter = n_sk_iter
        self.target_temp = target_temp
        self.pred_temp = pred_temp
        self.positionwise_sk = positionwise_sk
        self.layer = nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.normal_(self.layer.weight, std=1)
        if bias:
            torch.nn.init.zeros_(self.layer.bias)

    def forward(self, x: Float[Tensor, "*b in_dim"]) -> tuple[Float[Tensor, "*b out_dim"], Float[Tensor, ""]]:
        x_n = nn.functional.normalize(x, dim=-1, p=2, eps=1e-7)
        logits = self.layer(x_n)
        if not self.positionwise_sk:
            sk_in = logits.flatten(0, -2)
        else:
            sk_in = logits
        assignments = sinkhorn_knopp(sk_in.detach() / self.target_temp, n_iterations=self.n_sk_iter)
        if not self.positionwise_sk:
            assignments = assignments.unflatten(0, logits.shape[:-1])

        tgt = assignments.flatten(0, -2).float()
        pred = logits.flatten(0, -2).float()
        loss = -torch.sum(tgt * F.log_softmax(pred / self.pred_temp, dim=-1), dim=-1).mean()
        return assignments.detach(), loss


# API that a vision encoder must implement to fit in the benchmark
class VisionEncoder(nn.Module):
    patch_size: int

    def forward(
        self,
        x: Float[Tensor, "b c h w"],
    ) -> tuple[
        Float[Tensor, "b d"],
        Float[Tensor, "b k d"] | None,
        Float[Tensor, "b ih iw d"],
    ]:
        raise NotImplementedError


# If a file contains a __model_loader__ that returns a VisionEncoder
# then it can be evaluated
def __model_loader__(
    config_path: str | None = None,
    pretrained_weights: str | None = None,
    config_override: Mapping = {},
    device: str = "cuda",
) -> nn.Module:
    """Create a model on the specified device
    Plz no side fx (it's more polite)
    """
    # create model
    cfg = OmegaConf.unsafe_merge(
        OmegaConf.load(Path(__file__).parent / "default_pretrain_config.yaml"),
        OmegaConf.load(config_path) if config_path is not None else {},
        config_override,
    )
    logger.debug(f"Loading model with cfg {OmegaConf.to_yaml(cfg)}")
    with torch.device("meta"):
        model = EncoderDecoder(**cfg.model)
    model.to_empty(device=device)
    # load weights
    if isinstance(pretrained_weights, str) and pretrained_weights != "":
        parsed = urlsplit(pretrained_weights)
        # load ckpt, local or remote
        loader = torch.load if parsed.scheme == "" else torch.hub.load_state_dict_from_url
        sd = loader(pretrained_weights, map_location="cpu", weights_only=True)
        # remove 'backbone.' prefix
        pre = "backbone."
        sd = {k.removeprefix(pre): v for k, v in sd.items() if k.startswith(pre)}
        # load into model
        msg = model.load_state_dict(sd, strict=False)
        logger.info(f"Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}")
    else:
        model.init_weights()
    model.eval()
    return model
