# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
from model import __model_loader__


def capi_vitl14_lvd():
    return __model_loader__(
        config_path=None, pretrained_weights="https://dl.fbaipublicfiles.com/capi/capi_vitl14_lvd.pth", device="cpu"
    )


def capi_vitl14_p205():
    return __model_loader__(
        config_path=None, pretrained_weights="https://dl.fbaipublicfiles.com/capi/capi_vitl14_p205.pth", device="cpu"
    )


def capi_vitl14_in22k():
    return __model_loader__(
        config_path=None, pretrained_weights="https://dl.fbaipublicfiles.com/capi/capi_vitl14_i22k.pth", device="cpu"
    )


def capi_vitl14_in1k():
    return __model_loader__(
        config_path=None, pretrained_weights="https://dl.fbaipublicfiles.com/capi/capi_vitl14_in1k.pth", device="cpu"
    )
