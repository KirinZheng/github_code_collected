from numbers import Number
from typing import Optional, Sequence, Union

import torch
from mmpretrain.models import SelfSupDataPreprocessor
from mmpretrain.models.utils import ClsDataPreprocessor
from mmpretrain.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class StaticPreprocessor(ClsDataPreprocessor):
    def __init__(self,
                 time_step: int = None,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Number = 0,
                 to_rgb: bool = False,
                 to_onehot: bool = False,
                 num_classes: Optional[int] = None,
                 batch_augments: Optional[dict] = None):
        super(StaticPreprocessor, self).__init__(
            mean=mean, std=std, pad_size_divisor=pad_size_divisor, pad_value=pad_value, to_rgb=to_rgb,
            to_onehot=to_onehot, num_classes=num_classes, batch_augments=batch_augments)
        self.time_step = time_step

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super(StaticPreprocessor, self).forward(data, training)
        data['inputs'] = process_static_data(data['inputs'], self.time_step)
        return data


@MODELS.register_module()
class DVSPreprocessor(ClsDataPreprocessor):
    def __init__(self,
                 time_step: int = None,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Number = 0,
                 to_rgb: bool = False,
                 to_onehot: bool = False,
                 num_classes: Optional[int] = None,
                 batch_augments: Optional[dict] = None):
        super(DVSPreprocessor, self).__init__(
            mean=mean, std=std, pad_size_divisor=pad_size_divisor, pad_value=pad_value, to_rgb=to_rgb,
            to_onehot=to_onehot, num_classes=num_classes, batch_augments=batch_augments)
        self.time_step = time_step

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super(DVSPreprocessor, self).forward(data, training)
        data['inputs'] = process_dvs_data(data['inputs'])
        return data


@MODELS.register_module()
class StaticSelfSupDataPreprocessor(SelfSupDataPreprocessor):
    """Image pre-processor for operations, like normalization and bgr to rgb.

    Compared with the :class:`mmengine.ImgDataPreprocessor`, this module
    supports ``inputs`` as torch.Tensor or a list of torch.Tensor.
    """

    def __init__(self,
                 time_step: int = None,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 to_rgb: bool = False,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            to_rgb=to_rgb,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)

        self.time_step = time_step
        self._channel_conversion = to_rgb or bgr_to_rgb or rgb_to_bgr

    def forward(self, data: dict, training: bool = False):
        data = super(StaticSelfSupDataPreprocessor, self).forward(data, training)
        data['inputs'] = [process_static_data(sample, self.time_step) for sample in data['inputs']]
        return data


@MODELS.register_module()
class DVSSelfSupDataPreprocessor(SelfSupDataPreprocessor):
    """Image pre-processor for operations, like normalization and bgr to rgb.

    Compared with the :class:`mmengine.ImgDataPreprocessor`, this module
    supports ``inputs`` as torch.Tensor or a list of torch.Tensor.
    """

    def __init__(self,
                 time_step: int = None,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 to_rgb: bool = False,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            to_rgb=to_rgb,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)

        self.time_step = time_step
        self._channel_conversion = to_rgb or bgr_to_rgb or rgb_to_bgr

    def forward(self, data: dict, training: bool = False):
        data = super(DVSSelfSupDataPreprocessor, self).forward(data, training)
        data['inputs'] = [process_dvs_data(sample) for sample in data['inputs']]
        return data


def process_static_data(data: Tensor, repeats):
    # DATA[B, C, H, W] -> DATA[T, B, C, H, W]
    return torch.repeat_interleave(data.unsqueeze(0), repeats, dim=0)


def process_dvs_data(data: Tensor):
    # DATA[B, T, C, H, W] -> DATA[T, B, C, H, W]
    return torch.transpose(data, 0, 1)
