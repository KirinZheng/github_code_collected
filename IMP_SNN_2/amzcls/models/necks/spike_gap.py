# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule
from spikingjelly.activation_based import layer, functional
from spikingjelly.activation_based.base import StepModule

from ..builder import MODELS


@MODELS.register_module()
class SpikeGlobalAveragePooling(BaseModule, StepModule):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2, step_mode='m'):
        super(SpikeGlobalAveragePooling, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
                                 f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = layer.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = layer.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = layer.AdaptiveAvgPool3d((1, 1, 1))

        self.step_mode = step_mode
        if step_mode == 'm':
            functional.set_step_mode(self, 'm')

    def init_weights(self):
        pass

    def forward(self, inputs):
        flatten_dim = 2 if self.step_mode == 'm' else 1
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple([torch.flatten(out, flatten_dim) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = torch.flatten(outs, flatten_dim)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
