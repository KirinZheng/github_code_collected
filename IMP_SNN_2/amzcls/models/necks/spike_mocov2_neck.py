# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmpretrain.registry import MODELS
from spikingjelly.activation_based import layer, functional


@MODELS.register_module()
class SpikeMoCoV2Neck(BaseModule):
    """The non-linear neck of MoCo v2: fc-relu-fc.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 hid_channels: int,
                 out_channels: int,
                 with_avg_pool: bool = True,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super(SpikeMoCoV2Neck, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            layer.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            layer.Linear(hid_channels, out_channels))
        functional.set_step_mode(self, 'm')

    def forward(self, x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Forward function.

        Args:
            x (Tuple[torch.Tensor]): The feature map of backbone.

        Returns:
            Tuple[torch.Tensor]: The output features.
        """
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = self.mlp(x.view(x.size(0), x.size(1), -1))
        return x,
