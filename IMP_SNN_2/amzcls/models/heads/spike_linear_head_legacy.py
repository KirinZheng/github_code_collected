# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, List

import torch
from mmpretrain.models.heads import LinearClsHead
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from torch import nn


@MODELS.register_module()
class SpikeLinearClsHeadLegacy(LinearClsHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 enable_time_embed=False,
                 time_step=None,
                 init_cfg=None,
                 **kwargs):
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)

        super(SpikeLinearClsHeadLegacy, self).__init__(num_classes, in_channels, init_cfg, **kwargs)
        self.enable_time_embed = enable_time_embed
        if enable_time_embed:
            self.embed = nn.Parameter(torch.ones(time_step))

        self.in_channels = in_channels
        self.num_classes = num_classes
        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        if self.enable_time_embed:
            out = self.time_embed(feats[-1])
        else:
            out = feats[-1]
        return out

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats).mean(0)
        cls_score = self.fc(pre_logits)
        return cls_score

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self.fc(self.pre_logits(feats).mean(0))

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def time_embed(self, x):
        assert x.shape[0] == self.embed.shape[0], \
            f'[INFO] [AMZCLS] Check time step. ' \
            f'T1={x.shape[0]} and T2={self.embed.shape[0]}'
        return torch.einsum('i..., i->i...', x, self.embed.sigmoid())


@MODELS.register_module()
class TETLinearClsHeadLegacy(LinearClsHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 enable_time_embed=False,
                 time_step=None,
                 lamb=1e-3,
                 means=1.0,
                 init_cfg=None,
                 **kwargs):
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)

        super(TETLinearClsHeadLegacy, self).__init__(num_classes, in_channels, init_cfg, **kwargs)
        self.enable_time_embed = enable_time_embed
        if enable_time_embed:
            # weight decay should be considered.
            self.embed = nn.Parameter(torch.zeros(time_step))

        self.in_channels = in_channels
        self.num_classes = num_classes
        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.lamb = lamb
        self.means = means
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        if self.enable_time_embed:
            out = self.time_embed(feats[-1])
        else:
            out = feats[-1]
        return out

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats).mean(0)
        cls_score = self.fc(pre_logits)
        return cls_score

    def time_embed(self, x):
        assert x.shape[0] == self.embed.shape[0], \
            f'[INFO] [AMZCLS] Check time step. ' \
            f'T1={x.shape[0]} and T2={self.embed.shape[0]}'
        return torch.einsum('i..., i->i...', x, self.embed.sigmoid() * 2)

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self.fc(self.pre_logits(feats))
        time, batch, dim = cls_score.shape

        # cross_entropy [after voting]
        loss_mse = self._get_loss(cls_score.mean(0), data_samples, **kwargs)

        # cross_entropy [before voting]
        losses = self._get_loss(cls_score.view(time * batch, dim), data_samples * time, **kwargs)

        # merge loss
        losses['loss'] = losses['loss'] * (1. - self.lamb) + loss_mse['loss'] * self.lamb

        return losses


@MODELS.register_module()
class TALinearClsHeadLegacy(LinearClsHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 window_size: int = 1,
                 lamb=1e-3,
                 enable_time_embed=False,
                 time_step=None,
                 init_cfg=None,
                 **kwargs):
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)

        super(TALinearClsHeadLegacy, self).__init__(num_classes, in_channels, init_cfg, **kwargs)
        self.enable_time_embed = enable_time_embed
        if enable_time_embed:
            # weight decay should be considered.
            self.embed = nn.Parameter(torch.zeros(time_step))

        self.in_channels = in_channels
        self.num_classes = num_classes
        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.lamb = lamb
        self.window_size = window_size
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        if self.enable_time_embed:
            out = self.time_embed(feats[-1])
        else:
            out = feats[-1]
        return out

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats).mean(0)
        cls_score = self.fc(pre_logits)
        return cls_score

    def time_embed(self, x):
        assert x.shape[0] == self.embed.shape[0], \
            f'[INFO] [AMZCLS] Check time step. ' \
            f'T1={x.shape[0]} and T2={self.embed.shape[0]}'
        return torch.einsum('i..., i->i...', x, self.embed.sigmoid() * 2.)

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self.fc(self.pre_logits(feats))

        # cross_entropy [after voting]
        loss_after_voting = self._get_loss(cls_score.mean(0), data_samples, **kwargs)

        # cross_entropy [before voting]
        cls_score = temporal_aggregation(cls_score, self.window_size)
        time, batch, dim = cls_score.shape
        losses = self._get_loss(cls_score.view(time * batch, dim), data_samples * time, **kwargs)

        # merge loss
        losses['loss'] = losses['loss'] * (1. - self.lamb) + loss_after_voting['loss'] * self.lamb

        return losses


def temporal_aggregation(x: torch.Tensor, window_size):
    """ x: [x0, x1, x2, x3], window_size: 2
        y: [x0+x1, x1+x2, x2+x3]
    :param x: Tensor[Time, Batch, ...]
    :param window_size: int
    :return: Tensor[Time-window_size+1, Batch, ...]
    """
    x_temporal_aggregation = torch.stack(
        [x[i:i + x.shape[0] - window_size + 1] for i in range(window_size)]
    ).mean(0)
    return x_temporal_aggregation


def temporal_aggregation_loss(cls_score: torch.Tensor, target: torch.Tensor, window_size, loss_function, **kwargs):
    cls_score = temporal_aggregation(cls_score, window_size)
    time_step, batch = cls_score.shape[:2]
    cls_score = cls_score.reshape((time_step * batch, -1))
    target = torch.stack([target] * time_step).reshape((time_step * batch,))
    return loss_function(cls_score, target, **kwargs)
