from typing import Tuple, List, Union

import torch
from mmpretrain.evaluation import Accuracy
from mmpretrain.models.heads import LinearClsHead
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from torch import nn


@MODELS.register_module()
class SpikeLinearClsHead(LinearClsHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 time_step_embed: int = None,
                 out_time_step: Union[List[int], Tuple[int]] = None,
                 init_cfg=None,
                 **kwargs):
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)

        super(SpikeLinearClsHead, self).__init__(num_classes, in_channels, init_cfg, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        if time_step_embed is not None:
            self.embed = nn.Parameter(torch.ones(time_step_embed))
        self.out_time_step = out_time_step

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

        if not isinstance(self.loss_module, nn.Module):
            self.loss_module = MODELS.build(self.loss_module)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        out = feats[-1]
        if hasattr(self, 'embed'):
            out = self.time_embed(out)

        return out

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats)
        if self.out_time_step is not None:
            pre_logits = pre_logits[self.out_time_step]

        cls_score = self.fc(pre_logits.mean(0))
        return cls_score

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample], **kwargs) -> dict:
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
        pre_logits = self.pre_logits(feats)  # [T, B, ...]
        if self.out_time_step is not None:
            pre_logits = pre_logits[self.out_time_step]  # [T_out, B, ...]

        cls_score = self.fc(pre_logits)
        target = process_target(data_samples)

        loss_dict = self.get_loss(cls_score, target, **kwargs)  # x[T_out, B, ...], gt[B, ...]
        loss_dict = self.update_accuracy(cls_score.mean(0), target, loss_dict)
        return loss_dict

    def time_embed(self, x):
        assert x.shape[0] == self.embed.shape[0], \
            f'[INFO] [AMZCLS] Check time step. ' \
            f'T1={x.shape[0]} and T2={self.embed.shape[0]}'
        return torch.einsum('i..., i->i...', x, self.embed)

    def get_loss(self, cls_score: torch.Tensor, target: torch.Tensor, **kwargs):
        loss = self.loss_module(cls_score.mean(0), target, avg_factor=cls_score.size(1), **kwargs)
        return dict(loss=loss)

    def update_accuracy(self, cls_score: torch.Tensor, target: torch.Tensor, loss_dict: dict):
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                                     'like mixup during training, `cal_acc` is pointless.'
            loss_dict.update({f'accuracy_top-{topk}': accuracy for topk, accuracy in zip(
                self.topk,
                Accuracy.calculate(cls_score, target, self.topk)
            )})
        return loss_dict


@MODELS.register_module()
class TETLinearClsHead(SpikeLinearClsHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 time_step_embed: int = None,
                 out_time_step: Union[List[int], Tuple[int]] = None,
                 init_cfg=None,
                 **kwargs):
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)

        super(TETLinearClsHead, self).__init__(
            num_classes, in_channels, time_step_embed, out_time_step, init_cfg, **kwargs)

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample], **kwargs) -> dict:
        pre_logits = self.pre_logits(feats)  # [T, B, ...]
        if self.out_time_step is not None:
            pre_logits = pre_logits[self.out_time_step]  # [T_out, B, ...]

        cls_score = self.fc(pre_logits)
        target = process_target(data_samples)
        # TET cross entropy [before voting]
        loss_dict = self.get_loss(cls_score, target, **kwargs)  # x[T_out, B, ...], gt[B, ...]
        loss_dict = self.update_accuracy(cls_score.mean(0), target, loss_dict)
        return loss_dict

    def get_loss(self, cls_score: torch.Tensor, target: torch.Tensor, **kwargs):
        time, batch, dim = cls_score.shape
        # todo: avg_factor=cls_score.size(0) 但是size(0)表示T维度 #
        # todo: 分别检查 avg_factor=T, avg_factor=B, avg_factor=T*B #
        # =============================== V0 =============================== #
        # loss = self.loss_module(
        #     cls_score.view(time * batch, dim), torch.cat([target] * time, dim=0),
        #     avg_factor=cls_score.size(0), **kwargs)
        # =============================== V1 =============================== #
        loss = self.loss_module(
            cls_score.view(time * batch, dim), torch.cat([target] * time, dim=0),
            avg_factor=time * batch, **kwargs)  # x[T_out * B, ...], gt[T_out * B, ...]
        return dict(loss=loss)


@MODELS.register_module()
class TALinearClsHead(SpikeLinearClsHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 time_step_embed: int = None,
                 out_time_step: Union[List[int], Tuple[int]] = None,
                 window_sizes: Tuple[int] = (1,),
                 time_weights: Tuple[int] = (1.0,),
                 init_cfg=None,
                 **kwargs):
        if out_time_step is not None:
            assert all(ws <= (len(out_time_step) + 1) for ws in window_sizes)
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)
        self.window_sizes = window_sizes
        self.time_weights = time_weights
        super(TALinearClsHead, self).__init__(
            num_classes, in_channels, time_step_embed, out_time_step, init_cfg, **kwargs)

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample], **kwargs) -> dict:
        pre_logits = self.pre_logits(feats)  # [T, B, ...]
        if self.out_time_step is not None:
            pre_logits = pre_logits[self.out_time_step]  # [T_out, B, ...]

        cls_score = self.fc(pre_logits)
        target = process_target(data_samples)
        # TA cross entropy [before voting]
        loss_dict = self.get_loss(cls_score, target, **kwargs)  # x[T_out, B, ...], gt[B, ...]
        loss_dict = self.update_accuracy(cls_score.mean(0), target, loss_dict)
        return loss_dict

    def get_loss(self, cls_score: torch.Tensor, target: torch.Tensor, **kwargs):
        loss = 0.
        for window_size, time_weight in zip(self.window_sizes, self.time_weights):
            ta_score = temporal_aggregation(cls_score, window_size)
            time, batch, dim = ta_score.shape
            # =============================== V0 =============================== #
            # ta_loss = self.loss_module(
            #     ta_score.view(time * batch, dim), torch.cat([target] * time, dim=0),
            #     avg_factor=ta_score.size(0), **kwargs)
            # =============================== V1 =============================== #
            ta_loss = self.loss_module(
                ta_score.view(time * batch, dim), torch.cat([target] * time, dim=0),
                avg_factor=time * batch, **kwargs)  # x[T_ag * B, ...], gt[T_ag * B, ...]
            loss = loss + ta_loss * time_weight  # merge loss
        return dict(loss=loss)


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


def process_target(data_samples):
    if 'gt_score' in data_samples[0]:
        target = torch.stack([i.gt_score for i in data_samples])
    else:
        target = torch.cat([i.gt_label for i in data_samples])
    return target


def temporal_aggregation_loss(cls_score: torch.Tensor, target: torch.Tensor, window_size, loss_function, **kwargs):
    cls_score = temporal_aggregation(cls_score, window_size)
    time_step, batch = cls_score.shape[:2]
    cls_score = cls_score.reshape((time_step * batch, -1))
    target = torch.stack([target] * time_step).reshape((time_step * batch,))
    return loss_function(cls_score, target, **kwargs)
