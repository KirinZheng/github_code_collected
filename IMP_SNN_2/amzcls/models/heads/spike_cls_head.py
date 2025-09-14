from typing import Tuple, List, Union, Optional

import torch
from mmpretrain.evaluation import Accuracy
from mmpretrain.models.heads import ClsHead
from mmpretrain.registry import MODELS
from torch import nn
from mmpretrain.structures import DataSample
from amzcls.models.heads.spike_linear_head import process_target


@MODELS.register_module()
class SpikeClsHead(ClsHead):

    def __init__(self,
                 time_step_embed: int = None,
                 out_time_step: Union[List[int], Tuple[int]] = None,
                 loss=None,
                 topk: Union[int, Tuple[int]] = (1,),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        if loss is None:
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
        super(SpikeClsHead, self).__init__(loss, topk, cal_acc, init_cfg)
        if time_step_embed is not None:
            self.embed = nn.Parameter(torch.ones(time_step_embed))
        self.out_time_step = out_time_step
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

        cls_score = pre_logits.mean(0)
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
        cls_score = self.pre_logits(feats)  # [T, B, ...]
        if self.out_time_step is not None:
            cls_score = cls_score[self.out_time_step]  # [T_out, B, ...]
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
class TETClsHead(ClsHead):

    def __init__(self,
                 out_time_step: Union[List[int], Tuple[int]] = None,
                 loss=None,
                 topk: Union[int, Tuple[int]] = (1,),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        if loss is None:
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
        super(TETClsHead, self).__init__(loss, topk, cal_acc, init_cfg)
        self.out_time_step = out_time_step
        if not isinstance(self.loss_module, nn.Module):
            self.loss_module = MODELS.build(self.loss_module)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats)
        if self.out_time_step is not None:
            pre_logits = pre_logits[self.out_time_step]
        return pre_logits.mean(0)

    def loss(self, feats, data_samples, **kwargs) -> dict:
        cls_score = self.pre_logits(feats)
        if self.out_time_step is not None:
            cls_score = cls_score[self.out_time_step]
        target = process_target(data_samples)

        loss_dict = self.get_loss(cls_score, target, **kwargs)
        loss_dict = self.update_accuracy(cls_score.mean(0), target, loss_dict)
        return loss_dict

    def get_loss(self, cls_score: torch.Tensor, target: torch.Tensor, **kwargs):
        time, batch, dim = cls_score.shape
        loss = self.loss_module(
            cls_score.view(time * batch, dim), torch.cat([target] * time, dim=0),
            avg_factor=time * batch, **kwargs)
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
class LogClsHead(ClsHead):

    def __init__(self,
                 time_step: int,
                 trainable=False,
                 loss=None,
                 topk: Union[int, Tuple[int]] = (1,),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        if loss is None:
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
        super(LogClsHead, self).__init__(loss, topk, cal_acc, init_cfg)
        self.time_embed = nn.Parameter(logarithmic(time_step), requires_grad=trainable)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        assert feats[-1].shape[0] == self.time_embed.shape[0], \
            f'[INFO] [AMZCLS] Check time step. ' \
            f'T1{feats[-1].shape[0]} and T2{self.time_embed.shape[0]}'
        out = torch.einsum('i..., i->...', feats[-1], self.time_embed)
        return out

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats)
        return pre_logits


@MODELS.register_module()
class MovingAverageHead(ClsHead):

    def __init__(self,
                 time_step: int,
                 trainable=False,
                 beta=0.3,
                 loss=None,
                 topk: Union[int, Tuple[int]] = (1,),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        if loss is None:
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
        super(MovingAverageHead, self).__init__(loss, topk, cal_acc, init_cfg)
        self.time_embed = nn.Parameter(maea(time_step, beta), requires_grad=trainable)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        assert feats[-1].shape[0] == self.time_embed.shape[0], \
            f'[INFO] [AMZCLS] Check time step. ' \
            f'T1{feats[-1].shape[0]} and T2{self.time_embed.shape[0]}'
        out = torch.einsum('i..., i->...', feats[-1], self.time_embed)
        return out

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats)
        return pre_logits


@MODELS.register_module()
class FixClsHead(ClsHead):

    def __init__(self,
                 time_embed: list,
                 trainable=False,
                 loss=None,
                 topk: Union[int, Tuple[int]] = (1,),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        if loss is None:
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
        super(FixClsHead, self).__init__(loss, topk, cal_acc, init_cfg)
        self.time_embed = nn.Parameter(torch.tensor(time_embed), requires_grad=trainable)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        assert feats[-1].shape[0] == self.time_embed.shape[0], \
            f'[INFO] [AMZCLS] Check time step. ' \
            f'T1{feats[-1].shape[0]} and T2{self.time_embed.shape[0]}'
        return torch.einsum('i..., i->...', feats[-1], self.time_embed)

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats)
        return pre_logits


def check_time_step(time_embed, length):
    # todo: impl VIT like interpolate.
    if length == time_embed.shape[0]:
        return time_embed
    else:
        return torch.nn.functional.interpolate(time_embed, size=length)


def logarithmic(length):
    linespace = torch.range(start=1, end=length, step=1)
    out = torch.log(linespace)
    return out / out.sum()


def maea(length, beta=0.3):
    """
    Moving Average Expansion Approximate
    approximate: [b*(1-b)^(length-1-i) for i in range(length)]
    """
    exp1 = torch.range(start=length - 1, end=0, step=-1)
    exp2 = torch.ones(length)
    return torch.pow(beta, exp2) * torch.pow((1 - beta), exp1)


def mae(length, beta=0.3):
    """
    Moving Average Expansion
    length=2 ->                                (1-b), b
    length=3 ->                      (1-b)^2, b(1-b), b
    length=4 ->            (1-b)^3, b(1-b)^2, b(1-b), b
    """
    exp1 = torch.range(start=length - 1, end=0, step=-1)
    exp2 = torch.concat([torch.zeros(1), torch.ones(length - 1)])
    return torch.pow(beta, exp2) * torch.pow((1 - beta), exp1)
