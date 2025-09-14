# 2023-10-31

from typing import Optional

from mmpretrain.models.classifiers import HuggingFaceClassifier as MMPHuggingFaceClassifier
from mmpretrain.models.classifiers import ImageClassifier as MMPImageClassifier
from mmpretrain.models.classifiers import TimmClassifier as MMPTimmClassifier
from torch import nn

from ..builder import MODELS


@MODELS.register_module()
class ImageClassifier(MMPImageClassifier):
    def __init__(self, backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 *args, **kwargs):
        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)
        super(ImageClassifier, self).__init__(backbone, neck, head, *args, **kwargs)


@MODELS.register_module()
class TimmClassifier(MMPTimmClassifier):
    def __init__(self,
                 *args,
                 loss=None,
                 train_cfg: Optional[dict] = None,
                 with_cp: bool = False,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        if loss is None:
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        super(TimmClassifier, self).__init__(
            *args, loss, train_cfg, with_cp, data_preprocessor, init_cfg, **kwargs)
        self.loss_module = loss


@MODELS.register_module()
class HuggingFaceClassifier(MMPHuggingFaceClassifier):
    def __init__(self,
                 model_name,
                 pretrained=False,
                 *model_args,
                 loss=None,
                 train_cfg: Optional[dict] = None,
                 with_cp: bool = False,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        if loss is None:
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        super(HuggingFaceClassifier, self).__init__(
            model_name, pretrained, *model_args, loss, train_cfg,
            with_cp, data_preprocessor, init_cfg, **kwargs)
