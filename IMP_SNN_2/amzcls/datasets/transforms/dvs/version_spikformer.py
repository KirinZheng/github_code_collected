import math
from typing import List, Tuple, Optional, Dict

import torch
from torch import Tensor
from torchvision.transforms import functional as F, InterpolationMode
from torchvision.transforms.transforms import RandomErasing

from amzcls.registry import TRANSFORMS


def _apply_op(img: Tensor, op_name: str, magnitude: float,
              interpolation: InterpolationMode, fill: Optional[List[float]]):
    if op_name == "ShearX":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                       interpolation=interpolation, fill=fill)
    elif op_name == "ShearY":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                       interpolation=interpolation, fill=fill)
    elif op_name == "TranslateX":
        img = F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "TranslateY":
        img = F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        # img = F.adjust_brightness(img, 1.0 + magnitude)
        img[:, 0:1] = F.adjust_brightness(img[:, 0:1], 1.0 + magnitude)
        img[:, 1:2] = F.adjust_brightness(img[:, 1:2], 1.0 + magnitude)
    elif op_name == "Color":
        # img = F.adjust_saturation(img, 1.0 + magnitude)
        img[:, 0:1] = F.adjust_saturation(img[:, 0:1], 1.0 + magnitude)
        img[:, 1:2] = F.adjust_saturation(img[:, 1:2], 1.0 + magnitude)
    elif op_name == "Contrast":
        # img = F.adjust_contrast(img, 1.0 + magnitude)
        img[:, 0:1] = F.adjust_contrast(img[:, 0:1], 1.0 + magnitude)
        img[:, 1:2] = F.adjust_contrast(img[:, 1:2], 1.0 + magnitude)
    elif op_name == "Sharpness":
        # img = F.adjust_sharpness(img, 1.0 + magnitude)
        img[:, 0:1] = F.adjust_sharpness(img[:, 0:1], 1.0 + magnitude)
        img[:, 1:2] = F.adjust_sharpness(img[:, 1:2], 1.0 + magnitude)
    elif op_name == "Posterize":
        # img = F.posterize(img, int(magnitude))
        img[:, 0:1] = F.posterize(img[:, 0:1], int(magnitude))
        img[:, 1:2] = F.posterize(img[:, 1:2], int(magnitude))
    elif op_name == "Solarize":
        # img = F.solarize(img, magnitude)
        img[:, 0:1] = F.solarize(img[:, 0:1], magnitude)
        img[:, 1:2] = F.solarize(img[:, 1:2], magnitude)
    elif op_name == "AutoContrast":
        # img = F.autocontrast(img)
        img[:, 0:1] = F.autocontrast(img[:, 0:1])
        img[:, 1:2] = F.autocontrast(img[:, 1:2])
    elif op_name == "Equalize":
        # img = F.equalize(img)
        img[:, 0:1] = F.equalize(img[:, 0:1])
        img[:, 1:2] = F.equalize(img[:, 1:2])
    elif op_name == "Invert":
        # img = F.invert(img)
        img[:, 0:1] = F.invert(img[:, 0:1])
        img[:, 1:2] = F.invert(img[:, 1:2])


    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img


class SpikFormerAugmentWide(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        """

    def __init__(self, num_magnitude_bins: int = 31,
                 interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: Optional[List[float]] = None, augmentation_space: dict = None) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.cutout = RandomErasing(p=1, scale=(0.001, 0.11), ratio=(1, 1))  # todo cutout N holes
        # https://github.com/ChaotengDuan/TEBN/blob/8f82f6a307093faddeb87127aa432a66d65352ea/dataloader.py#L8
        self.augmentation_space = augmentation_space

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        if self.augmentation_space is None:
            return {
                # op_name: (magnitudes, signed)
                "Identity": (torch.tensor(0.0), False),
                "ShearX": (torch.linspace(-0.3, 0.3, num_bins), True),
                "TranslateX": (torch.linspace(-5.0, 5.0, num_bins), True),
                "TranslateY": (torch.linspace(-5.0, 5.0, num_bins), True),
                "Rotate": (torch.linspace(-30.0, 30.0, num_bins), True),
                "Cutout": (torch.linspace(1.0, 30.0, num_bins), True)
            }
        else:
            return self.augmentation_space

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item()) \
            if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        if op_name == "Cutout":
            return self.cutout(img)
        else:
            return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)


def prepare(augmentation):
    try:
        import torch
    except NotImplemented:
        raise NotImplemented
    for key in augmentation:
        augmentation[key][0] = eval(augmentation[key][0])


@TRANSFORMS.register_module()
class SpikFormerDVS(object):
    """
    augmentation_space = {
        # op_name: (magnitudes, signed)
        "Identity": (torch.tensor(0.0), False),
        "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (torch.linspace(0.0, 5.0, num_bins), True),
        "TranslateY": (torch.linspace(0.0, 5.0, num_bins), True),
        "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
        "Cutout": (torch.linspace(0.0, 30.0, num_bins), True),
        "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
        "Color": (torch.linspace(0.0, 0.9, num_bins), True),
        "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
        "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
        "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
        "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
        "AutoContrast": (torch.tensor(0.0), False),
        "Equalize": (torch.tensor(0.0), False),
        "Invert": (torch.tensor(0.0), False),
    }
    """

    def __init__(self, keys, num_magnitude_bins: int = 31, augmentation_space: dict = None):
        self.keys = keys
        self.snn_augment = SpikFormerAugmentWide(
            num_magnitude_bins,
            augmentation_space=prepare(augmentation_space))

    def __call__(self, results):
        for k in self.keys:
            if isinstance(results[k], torch.Tensor):
                results[k] = self.snn_augment(results[k])
            else:
                raise NotImplemented

        return results
