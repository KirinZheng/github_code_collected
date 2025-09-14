# https://arxiv.org/pdf/2203.06145.pdf
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from amzcls.registry import TRANSFORMS


@TRANSFORMS.register_module()
class NDANCaltech101(object):
    def __init__(self, keys, rotate_degrees=15, affine_degrees=0, shear=(-15, 15)):
        self.keys = keys
        self.rotate = transforms.RandomRotation(degrees=rotate_degrees)
        self.shearx = transforms.RandomAffine(degrees=affine_degrees, shear=shear)

    def process(self, data):
        # data[T, C, H, W]
        choices = ['roll', 'rotate', 'shear']
        aug = np.random.choice(choices)
        if aug == 'roll':
            off1 = random.randint(-3, 3)
            off2 = random.randint(-3, 3)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
        if aug == 'rotate':
            data = self.rotate(data)
        if aug == 'shear':
            data = self.shearx(data)
        return data

    def __call__(self, results):
        for k in self.keys:
            if isinstance(results[k], torch.Tensor):
                results[k] = self.process(results[k])
            else:
                raise NotImplemented

        return results


@TRANSFORMS.register_module()
class NDADVSCifar10(object):
    def __init__(self, keys, resize=(48, 48), rotate_degrees=30, affine_degrees=0, shear=(-30, 30)):
        self.keys = keys
        self.resize = transforms.Resize(size=resize, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.rotate = transforms.RandomRotation(degrees=rotate_degrees)
        self.shearx = transforms.RandomAffine(degrees=affine_degrees, shear=shear)

    def process(self, data):
        # data[T, C, H, W]
        choices = ['roll', 'rotate', 'shear']
        aug = np.random.choice(choices)
        if aug == 'roll':
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
        if aug == 'rotate':
            data = self.rotate(data)
        if aug == 'shear':
            data = self.shearx(data)
        return data

    def __call__(self, results):
        for k in self.keys:
            if isinstance(results[k], torch.Tensor):
                results[k] = self.process(results[k])
            else:
                raise NotImplemented

        return results
