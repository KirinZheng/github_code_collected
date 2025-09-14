#!/bin/bash

# CIFAR10DVS VGG
CUDA_VISIBLE_DEVICES=0 python test.py configs/vgg/vgg11_dvscifar10_nda.py
CUDA_VISIBLE_DEVICES=1 python test.py configs/vgg/vgg11_state_dvscifar10_nda.py
CUDA_VISIBLE_DEVICES=2 python test.py configs/vgg/vgg11_dvs128gesture_nda.py

# CIFAR10DVS SpikFormer
CUDA_VISIBLE_DEVICES=0 python test.py configs/spikformer/spikformer_dvscifar10.py
CUDA_VISIBLE_DEVICES=1 python test.py configs/spikformer/spikformer_dvscifar10_nda.py

# N-CALTECH101 VGG
CUDA_VISIBLE_DEVICES=1 python test.py configs/vgg/vgg11_ncaltech101.py
