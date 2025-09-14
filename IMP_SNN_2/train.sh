#!/bin/bash

# CIFAR10DVS VGG
CUDA_VISIBLE_DEVICES=0 python main.py configs/vgg/vgg11_dvscifar10_nda.py &
CUDA_VISIBLE_DEVICES=1 python main.py configs/vgg/vgg11_state_dvscifar10_nda.py &
CUDA_VISIBLE_DEVICES=2 python main.py configs/vgg/vgg11_dvs128gesture_nda.py

# CIFAR10DVS SpikFormer
CUDA_VISIBLE_DEVICES=0 python main.py configs/spikformer/spikformer_dvscifar10.py &
CUDA_VISIBLE_DEVICES=1 python main.py configs/spikformer/spikformer_dvscifar10_nda.py

# N-CALTECH101 VGG
CUDA_VISIBLE_DEVICES=1 python main.py configs/vgg/vgg11_ncaltech101.py



## Mine
## CIFAR10 SEW-ResNet (cd /zhengzeqi/Githubs_code/IMP_SNN_2)

PYTHONPATH=/zhengzeqi/Githubs_code/IMP_SNN_2/amzcls CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 main.py configs/cifar10/sew_resnet18_rsb-a2-300e_cifar10_sdt_t4.py --launcher pytorch
