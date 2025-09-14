_base_ = [
    '../_base_/models/resnet18.py',
    # '../_base_/datasets/imagenet100_bs256_rsb_a12.py',
    '../_base_/datasets/imagenet100_bs64.py',  # 81.1800
    '../_base_/schedules/imagenet_sgd_coslr_100e.py',
    '../_base_/default_runtime.py'
]
# 81.1800
model = dict(
    # backbone=dict(init_cfg=dict(
    #     type='Pretrained', checkpoint='', prefix='backbone.'
    # )),
    # backbone=dict(init_cfg=dict(
    #     type='Pretrained'
    # )),
    head=dict(num_classes=100))

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
)
