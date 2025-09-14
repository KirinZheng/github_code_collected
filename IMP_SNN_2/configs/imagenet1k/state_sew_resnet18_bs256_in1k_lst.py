_base_ = [
    '../_base_/models/sew_resnet18.py',
    '../_base_/datasets/imagenet_bs64.py',
    # '../_base_/datasets/imagenet_bs256_rsb_a3.py',
    '../_base_/default_runtime.py'
]

data_preprocessor = dict(
    type='StaticPreprocessor',
    time_step=4,
)

model = dict(
    backbone=dict(
        neuron_cfg=dict(type='LazyStateIFNode', v_reset=None),
        pretrained=True
    ),
    head=dict(num_classes=1000, out_time_step=[3, ])
)

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    # optimizer=dict(type='Lamb', lr=0.005, weight_decay=0.02)
    optimizer=dict(type='SGD', lr=0.1, weight_decay=0.0)
)

# learning policy
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=0,
        end=30)
]

train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
# train, val, test setting
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
auto_scale_lr = dict(base_batch_size=256)
