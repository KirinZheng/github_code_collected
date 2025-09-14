# dataset settings
dataset_type = 'DVSPack'
time_step = 16

data_preprocessor = dict(
    type='DVSSelfSupDataPreprocessor',
    time_step=time_step,
    to_rgb=False
)

# The difference between mocov2 and mocov1 is the transforms in the pipeline
# view_pipeline = [
#     dict(
#         type='RandomResizedCrop',
#         scale=32,
#         crop_ratio_range=(0.8, 1.),
#         backend='pillow'),
#     dict(
#         type='RandomApply',
#         transforms=[
#             dict(
#                 type='ColorJitter',
#                 brightness=0.4,
#                 contrast=0.4,
#                 saturation=0.4,
#                 hue=0.1)
#         ],
#         prob=0.8),
#     dict(
#         type='RandomGrayscale',
#         prob=0.2,
#         keep_channels=True,
#         channel_weights=(0.114, 0.587, 0.2989)),
#     dict(
#         type='GaussianBlur',
#         magnitude_range=(0.1, 2.0),
#         magnitude_std='inf',
#         prob=0.5),
#     dict(type='RandomFlip', prob=0.5),
# ]

augmentation_space = {
    "Identity": ['torch.tensor(0.0)', False],
    "ShearX": ['torch.linspace(-0.3, 0.3, 31)', True],
    "ShearY": ['torch.linspace(-0.3, 0.3, 31)', True],
    "TranslateX": ['torch.linspace(-5.0, 5.0, 31)', True],
    "TranslateY": ['torch.linspace(-5.0, 5.0, 31)', True],
    "Rotate": ['torch.linspace(-30.0, 30.0, 31)', True],
    "Cutout": ['torch.linspace(1.0, 30.0, 31)', True],
}

view_pipeline = [
    dict(type='ToFloatTensor', keys=['img']),
    dict(type='RandomHorizontalFlipDVS', prob=0.5, keys=['img']),
    # SpikFormerDVS
    dict(type='SpikFormerDVS', keys=['img'], augmentation_space=augmentation_space),
]

train_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackInputs')
]

dvscifar10_cfg = dict(
    type='DVSCifar10',
    time_step=time_step,
    data_type='frame',
    split_by='number',
    test_mode=False,
    data_prefix='/home/stephen/Desktop/workspace/2023/amz-cls-0.0.2/data/dvs-cifar10',
    use_ckpt=False,
    pipeline=[])

gesture_cfg = dict(
    type='DVS128Gesture',
    time_step=time_step,
    data_type='frame',
    split_by='number',
    test_mode=False,
    data_prefix='/home/stephen/Desktop/workspace/2023/amz-cls-0.0.2/data/dvs-gesture',
    pipeline=[])

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        dataset_cfgs=[
            dvscifar10_cfg,
            gesture_cfg
        ],
        pipeline=train_pipeline,
    )
)
