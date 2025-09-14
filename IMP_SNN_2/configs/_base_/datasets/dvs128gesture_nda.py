# dataset settings

dataset_type = 'DVS128Gesture'
time_step = 16

data_preprocessor = dict(
    type='DVSPreprocessor',
    time_step=time_step,
    num_classes=11,
    to_rgb=False
)

train_pipeline = [
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='ToFloatTensor', keys=['img']),
    dict(type='RandomHorizontalFlipDVS', prob=0.5, keys=['img']),
    # SpikFormerDVS
    dict(type='NDADVSCifar10', keys=['img']),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='ToFloatTensor', keys=['img']),
    # dict(type='RandomHorizontalFlipDVS', prob=1.0, keys=['img']),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    # batch_size=32,
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=False,
        data_prefix='./data/dvs-gesture',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=True,
        data_prefix='./data/dvs-gesture',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1,))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
