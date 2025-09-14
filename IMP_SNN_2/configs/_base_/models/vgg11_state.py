# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='StateVGG11',
        layers=[1, 1, 2, 4],
        width=[64, 128, 256, 512],
        in_channels=3,
        neuron_cfg=dict(
            type='StateLIFNode',
            surrogate_function=dict(
                type='Sigmoid'
            )
        ),
    ),
    neck=dict(
        type='SpikeGlobalAveragePooling',
    ),
    head=dict(
        type='SpikeLinearClsHead',
        num_classes=10,
        in_channels=512,
        time_step_embed=None,
        out_time_step=None,
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
