# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SpikformerCifar',
        in_channels=3,
        img_size_h=32,
        img_size_w=32,
        patch_size=4,
        embed_dims=384,
        num_heads=12,
        mlp_ratios=4,
        norm_layer=None,
        depths=4,
    ),
    head=dict(
        type='SpikeLinearClsHead',
        num_classes=10,
        in_channels=384,
        time_step_embed=None,
        out_time_step=None,
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
