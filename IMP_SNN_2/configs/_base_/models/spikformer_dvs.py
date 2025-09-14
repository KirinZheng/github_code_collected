# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SpikformerDVS',
        in_channels=2,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        embed_dims=256,
        num_heads=16,
        mlp_ratios=4,
        norm_layer=None,
        depths=2,
    ),
    head=dict(
        type='SpikeLinearClsHead',
        num_classes=10,
        in_channels=256,
        time_step_embed=None,
        out_time_step=None,
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
