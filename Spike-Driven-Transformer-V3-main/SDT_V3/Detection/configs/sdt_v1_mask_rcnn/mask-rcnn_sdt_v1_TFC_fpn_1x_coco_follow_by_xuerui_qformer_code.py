_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = "/zhengzeqi/top_down/spikedriven/github_version/official_weights/imagenet/8_384.pth.tar"  # SDT_V1 less 16.81M

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [

    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(853, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    dataset=dict(pipeline=train_pipeline))

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SpikeDrivenTransformer_TFC',
        T=2, 
        drop_rate=0.0,
        drop_path_rate=0.2,
        # drop_block_rate=None, #这个模型有吗？
        num_heads=8,
        pooling_stat="1111",
        # img_size_h=args.img_size,
        # img_size_w=args.img_size,
        patch_size=16,
        embed_dims=384,
        mlp_ratios=4,
        in_channels=3,
        qkv_bias=False,
        depths=8,
        sr_ratios=1,
        spike_mode="lif",
        dvs_mode=False,
        TET=False,
        recurrent_coding=True,
        # recurrent_coding=False, # changed on 2025-04-13
        # recurrent_lif='lif',  # changed on 2025-04-13
        # pe_type=None,              # changed on 2025-04-17
        # use_imp_lif=False,  # changed on 2025-05-24
        # maxpooling_lif_change_order=False,
        # gac_coding=False,  # changed on 2025-06-27
        # init_imp_a=0.4, init_imp_b=0.6,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        type='SpikeFPN',        # /zhengzeqi/Githubs_code/Spike-Driven-Transformer-V3-main/SDT_V3/Detection/mmdet/models/necks/spike_fpn.py
        in_channels=[384, 384, 384, 384],  
        out_channels=384,     
        num_outs=5),
    rpn_head=dict(
        type='SpikeRPNHead',    # /zhengzeqi/Githubs_code/Spike-Driven-Transformer-V3-main/SDT_V3/Detection/mmdet/models/dense_heads/spike_rpn_head.py
        in_channels=384,   
        feat_channels=384,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
        #                loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        # loss_bbox=dict(type='GIoULoss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='SpikeStandardRoIHead',        # /zhengzeqi/Githubs_code/Spike-Driven-Transformer-V3-main/SDT_V3/Detection/mmdet/models/roi_heads/spike_standard_roi_head.py
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=384, 
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='SharedSpike2FCBBoxHead',  # /zhengzeqi/Githubs_code/Spike-Driven-Transformer-V3-main/SDT_V3/Detection/mmdet/models/roi_heads/bbox_heads/spike_convfc_bbox_head.py
            in_channels=384,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            # loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
            #                loss_weight=2.0),
            # loss_bbox=dict(type='GIoULoss', loss_weight=1.0), #有bug，loss变负数
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=384,  
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='SpikeFCNMaskHead',  # /zhengzeqi/Githubs_code/Spike-Driven-Transformer-V3-main/SDT_V3/Detection/mmdet/models/roi_heads/mask_heads/spike_fcn_mask_head.py
            num_convs=4,
            in_channels=384,       
            conv_out_channels=384, 
            num_classes=80,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),)

max_epochs = 5  #训练最大轮次是由新设置的train_config决定的，但学习率策略是resume的模型(param_scheduler)决定的
max_iter = max_epochs * 23454
train_cfg = dict(max_epochs=max_epochs)

# learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
#         end=1000),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[27, 33],
#         gamma=0.1)
# ]

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
#         end=4000),
#     dict(
#         type='LinearLR', start_factor=1, end_factor=0.01, by_epoch=False, begin=1,
#         end=max_iter),
#     # dict(
#     #     type='MultiStepLR',
#     #     begin=0,
#     #     end=max_epochs,
#     #     by_epoch=True,
#     #     milestones=[27, 33],
#     #     gamma=0.1)

# ]

# xuerui param scheduler setting
# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],     # 这里应该需要根据目前的设置进行调整
        gamma=0.1)
]


# optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     paramwise_cfg=dict(
#         custom_keys={
#             # 'absolute_pos_embed': dict(decay_mult=0.),
#             # 'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }),
#     optimizer=dict(
#         _delete_=True,
#         type='AdamW',
#         lr=0.000025,
#         betas=(0.9, 0.999),
#         weight_decay=0.05),
#         # optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)

# )

# xuerui optim wrapper setting
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            # 'absolute_pos_embed': dict(decay_mult=0.),
            # 'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0002,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=0.05),
)



# fp16 = dict(loss_scale=512.)



## hook cuda memory occupy 40GB
# custom_imports = dict(
#     imports=['mmdet.hooks.lock_memory_hook'],
#     allow_failed_imports=False
# )

# custom_hooks = [
#     dict(type='LockMemoryHook', gb=40.0, dtype='auto')  # 自动匹配混合精度
# ]


# custom_imports = dict(
#     imports=['mmdet.hooks.gpu_preheat_hook'],
#     allow_failed_imports=False
# )

# custom_hooks = [
#     dict(type='PreheatGPUHook', gb=20.0)
# ]


# custom_imports = dict(
#     imports=['mmdet.hooks.check_unused_params_hook'],  # 改成你的真实模块路径
#     allow_failed_imports=False
# )
