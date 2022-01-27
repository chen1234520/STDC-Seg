
cfg = dict(
    model_type='STDC_Del_infer',
    # respth='./res',
    use_boundary_2=False,
    use_boundary_4 = False,
    use_boundary_8 = False,      #原本是True  
    use_boundary_16 = False,
    use_conv_last = False,
    n_classes = 2,  # mgchen
    backbone='STDCNet813Del',
    # methodName = 'train_STDC1-Seg-shiliu20211118',
    inputSize = 192,    #512
    # inputScale = 50,
    # inputDimension = (1, 3, 192, 192),     # (1, 3, 512, 1024)
    # n_cats=171,
    # num_aux_heads=2,
    # lr_start=1e-2,
    # weight_decay=1e-4,
    # warmup_iters=1000,
    # max_iter=90000,
    # dataset='CocoStuff',
    # im_root='./datasets/coco',
    # train_im_anns='./datasets/coco/train.txt',
    # val_im_anns='./datasets/coco/val.txt',
    # scales=[0.5, 2.],
    # cropsize=[512, 512],
    # eval_crop=[512, 512],
    # eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    # ims_per_gpu=4,
    # eval_ims_per_gpu=1,
    # use_fp16=True,
    use_sync_bn=True,
)
