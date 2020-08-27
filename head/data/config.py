# config.py
cfg_mnet = {
    'name': 'mobilenet0.25',
    'classes': ['background', 'head','helmet'],
    'min_sizes': [[32, 16], [64, 32], [96, 48]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'cls_weight': 1.0,
    'gpu_train': True,
    'batch_size': 96,
    'ngpu': 1,
    'epoch': 500,
    'decay1': 400,
    'decay2': 450,
    'image_size': 384,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 128
}
'''
cfg_mnet = {
    'name': 'mobilenet0.25',
    'classes': ['background', 'head','helmet'],
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 1.0,
    'cls_weight': 1.0,
    'gpu_train': True,
    'batch_size': 48,
    'ngpu': 1,
    'epoch': 150,
    'decay1': 90,
    'decay2': 120,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}
'''
cfg_re50 = {
    'name': 'Resnet50',
    'classes': ['head','helmet'],
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

