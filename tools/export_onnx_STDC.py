import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')

import torch

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.models.model_stages import BiSeNet
from torchsummary import summary
torch.set_grad_enabled(False)

parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str,
        default='configs/bisenetv2.py',)
parse.add_argument('--weight-path', dest='weight_pth', type=str,
        default='model_final.pth')
parse.add_argument('--outpath', dest='out_pth', type=str,
        default='model.onnx')
parse.add_argument('--aux-mode', dest='aux_mode', type=str,
        default='pred')
args = parse.parse_args()

# pth模型路径
args.weight_pth = 'checkpoints/train_STDC1-Seg-lifadian20220126/pths/model_iter116000_mIOU50_0.6309_mIOU75_0.9243.pth'
# args.weight_pth = 'checkpoints/STDC1-Del-ShengXC-128*96-20220115inside/pths/model_iter29500_mIOU50_0.7989_mIOU75_0.8853.pth'

# args.weight_pth = 'tools/STDC1-shengXC_Cut-new.pth'

# config文件
args.config = './configs/STDC1-Seg.py'
# args.config = './configs/STDC1-Cut.py'
# args.config = './configs/STDC1-DW.py'
# args.config = './configs/STDC1-Cut_infer.py'
# args.config = './configs/STDC1-Del_infer.py'


# 设置输入尺寸宽高
w, h = (320, 320)
# onnx模型保存路径
# onnxmodel_name = 'STDC1-shengXC_Del-class2_0117-inside'
onnxmodel_name = 'STDC1-lifadian_20220126'
args.out_pth = './tools/{}_{}_{}.onnx'.format(onnxmodel_name, w, h)

dummy_input = torch.randn(1, 3, h, w)       # N C H W

cfg = set_cfg_from_file(args.config)

if cfg.use_sync_bn: 
        cfg.use_sync_bn = False

if cfg.model_type == 'bisenetv1' or cfg.model_type == 'bisenetv2':
    net = model_factory[cfg.model_type](cfg.n_cats, aux_mode=args.aux_mode)
else:
    net = model_factory[cfg.model_type](backbone=cfg.backbone, n_classes=cfg.n_classes,
                                        use_boundary_2=cfg.use_boundary_2, use_boundary_4=cfg.use_boundary_4,
                                        use_boundary_8=cfg.use_boundary_8, use_boundary_16=cfg.use_boundary_16,
                                        use_conv_last=cfg.use_conv_last)
net.load_state_dict(torch.load(args.weight_pth, map_location='cpu'), strict=False)

net.eval()
summary(net,(3, w, h), batch_size=1,device="cpu")
# 模型输入输出名称
input_names = ['input_image']
output_names = ['preds',]

torch.onnx.export(net, 
                                        dummy_input, 
                                        args.out_pth,
                                        input_names=input_names, 
                                        output_names=output_names,
                                        verbose=True, 
                                        opset_version=11)

