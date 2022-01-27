import torch
from collections import OrderedDict
import os
import torch.nn as nn
import torch.nn.init as init

import argparse
import sys
sys.path.insert(0, '.')


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
args.weight_pth = 'checkpoints/STDC1-cut-shengXC-128*96-20220107/pths/model_final.pth'
# config文件
# args.config = './configs/STDC1-Seg.py'
args.config = './configs/STDC1-Cut.py'
# args.config = './configs/STDC1-DW.py'

new_model_savepath = "./tools/STDC1-shengXC_Cut-new.pth"

# 设置输入尺寸宽高
# w, h = (128, 96)
# # onnx模型保存路径
# onnxmodel_name = 'STDC1-shengXC_Cut-new'
# args.out_pth = './tools/{}_{}_{}.onnx'.format(onnxmodel_name, w, h)
# dummy_input = torch.randn(1, 3, h, w)       # N C H W

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
# net.load_state_dict(torch.load(args.weight_pth, map_location='cpu'), strict=False)


def init_weight(modules):    
    for m in modules:        
        if isinstance(m, nn.Conv2d):            
            init.xavier_uniform_(m.weight.data)            
            if m.bias is not None:                
                m.bias.data.zero_()        
        elif isinstance(m, nn.BatchNorm2d):            
            m.weight.data.fill_(1)            
            m.bias.data.zero_()        
        elif isinstance(m, nn.Linear):            
            m.weight.data.normal(0,0.01)            
            m.bias.data.zero_() 
            
def copyStateDict(state_dict):    
    if list(state_dict.keys())[0].startswith('module'):        
        start_idx = 1    
    else:        
        start_idx = 0    
    new_state_dict = OrderedDict()    
    for k,v in state_dict.items():        
        name = ','.join(k.split('.')[start_idx:])        
        new_state_dict[name] = v    
    return new_state_dict 

#加载pretrain model
state_dict = torch.load(args.weight_pth) 

# import pdb
# pdb.set_trace()

new_dict = copyStateDict(state_dict)
keys = []
for k,v in new_dict.items():    
    if k.startswith('conv_out_sp'):    #将‘conv_out_sp’开头的key过滤掉，这里是要去除的层的key        
        continue    
    keys.append(k) 

#去除指定层后的模型
new_dict = {k:new_dict[k] for k in keys} 

# net = new_VGG()   #自己定义的模型，但要保证前面保存的层和自定义的模型中的层一致 

#加载pretrain model中的参数到新的模型中，此时自定义的层中是没有参数的，在使用的时候需要init_weight一下
net.state_dict().update(new_dict) 

print(net.state_dict().keys())

#保存去除指定层后的模型
torch.save(net.state_dict(), new_model_savepath)