from __future__ import division

import os
import sys
import logging
import torch
import numpy as np
import cv2
import torch.nn.functional as F


from thop import profile
sys.path.append("./")

from transform import ToTensor


from utils.darts_utils import create_exp_dir, plot_op, plot_path_width, objective_acc_lat
try:
    from utils.darts_utils import compute_latency_ms_tensorrt as compute_latency
    print("use TensorRT for latency test")
except:
    from utils.darts_utils import compute_latency_ms_pytorch as compute_latency
    print("use PyTorch for latency test")
# from utils.darts_utils import compute_latency_ms_pytorch as compute_latency
# print("use PyTorch for latency test")

try:
    from lib.models.model_stages_trt import BiSeNet
except:
    from lib.models.model_stages import BiSeNet
    print("No TensorRT")


def main():
    
    print("begin")
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Configuration ##############
    use_boundary_2 = False
    use_boundary_4 = False
    use_boundary_8 = True
    use_boundary_16 = False
    use_conv_last = False
    n_classes = 4  #mgchen
    
    # case1.STDC1Seg-50 250.4FPS on NVIDIA GTX 1080Ti
    backbone = 'STDCNet813'
    methodName = 'train_STDC1-Seg-20211118'
    inputSize = 192
    inputScale = 50
    inputDimension = (1, 3, 192, 192)

    # case2.STDC1Seg-75 126.7FPS on NVIDIA GTX 1080Ti
    # backbone = 'STDCNet813'
    # methodName = 'STDC1-Seg'
    # inputSize = 768
    # inputScale = 75
    # inputDimension = (1, 3, 768, 1536)

    # case3.STDC2Seg-50 188.6FPS on NVIDIA GTX 1080Ti
    # backbone = 'STDCNet1446'
    # methodName = 'STDC2-Seg'
    # inputSize = 512
    # inputScale = 50
    # inputDimension = (1, 3, 512, 1024)

    # case4.STDC2Seg-75 97.0FPS on NVIDIA GTX 1080Ti
    # backbone = 'STDCNet1446'
    # methodName = 'STDC2-Seg'
    # inputSize = 768
    # inputScale = 75
    # inputDimension = (1, 3, 768, 1536)
    
    model = BiSeNet(backbone=backbone, n_classes=n_classes, 
    use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, 
    use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16, 
    input_size=inputSize, use_conv_last=use_conv_last)
    

    print('loading parameters...')
    respth = './checkpoints/{}/pths/'.format(methodName)
    save_pth = os.path.join(respth, 'model_final.pth')
    model.load_state_dict(torch.load(save_pth))
    model.eval()
    model.cuda()
    #####################################################

    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    to_tensor = ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
    )
    im = cv2.imread("/root/chenguang/project/STDC-Seg-master/data/shiliu/leftImg8bit/val/shiliu/20210505095657752_leftImg8bit.png")[:, :, ::-1]
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
    # inference
    # out = model(im).squeeze().detach().cpu().numpy().astype('int64')
    out = model(im)[0]
    prob = F.softmax(out, 1).detach().cpu().numpy().astype('int64')

    print("out")
    print(out.shape)
    print(prob.shape)
    print(prob)

    pred = palette[prob]

    print("pred")
    print()

    cv2.imwrite('./res.jpg', pred)
    print("access!")



    # latency = compute_latency(model, inputDimension)
    # print("{}{} FPS:".format(methodName, inputScale) + str(1000./latency))
    # logging.info("{}{} FPS:".format(methodName, inputScale) + str(1000./latency))

    # calculate FLOPS and params
    '''
    model = model.cpu()
    flops, params = profile(model, inputs=(torch.randn(inputDimension),), verbose=False)
    print("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    logging.info("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    '''


if __name__ == '__main__':
    main() 
