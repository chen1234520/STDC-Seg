from __future__ import division

import os
import sys
import logging
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import time
import torchvision.transforms as transforms

import pdb

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

try:
    from lib.models.model_stages_trt import BiSeNet
except:
    from lib.models.model_stages import BiSeNet
    from lib.models.model_stages_DW import BiSeNet_DW
    from lib.models.model_stages_modifies import BiSeNet_cutNet
    from lib.models.model_stages_del import BiSeNet_delNet
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
    n_classes = 2  #mgchen

    # 设置输入 输出路径
    # srcPath = "data/shengxiancheng/leftImg8bit/val/shengxiancheng20220111"      # 输入数据
    srcPath = "data/20220124/img"      # 输入数据
    savePath = "data/20220124/pytorch"                   # 保存结果路径
    model_path = 'tools/STDC1-lifadian_320_320.pth'   # 模型路径

    model_type = 'BiSeNet'

    if model_type == 'BiSeNet_cutNet':
        Net_Model = BiSeNet_cutNet
        backbone = 'STDCNet813'
    elif model_type == 'BiSeNet_DW':
        backbone = 'STDCNet813'
        Net_Model = BiSeNet_DW
    elif model_type == 'BiSeNet_Del':
        backbone = 'STDCNet813Del'
        Net_Model = BiSeNet_delNet    
    else:   # BiSeNet
        backbone = 'STDCNet813'
        Net_Model = BiSeNet

    # case1.STDC1Seg-50 250.4FPS on NVIDIA GTX 1080Ti
    # backbone = 'STDCNet813'
    # methodName = 'train_STDC1-Seg-20211118'
    # inputSize = 192
    # inputScale = 50
    # inputDimension = (1, 3, 192, 192)

    # 槟榔 1024*512
    # methodName = 'train_STDC1-Seg-bianglang20211126'
    inputSize = 128
    # inputScale = 50
    w, h = (320, 320)
    # inputDimension = (1, 3, h, w)

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
    
    model = Net_Model(backbone=backbone, n_classes=n_classes, 
    use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, 
    use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16, 
    input_size=inputSize, use_conv_last=use_conv_last)
    
    # model = BiSeNet(backbone=backbone, n_classes=n_classes, 
    # use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, 
    # use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16, 
    # input_size=inputSize, use_conv_last=use_conv_last)

    print('loading parameters...')
    # respth = './checkpoints/{}/pths/'.format(methodName)
    # model_path = os.path.join(respth, 'model_maxmIOU50.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()
    #####################################################

    # to_tensor = ToTensor(
    # mean = (0.485, 0.456, 0.406), # city, rgb
    # std = (0.229, 0.224, 0.225),
    # )

    to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),     # 均值归一化
    ])

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    filename = os.listdir(srcPath)     # 获取文件夹中所以文件名
    t0 = time.time()
    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)   # (256, 3)
    print(palette.shape)
    print(palette)
    for path in filename:
        # print(path)
        imgName = path.rsplit('.', 1)[0]

        srcimg = cv2.imread(srcPath + '/' + path)[:, :, ::-1]  # bgr->rgb
        img = cv2.resize(srcimg, (w, h), interpolation = cv2.INTER_AREA)
        # im = to_tensor(dict(im=img, lb=None))['im'].unsqueeze(0).cuda()
        im = to_tensor(img).unsqueeze(0).cuda()

        # im = im.transpose(2, 0, 1).astype(np.float32)
        # im = torch.from_numpy(im).div_(255).unsqueeze(0).cuda()

        # inference
        # out = model(im).squeeze().detach().cpu().numpy().astype('int64')
        out = model(im)[0]      # torch.Size([4, 190, 190])

        probs = torch.softmax(out, dim=1)   # torch.Size([4, 190, 190])
        preds = torch.argmax(probs, dim=1)   # 取每个像素点得分最大的类别   torch.Size([4, 190])
        aaa = preds.squeeze().detach().cpu().numpy()    # (190, 190)

        pred = palette[aaa]     # (190, 190, 3)

        # print(srcimg.shape)
        pred = cv2.resize(pred, (srcimg.shape[1], srcimg.shape[0]), interpolation = cv2.INTER_NEAREST)
        add_img = cv2.addWeighted(pred, 0.4, srcimg, 0.6, 0) #图像融合

        # add_img = cv2.addWeighted(pred, 0.4, img, 0.6, 0) #图像融合


        cv2.imwrite(savePath + '/' + imgName +'_add_img.jpg', add_img)
        # cv2.imwrite(savePath + '/' + imgName +'_demo_res.jpg', pred)

        # cv2.imwrite('./demo_out.jpg', aaa)

        ###########################
        # prob = F.softmax(out, 1).detach().cpu().numpy().astype('int64')
        # palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
        # pred = palette[prob]
        # cv2.imwrite('./res2.jpg', pred)
        ###########################

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
    t1 = time.time()
    print("Total running time: %s s" % (str(t1 - t0)))

if __name__ == '__main__':
    main() 
