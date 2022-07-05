from __future__ import division

import os
import sys
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import time
import torchvision.transforms as transforms

from thop import profile
sys.path.append("./")

from lib.models.model_stages import BiSeNet
from lib.models.model_stages_del import BiSeNet_delNet

print("No TensorRT")


def to_onnx():
    dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float)
    # model = model_res()
    model = model_osnet()

    input_names = ["data"]
    output_names = ["fc"]
    torch.onnx.export(
        model,
        dummy_input,
        "./osnet.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )
    print("转换模型成功^^")


def pytorch_out(input):
    # model = model_res() #model.eval

    # Configuration ##############
    n_classes = 2  #mgchen
    inputSize = 128
    model = BiSeNet_delNet(backbone= 'STDCNet813Del', n_classes=n_classes, 
    use_boundary_2=False, use_boundary_4=False, 
    use_boundary_8=True, use_boundary_16=False, 
    input_size=inputSize, use_conv_last=False)

    save_pth = 'checkpoints/STDC1-Del-ShengXC-128*96-20220115inside/pths/model_iter29500_mIOU50_0.7989_mIOU75_0.8853.pth'
    model.load_state_dict(torch.load(save_pth))
    model.eval()
    # model.cuda()

    # input = input.cuda()
    # model.cuda()
    torch.no_grad()
    output = model(input)[0]
    # print output[0].flatten()[70:80]
    return output

def pytorch_onnx_test():
    import onnxruntime
    from onnxruntime.datasets import get_example

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # 测试数据
    torch.manual_seed(66)
    dummy_input = torch.randn(1, 3, 96, 128, device='cpu')

    example_model = get_example("/root/project/STDC-Seg-master/tools/STDC1-shengXC_Del-class2_0117-inside_128_96.onnx")
    # netron.start(example_model) 使用 netron python 包可视化网络
    sess = onnxruntime.InferenceSession(example_model)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    # onnx 网络输出
    # onnx_out = np.array(sess.run(None, { "data": to_numpy(dummy_input)}))  #fc 输出是三维列表
    onnx_out = np.array(sess.run([label_name], {input_name: to_numpy(dummy_input)})[0])  #fc 输出是三维列表
    print("==============>")
    print(onnx_out)
    print(onnx_out.shape)
    print("==============>")
    torch_out_res = pytorch_out(dummy_input).detach().numpy()   #fc输出是二维 列表
    print(torch_out_res)
    print(torch_out_res.shape)

    print("===================================>")
    print("输出结果验证小数点后五位是否正确,都变成一维np")

    torch_out_res = torch_out_res.flatten()
    onnx_out = onnx_out.flatten()

    pytor = np.array(torch_out_res,dtype="float32") #need to float32
    onn=np.array(onnx_out,dtype="float32")  ##need to float32
    np.testing.assert_almost_equal(pytor,onn, decimal=5)  #精确到小数点后5位，验证是否正确，不正确会自动打印信息
    print("恭喜你 ^^ ，onnx 和 pytorch 结果一致 ， Exported model has been executed decimal=5 and the result looks good!")


pytorch_onnx_test()

