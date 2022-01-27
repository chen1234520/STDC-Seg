#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import onnxruntime as rt
import numpy as  np
import cv2
import torchvision.transforms as transforms
import time


w, h = (128, 96)
# srcPath = "data/shengxiancheng/leftImg8bit/val/shengxiancheng20220104"
srcPath = "data/shengxiancheng-inside/leftImg8bit/val/shengxiancheng"
savePath = "result/shengxiancheng-inside-CutDel-0117/onnx-infer"
if not os.path.exists(savePath):
    os.makedirs(savePath)

# 加载模型
sess = rt.InferenceSession('tools/STDC1-shengXC_Del-class2_0117-inside_128_96.onnx')
# sess.set_providers(['CUDAExecutionProvider'], [ {'device_id': 1}])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

to_tensor = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),     # 均值归一化
])

t0 = time.time()

filename = os.listdir(srcPath)     # 获取文件夹中所以文件名
t0 = time.time()
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)   # (256, 3)
for path in filename:
    # print(path)
    imgName = path.rsplit('.', 1)[0]

    # img = cv2.imread(srcPath + '/' + path)
    img = cv2.imread(srcPath + '/' + path)[:, :, ::-1]  # bgr->rgb
    resized = cv2.resize(img, (w, h))
    # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # tensor = transforms.ToTensor()(resized)
    tensor = to_tensor(resized)
    tensor = tensor.unsqueeze_(0)

    # 执行推理
    pred_onx = sess.run([label_name], {input_name: tensor.cpu().numpy()})[0]

    preds = np.argmax(pred_onx, axis = 1)
    preds = np.squeeze(preds, axis = None)
    # print(preds.shape)
    # np.savetxt("./dets2_onnx.txt", preds, fmt='%d', delimiter=' ')

    pred = palette[preds]     # (190, 190, 3)
    # print("pred.shape:{}".format(pred.shape))

    pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)  # 扩充到原图大小

    # add_img = cv2.addWeighted(pred, 0.4, resized, 0.6, 0) #图像融合
    add_img = cv2.addWeighted(pred, 0.4, img, 0.6, 0) #图像融合

    cv2.imwrite(savePath + '/' + imgName +'_add_img_onnx.jpg', add_img)
    # cv2.imwrite(savePath + '/' + imgName +'_demo_res_onnx.jpg', pred)

t1 = time.time()
print("Total running time: %s s" % (str(t1 - t0)))

# print(pred_onx)
# print(pred_onx.shape)
# print(pred_onx.dtype)
# print(np.argmax(pred_onx)

