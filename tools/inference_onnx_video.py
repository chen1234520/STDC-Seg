#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
from re import L
import onnxruntime as rt
import numpy as  np
import cv2
import torchvision.transforms as transforms
import time

to_tensor = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),     # 均值归一化
])

# 加载模型
sess = rt.InferenceSession('tools/STDC1-lifadian_20220126_320_320.onnx')
# sess.set_providers(['CUDAExecutionProvider'], [ {'device_id': 1}])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

t0 = time.time()

srcPath = "/home/mgchen/data/2022-01-23/video"
savePath = "/home/mgchen/data/2022-01-23/分割效果差视频"
w,h = (320, 320)

palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)   # (256, 3)  # 显示mask的随机颜色

filename = os.listdir(srcPath)     # 获取文件夹中所以文件名
t0 = time.time()
for path in filename:
    nums = 0
    if nums % 3 != 0:
        continue

    # print(path)
    imgName = path.rsplit('.', 1)[0]

    saveImg = os.path.join(savePath, "img", imgName)     # 创建二级目录
    if not os.path.exists(saveImg):
        os.makedirs(saveImg)
    # saveImg = os.path.join(saveImg, imgName)
    # if not os.path.exists(saveImg):
    #     os.makedirs(saveImg)

    saveResult = os.path.join(savePath, "result", imgName)     # 创建二级目录
    if not os.path.exists(saveResult):
        os.makedirs(saveResult)

    cap = cv2.VideoCapture(srcPath + '/' + path)
    while(1):
        nums = nums + 1
        ret, frame = cap.read()
        cv2.imwrite(saveImg + '/' + imgName + "-" + str(nums) +'.jpg', frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        
        if frame is None:
            break

        resized = cv2.resize(frame, (w, h))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
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
        pred = cv2.resize(pred, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_NEAREST)
        add_img = cv2.addWeighted(pred, 0.4, frame, 0.6, 0) #图像融合
        # add_img = cv2.addWeighted(pred, 0.4, resized, 0.6, 0) #图像融合
        cv2.imwrite(saveResult + '/' + imgName + str(nums) +'_add_img_onnx.jpg', add_img)
        cv2.imshow("show",add_img)
        # cv2.imwrite(savePath + '/' + imgName +'_demo_res_onnx.jpg', pred)

    t1 = time.time()
    print("Total running time: %s s" % (str(t1 - t0)))
    print("average running time: %s s" % (str((t1 - t0) / nums)))

# print(pred_onx)
# print(pred_onx.shape)
# print(pred_onx.dtype)
# print(np.argmax(pred_onx)

