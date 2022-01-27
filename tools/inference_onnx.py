#!/usr/bin/python
# -*- encoding: utf-8 -*-

import onnxruntime as rt
import numpy as  np
import cv2
import torchvision.transforms as transforms
import time
import torch

to_tensor = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),     # 均值归一化
])

# 加载图片
# data = np.array(np.random.randn(1, 3, 512, 1024))
img_path = '/root/chenguang/data/binglang/leftImg8bit/val/binglang/MV-UB300#571568B8-1-Snapshot-20210428145405-4854703415155_leftImg8bit.png'
w, h = (128, 96)
img = cv2.imread(img_path)
resized = cv2.resize(img, (w, h))
resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
# tensor = transforms.ToTensor()(resized)
tensor = to_tensor(resized)
tensor = tensor.unsqueeze_(0)

# 加载模型
# sess = rt.InferenceSession('tools/STDC1-shiliu_192_192.onnx')
# sess = rt.InferenceSession('tools/STDC1-binglang_320_320.onnx')
sess = rt.InferenceSession('tools/STDC1-shengXC_Cut-new2_128_96.onnx')

# sess.set_providers(['CUDAExecutionProvider'], [ {'device_id': 1}])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# torch.cuda.synchronize()
t0 = time.time()
iter_num = 100
for item in list(range(iter_num)):
    # 执行推理
    # pred_onx = sess.run([label_name], {input_name:data.astype(np.float32)})[0]
    pred_onx = sess.run([label_name], {input_name: tensor.cpu().numpy()})[0]
# torch.cuda.synchronize()
t1 = time.time()
latency = (t1-t0) / iter_num * 1000
FPS = 1000. / latency
print("Total running time: %s s" % (str(t1 - t0)))
print("latency time:{}ms".format(latency))
print("FPS: {}".format(int(FPS)))
# print(pred_onx)
# print(pred_onx.shape)
# print(pred_onx.dtype)
# print(np.argmax(pred_onx)

# probs = torch.softmax(pred_onx, dim=0)   # torch.Size([4, 190, 190])
# preds = torch.argmax(probs, dim=0)   # 取每个像素点得分最大的类别   torch.Size([4, 190])
preds = np.argmax(pred_onx, axis = 1)
preds = np.squeeze(preds, axis = None)
# preds = softmax(preds, axis=0)
print(preds.shape)

# aaa = preds.squeeze().detach().cpu().numpy()    # (190, 190)
np.savetxt("./dets2_onnx.txt", preds, fmt='%d', delimiter=' ')

palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)   # (256, 3)
pred = palette[preds]     # (190, 190, 3)
print("pred.shape:{}".format(pred.shape))

add_img = cv2.addWeighted(pred, 0.4, resized, 0.6, 0) #图像融合
cv2.imwrite('./add_img_onnx.jpg', add_img)
cv2.imwrite('./demo_res_onnx.jpg', pred)
# cv2.imwrite('./demo_out.jpg', preds)