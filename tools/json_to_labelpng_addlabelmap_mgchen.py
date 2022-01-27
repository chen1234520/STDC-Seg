#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
function:根据coco的json标注文件，生成灰度png标签
使用说明：
输入参数1：json_file 提供json文件夹的路径
输入参数2：outputpath 输出结果保存的路径(包含各种形式的png文件)
"""

import argparse
import json
import os
import os.path as osp
import base64
import warnings

import PIL.Image
import yaml

from labelme import utils

import cv2
import numpy as np
from skimage import img_as_ubyte
# from sys import argv

def main():
    warnings.warn(
        "This script is aimed to demonstrate how to convert the\n"
        "JSON file to a single image dataset, and not to handle\n"
        "multiple JSON files to generate a real-use dataset."


    # label_name_to_value = {"_background_": 0,   "all": 1, "flower": 2, "part": 3}
    label_name_to_value = {"_background_": 0,"1": 1}
    json_file = "/home/mgchen/data/2022-01-23/1/json/"      # json文件路径
    outputpath = "/home/mgchen/data/2022-01-23/1/"          # 结果输出路径

    if not osp.exists(outputpath):  # 创建image保存文件夹
        os.mkdir(outputpath)

    # freedom
    list_path = os.listdir(json_file)
    print("freedom =", json_file)
    for i in range(0, len(list_path)):
        print('process_{}...'.format(list_path[i]))
        path = os.path.join(json_file, list_path[i])
        if os.path.isfile(path):
            data = json.load(open(path))  # 读取json
            img = utils.img_b64_to_arr(data["imageData"])  # 读取图片

            # 方法1：使用labelme_shapes_to_label,无法传递label_map进接口内部
            # # labelme_api .根据json得到灰度png标签图
            # lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data["shapes"])

            # 方法2：使用utils.shapes_to_label.可以直接传递label_map到接口内部
            shapes = data["shapes"]  # json文件中的标注内容

            ## 查看json文件中的label是否在label_map里，没有的话添加进去
            for shape in shapes:   
                label_name = shape["label"]
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
                    print("add_new_label:{}".format(label_name))
        
            ## 对label进行排序，将类别'all'放在对前面(为了将其覆盖掉)
            label_all = []
            label_other = []
            for shape in shapes:    
                label_name = shape["label"]
                if label_name == 'all':
                    label_all.append(shape)
                else:
                    label_other.append(shape)
            label_all.extend(label_other)
            shapes = label_all

            # 提取指定的label标签
            labels_select = []
            if len(shapes) > 1: # 存在多个标签时
                for shape in shapes:
                    if shape["label"] == "goods-in-baoxiandai" or shape["label"] == "goos-in-baoxiandai":
                        labels_select.append(shape)
                if len(labels_select) != 0:
                    shapes = labels_select

            # labels_modify = []
            # for shape in shapes:
            #     shape["label"] = "fresh"
            #     labels_modify.append(shape)
            # shapes = labels_modify

            ## labelme_api .根据json得到灰度png标签图
            lbl = utils.shapes_to_label(img.shape, shapes, label_name_to_value)     
            lbl_names = label_name_to_value

            # 灰度图png保存为txt
            # txt = []
            # for y in lbl:
            #     txt.append(" ".join([str(x) for x in y]))
            # content = "\n".join(txt)
            # with open("a.txt", "w+", encoding="utf-8") as f:
            #     f.write(content)

            captions = [ "%d: %s" % (l, name) for l, name in enumerate(lbl_names)]  # 获取类别字典键值
            # print(captions)

            # # 画彩色RGB图
            lbl_viz = utils.draw_label(lbl, img, captions)
            # # out_dir = osp.basename(path).replace('.', '_')
            out_dir = osp.basename(path).split(".json")[0]  # 获取图片名称
            save_file_name = out_dir
            # out_dir = osp.join(osp.dirname(path), out_dir)

            # case1.保存原图
            if not osp.exists(outputpath + "image"):  # 创建image保存文件夹
                os.mkdir(outputpath + "image")
            imagedir = outputpath + "image"
            PIL.Image.fromarray(img).save(imagedir + '/' + save_file_name + '_leftImg8bit.png')

            # case2.保存灰度png
            if not osp.exists(outputpath + "mask"):  # 创建mask保存文件夹(灰度png)(model=I)
                os.mkdir(outputpath + "mask")
            maskdir = outputpath + "mask"
            out_dir1 = maskdir
            PIL.Image.fromarray(lbl).save(out_dir1 + "/" + save_file_name + "_label.png")

            # case3.保存彩色rgb_png
            if not osp.exists(outputpath + "mask_rgb"):  # 创建mask_rgb保存文件夹(彩色png)
                os.mkdir(outputpath + "mask_rgb")
            mask_rgbdir = outputpath + "mask_rgb"
            utils.lblsave(mask_rgbdir + "/" + save_file_name +  "_label_rgb.png", lbl)  # 1)保存彩色png标注

            # case4.保存"L"模式的png
            if not osp.exists(outputpath + "mask_L"):  # 创建mask_rgb保存文件夹(彩色png)
                os.mkdir(outputpath + "mask_L")
            mask_Ldir = outputpath + "mask_L"
            PIL.Image.fromarray(lbl).convert('L').save(mask_Ldir + "/" + save_file_name + "_label_L.png")
            # utils.lblsave_L(mask_Ldir + "/" + save_file_name +  "_label_L.png", lbl)  #lblsave_L为自定义函数,丢失了。现在无法使用

            # case5.保存彩色viz.png
            if not osp.exists(outputpath + "mask_viz"):  # 创建mask_viz保存文件夹(彩色png)
                os.mkdir(outputpath + "mask_viz")
            maskvizdir = outputpath + "mask_viz"
            PIL.Image.fromarray(lbl_viz).save(maskvizdir + "/" + save_file_name + "_label_viz.png")

            ################################
            # mask_pic = cv2.imread(out_dir1+'\\'+save_file_name+'_label.png',)
            # print('pic1_deep:',mask_pic.dtype)

            # mask_dst = img_as_ubyte(lbl)  # mask_pic
            # print('pic2_deep:', mask_dst.dtype)
            # cv2.imwrite(mask_save2png_path + '\\' + save_file_name + '_label.png', mask_dst)
            ##################################

            # 保存label_names.txt
            with open(osp.join(outputpath, "label_names.txt"), "w") as f:
                for lbl_name in label_name_to_value:
                    f.write(lbl_name + "\n")

            # 保存info.yaml(标签和对应的类别号)
            warnings.warn("info.yaml is being replaced by label_names.txt")
            info = dict(label_names=label_name_to_value)
            with open(osp.join(outputpath, "info.yaml"), "w") as f:
                yaml.safe_dump(info, f, default_flow_style=False)

            # print("Saved to: %s" % out_dir1)


if __name__ == "__main__":
    # base64path = argv[1]
    main()
