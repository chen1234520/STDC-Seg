#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from lib.models.model_stages import BiSeNet
from lib.models.model_stages_DW import BiSeNet_DW
from lib.models.model_stages_modifies import BiSeNet_cutNet
from cityscapes import CityScapes 
from cityscapes import CityScapesBingLang
from cityscapes import CityScapesShengXianCheng

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math
import cv2

class MscEvalV0(object):

    def __init__(self, scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, net, dl, n_classes):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()     #label.shape = (batch, W , H) or (batch, H , W)

            size = label.size()[-2:]

            imgs = imgs.cuda()

            N, C, H, W = imgs.size()
            new_hw = [int(H*self.scale), int(W*self.scale)]

            imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True) # 采样
            # import pdb
            # pdb.set_trace()     
            # print("img.shape:{}".format(imgs.shape))

            logits = net(imgs)[0]
  
            logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)  # torch.Size([1, 4, 190, 190])
            probs = torch.softmax(logits, dim=1)                        # torch.Size([1, 4, 190, 190])
            preds = torch.argmax(probs, dim=1)   # 取每个像素点得分最大的类别   # torch.Size([1, 190, 190]

            # np_preds = preds.squeeze().detach().cpu().numpy()
            # np_label = label.squeeze().detach().cpu().numpy()
            # np.savetxt("preds.txt", np_preds)    
            # np.savetxt("label.txt", np_label)

            keep = label != self.ignore_label   # 过滤255的区域(不等于255值为true，等于255值为false)
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes).float()
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        
 
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        # import pdb
        # pdb.set_trace()   
        # # ious = ious[~torch.any(ious.isnan())]
        # p = ious.isnan()
        # p2 = (ious==nan).nonzero()

        # b=torch.cat([torch.cat((ious[0:j],ious[j+1:])) if p[j]==true for j in p])  #用torch.cat()删除所有的1
        # np_ious = ious.detach().cpu().numpy()
        # iou = torch.tensor(np.array([]))
        # # for idex in p:
        #     if p[idex] == True:
        #         iou = torch.cat(iou, ious[idex])

        print("ious:{}".format(ious))
        # print("iou:{}".format(iou))

        miou = ious.mean()
        return miou.item()

def evaluatev0(respth='./pretrained', dspth='./data', backbone='CatNetSmall', scale=0.75, use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False):
    print('scale', scale)
    print('use_boundary_2', use_boundary_2)
    print('use_boundary_4', use_boundary_4)
    print('use_boundary_8', use_boundary_8)
    print('use_boundary_16', use_boundary_16)
    ## dataset
    batchsize = 1
    n_workers = 2

    cropsize = (128, 96)      #  [1024, 512]
    randomscale = (0.625, 0.75, 1.0, 1.25, 1.5)

    dsval = CityScapesShengXianCheng(dspth, mode='val', cropsize=cropsize, randomscale=randomscale)
    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    n_classes = 2  # 19 mgchen
    print("backbone:", backbone)
    net = BiSeNet_cutNet(backbone=backbone, n_classes=n_classes,
     use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, 
     use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16, 
     use_conv_last=use_conv_last)
    net.load_state_dict(torch.load(respth))
    net.cuda()
    net.eval()

    with torch.no_grad():
        single_scale = MscEvalV0(scale=scale)
        mIOU = single_scale(net, dl, n_classes)    #19mgchen
    logger = logging.getLogger()
    logger.info('mIOU is: %s\n', mIOU)

class MscEval(object):
    def __init__(self,
            model,
            dataloader,
            scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75],
            n_classes = 4, # 19 mgchen
            lb_ignore = 255,
            cropsize = 192,    #mgchen 1024
            flip = True,
            *args, **kwargs):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        self.cropsize = cropsize
        ## dataloader
        self.dl = dataloader
        self.net = model


    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0]-H, size[1]-W
        hst, hed = margin_h//2, margin_h//2+H
        wst, wed = margin_w//2, margin_w//2+W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]


    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)[0]
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop)[0]
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
            prob = torch.exp(prob)
        return prob


    def crop_eval(self, im):
        cropsize = self.cropsize
        stride_rate = 5/6.
        N, C, H, W = im.size()
        long_size, short_size = (H,W) if H>W else (W,H)
        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        else:
            stride = math.ceil(cropsize*stride_rate)
            if short_size < cropsize:
                if H < W:
                    im, indices = self.pad_tensor(im, (cropsize, W))
                else:
                    im, indices = self.pad_tensor(im, (H, cropsize))
            N, C, H, W = im.size()
            n_x = math.ceil((W-cropsize)/stride)+1
            n_y = math.ceil((H-cropsize)/stride)+1
            prob = torch.zeros(N, self.n_classes, H, W).cuda()
            prob.requires_grad = False
            for iy in range(n_y):
                for ix in range(n_x):
                    hed, wed = min(H, stride*iy+cropsize), min(W, stride*ix+cropsize)
                    hst, wst = hed-cropsize, wed-cropsize
                    chip = im[:, :, hst:hed, wst:wed]
                    prob_chip = self.eval_chip(chip)
                    prob[:, :, hst:hed, wst:wed] += prob_chip
            if short_size < cropsize:
                prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        return prob


    def scale_crop_eval(self, im, scale):
        N, C, H, W = im.size()
        new_hw = [int(H*scale), int(W*scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob


    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.lb_ignore
        keep = np.logical_not(lb==ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist


    def evaluate(self):
        ## evaluate
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        dloader = tqdm(self.dl)
        if dist.is_initialized() and not dist.get_rank()==0:
            dloader = self.dl
        for i, (imgs, label) in enumerate(dloader):
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            for sc in self.scales:
                # prob = self.scale_crop_eval(imgs, sc)
                prob = self.eval_chip(imgs)
                probs += prob.detach().cpu()
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)

            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once
        IOUs = np.diag(hist) / (np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist))
        import pdb
        pdb.set_trace()
        IOUs = IOUs[~torch.any(IOUs.isnan(),dim=1)]
        mIOU = np.mean(IOUs)
        return mIOU


def evaluate(respth='./resv1_catnet/pths/', dspth='./data'):
    ## logger
    logger = logging.getLogger()

    ## model
    logger.info('\n')
    logger.info('===='*20)
    logger.info('evaluating the model ...\n')
    logger.info('setup and restore model')
    n_classes = 4      #19 mgchen
    net = BiSeNet(n_classes=n_classes)

    net.load_state_dict(torch.load(respth))
    net.cuda()
    net.eval()

    ## dataset
    batchsize = 5
    n_workers = 2
    dsval = CityScapes(dspth, mode='val')
    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(net, dl, scales=[1], flip = False)

    ## eval
    mIOU = evaluator.evaluate()
    logger.info('mIOU is: {:.6f}'.format(mIOU))



if __name__ == "__main__":
    log_dir = 'evaluation_logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    setup_logger(log_dir)
    
    # STDC1-Seg50 mIoU 0.7222
    # evaluatev0('./checkpoints/train_STDC1-Seg/pths/model_maxmIOU50.pth', dspth='./data', backbone='STDCNet813', scale=0.5, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

    # 测试石榴任务
    # evaluatev0('./checkpoints/train_STDC1-Seg-20211117/pths/model_final.pth', dspth='./data/shiliu', backbone='STDCNet813', scale=1, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

    # 测试槟榔任务
    # model_path = './checkpoints/train_STDC1-Seg-lifadian20211130/pths/model_iter205000_mIOU50_0.5056_mIOU75_0.8916.pth'
    # evaluatev0(model_path, dspth='./data/seg-data', backbone='STDCNet813', scale=1, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

    # 测试槟榔任务
    model_path = 'checkpoints/STDC1-Cut-ShengXC-128*96-20220113/pths/model_iter28000_mIOU50_0.7699_mIOU75_0.9086.pth'
    evaluatev0(model_path, dspth='/root/mgchen/data/shengxiancheng/', backbone='STDCNet813', scale=1,
    use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

    #STDC1-Seg75 mIoU 0.7450
    # evaluatev0('./checkpoints/STDC1-Seg/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet813', scale=0.75, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)


    #STDC2-Seg50 mIoU 0.7424
    # evaluatev0('./checkpoints/STDC2-Seg/model_maxmIOU50.pth', dspth='./data', backbone='STDCNet1446', scale=0.5, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

    #STDC2-Seg75 mIoU 0.7704
    # evaluatev0('./checkpoints/STDC2-Seg/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet1446', scale=0.75, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

   

