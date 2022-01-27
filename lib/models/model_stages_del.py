#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


from .stdcnet import STDCNet1446, STDCNet813,STDCNet813_Del
# from modules.bn import InPlaceABNSync as BatchNorm2d
BatchNorm2d = nn.BatchNorm2d    #mgchen

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)         #mgchen
        # self.bn = BatchNorm2d(out_chan, activation='none')
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)   #mgchen
        # self.bn_atten = BatchNorm2d(out_chan, activation='none')

        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        # x_shape = [int(s) for s in feat.shape[2:]]
        # atten = F.avg_pool2d(feat, x_shape)
        # atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)  # 全局平均池化

        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, backbone='CatNetSmall', pretrain_model='', use_conv_last=False, *args, **kwargs):
        super(ContextPath, self).__init__()
        
        self.backbone_name = backbone
        if backbone == 'STDCNet1446':
            self.backbone = STDCNet1446(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)

        elif backbone == 'STDCNet813':
            self.backbone = STDCNet813(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)
        elif backbone == 'STDCNet813Del':   #删除下采样32倍的结构
            self.backbone = STDCNet813_Del(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            # self.arm32 = AttentionRefinementModule(inplanes, 128)
            # self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            # self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)    
            self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)    
        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]

        # step1.经过主干网络
        # feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        feat2, feat4, feat8, feat16 = self.backbone(x) #mgchen

        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        # H32, W32 = feat32.size()[2:]
        
        # step2.全局平均池化
        # avg = F.avg_pool2d(feat32, feat32.size()[2:])
        # feat32_shape = [int(s) for s in feat32.shape[2:]]
        # avg = F.avg_pool2d(feat32, feat32_shape)
        # avg = torch.mean(feat32, dim=(2, 3), keepdim=True)  # 全局平均池化
        avg = torch.mean(feat16, dim=(2, 3), keepdim=True)  # 全局平均池化
        # print("feat16:{}".format(feat16.size()))

        avg = self.conv_avg(avg)    # 1*1 conv
        # avg_up = F.interpolate(avg, (H32, W32), mode='nearest')     # 上采样到32X
        avg_up = F.interpolate(avg, (H16, W16), mode='nearest')     # 上采样到32X


        # step3. 32X与全局平均池化结果融合
        # feat32_arm = self.arm32(feat32)     # 32X的ARM模块
        # feat32_sum = feat32_arm + avg_up    # 特征融合(32X + avg)
        # feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')   #上采样到16X
        # feat32_up = self.conv_head32(feat32_up) # 3*3conv

        feat16_arm = self.arm16(feat16)             # 16X的ARM模块
        # feat16_sum = feat16_arm + feat32_up     # 特征融合(16X + 32X)
        # print(feat16_arm.size())
        # print(avg_up.size())
        feat16_sum = feat16_arm + avg_up     # 特征融合(16X + 32X)
        
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')  #上采样到8X
        feat16_up = self.conv_head16(feat16_up)     # 3*3conv
        
        #            2X       4X        8X       16X        16up            32up
        # return feat2, feat4, feat8, feat16, feat16_up, feat32_up # x8, x16
        return feat2, feat4, feat8, feat16, feat16_up # x8

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        # atten = F.avg_pool2d(feat, feat.size()[2:])
        # feat_shape = [int(s) for s in feat.shape[2:]]
        # atten = F.avg_pool2d(feat, feat_shape)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)  # 全局平均池化

        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

# 训练网络结构,delNet的意思是:删除了网络中下采样32倍的模块(为了提升推理速度)
class BiSeNet_delNet(nn.Module):
    def __init__(self, backbone, n_classes, pretrain_model='', use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False, heat_map=False, *args, **kwargs):
        super(BiSeNet_delNet, self).__init__()
        
        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        # self.heat_map = heat_map
        self.cp = ContextPath(backbone, pretrain_model, use_conv_last=use_conv_last)

        if backbone == 'STDCNet1446':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        elif backbone == 'STDCNet813':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes
        elif backbone == 'STDCNet813Del':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes
        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.ffm = FeatureFusionModule(inplane, 128)    # FFN融合模块
        self.conv_out = BiSeNetOutput(128, 128, n_classes)          # 256 --> 128  mgchen
        self.conv_out16 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)
        # self.conv_out32 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)

        self.conv_out_sp16 = BiSeNetOutput(sp16_inplanes, 64, 1)
        
        self.conv_out_sp8 = BiSeNetOutput(sp8_inplanes, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(sp4_inplanes, 64, 1)      # 4X的热力图
        self.conv_out_sp2 = BiSeNetOutput(sp2_inplanes, 64, 1)      # 2X的热力图
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        
        # feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)
        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8 = self.cp(x)

        feat_out_sp2 = self.conv_out_sp2(feat_res2)     # 2X的热力图

        feat_out_sp4 = self.conv_out_sp4(feat_res4)     # 4X的热力图
  
        feat_out_sp8 = self.conv_out_sp8(feat_res8)     # 8X的热力图

        feat_out_sp16 = self.conv_out_sp16(feat_res16)       # 16X的热力图

        feat_fuse = self.ffm(feat_res8, feat_cp8)   # 融合stage3(8X)和stage4上采样后 的特征图

        # 网络输出结果
        feat_out = self.conv_out(feat_fuse)               # 8X的输出output
        feat_out16 = self.conv_out16(feat_cp8)      # 16x的output(尺寸应该是8X)
        # feat_out32 = self.conv_out32(feat_cp16)    # 32x的output(尺寸应该是16X)

        # 上采样到原图大小
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        # feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)

        if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8:
            return feat_out, feat_out16, feat_out_sp2, feat_out_sp4, feat_out_sp8
        
        if (not self.use_boundary_2) and self.use_boundary_4 and self.use_boundary_8:
            return feat_out, feat_out16, feat_out_sp4, feat_out_sp8

        if (not self.use_boundary_2) and (not self.use_boundary_4) and self.use_boundary_8:
            return feat_out, feat_out16, feat_out_sp8
        
        if (not self.use_boundary_2) and (not self.use_boundary_4) and (not self.use_boundary_8):
            return feat_out, feat_out16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

# 转换onnx模型时使用的网络结构(会删除训练阶段冗余的网络结构,提升推理速度)
class BiSeNet_delNet_infer(nn.Module):
    def __init__(self, backbone, n_classes, pretrain_model='', use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False, heat_map=False, *args, **kwargs):
        super(BiSeNet_delNet_infer, self).__init__()
        
        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        # self.heat_map = heat_map
        self.cp = ContextPath(backbone, pretrain_model, use_conv_last=use_conv_last)

        if backbone == 'STDCNet1446':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        elif backbone == 'STDCNet813':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes
        elif backbone == 'STDCNet813Del':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes
        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.ffm = FeatureFusionModule(inplane, 128)    # FFN融合模块
        self.conv_out = BiSeNetOutput(128, 128, n_classes)          # 256 --> 128  mgchen
        self.conv_out16 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)
        # self.conv_out32 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)

        self.conv_out_sp16 = BiSeNetOutput(sp16_inplanes, 64, 1)
        self.conv_out_sp8 = BiSeNetOutput(sp8_inplanes, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(sp4_inplanes, 64, 1)      # 4X的热力图
        self.conv_out_sp2 = BiSeNetOutput(sp2_inplanes, 64, 1)      # 2X的热力图
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        
        # feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)
        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8 = self.cp(x)

        # feat_out_sp2 = self.conv_out_sp2(feat_res2)     # 2X的热力图

        # feat_out_sp4 = self.conv_out_sp4(feat_res4)     # 4X的热力图
  
        # feat_out_sp8 = self.conv_out_sp8(feat_res8)     # 8X的热力图

        # feat_out_sp16 = self.conv_out_sp16(feat_res16)       # 16X的热力图

        feat_fuse = self.ffm(feat_res8, feat_cp8)   # 融合stage3(8X)和stage4上采样后 的特征图

        # 网络输出结果
        feat_out = self.conv_out(feat_fuse)               # 8X的输出output
        # feat_out16 = self.conv_out16(feat_cp8)      # 16x的output(尺寸应该是8X)
        # feat_out32 = self.conv_out32(feat_cp16)    # 32x的output(尺寸应该是16X)

        # 上采样到原图大小
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        # feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        # feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)

        if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8:
            return feat_out#, feat_out16, feat_out_sp2, feat_out_sp4, feat_out_sp8
        
        if (not self.use_boundary_2) and self.use_boundary_4 and self.use_boundary_8:
            return feat_out#, feat_out16, feat_out_sp4, feat_out_sp8

        if (not self.use_boundary_2) and (not self.use_boundary_4) and self.use_boundary_8:
            return feat_out#, feat_out16, feat_out_sp8
        
        # if (not self.use_boundary_2) and (not self.use_boundary_4) and (not self.use_boundary_8):
        #     return feat_out, feat_out16

        # 极简模式
        if (not self.use_boundary_2) and (not self.use_boundary_4) and (not self.use_boundary_8):
            return feat_out    

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
    
    net = BiSeNet_cutNet('STDCNet813', 19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 768, 1536).cuda()
    out, out16, out32 = net(in_ten)
    print(out.shape)
    torch.save(net.state_dict(), 'STDCNet813.pth')

    
