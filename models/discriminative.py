"""
Discriminative Architectures for Transmitter Placement

Contains the four architectures used in the paper, all operating on
150×150 center-cropped building maps:
  - DeepXL_150:  UNet with symmetric encoder-decoder and skip connections
  - PMNet_150:   ResNet encoder-decoder with ASPP bottleneck
  - SIP2Net_150: PMNet backbone with ACNet asymmetric convolutions
  - DC-Net_150:  UNet with AOT (Aggregated Contextual Transformations) blocks

Factory function:
  create_model_deep(arch='deepxl_150') -> nn.Module

Paper name → code arch string:
  DeepXL  → 'deepxl_150'
  PMNet   → 'pmnet_150'
  SIP2Net → 'sip2net_150'
  DC-Net  → 'dcnet_150'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv_block(in_ch, out_ch):
    """
    Standard conv block with LocUNet-style improvements:
    - Uses LeakyReLU instead of ReLU
    - Conv3x3 -> BN -> LeakyReLU -> Conv3x3 -> BN -> LeakyReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),  # Changed from ReLU to LeakyReLU!
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True)   # Changed from ReLU to LeakyReLU!
    )



class TxLocatorDeepXL_150(nn.Module):
    """
    DeepXL for 160×160: 6 levels, 2×2 bottleneck, ~50M params
    160→80→40→20→10→5→2 (spatial)
    64→128→256→384→512→640→768 (channels)
    """
    def __init__(self, coord_method='soft_argmax', temperature=1.0, 
                 use_masking=True, img_size=150):
        super().__init__()
        self.coord_method = coord_method
        self.temperature = temperature
        self.use_masking = use_masking
        self.img_size = img_size
        
        # Encoder (6 levels)
        self.enc1 = conv_block(1, 64)
        self.pool1 = nn.AvgPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.AvgPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.AvgPool2d(2)
        self.enc4 = conv_block(256, 384)
        self.pool4 = nn.AvgPool2d(2)
        self.enc5 = conv_block(384, 512)
        self.pool5 = nn.AvgPool2d(2)
        self.enc6 = conv_block(512, 640)
        self.pool6 = nn.AvgPool2d(2)
        
        # Bottleneck
        self.bottleneck = conv_block(640, 768)
        
        # Decoder (6 levels)
        self.up6 = nn.ConvTranspose2d(768, 640, 2, stride=2)
        self.dec6 = conv_block(640 + 640, 640)
        self.up5 = nn.ConvTranspose2d(640, 512, 2, stride=2)
        self.dec5 = conv_block(512 + 512, 512)
        self.up4 = nn.ConvTranspose2d(512, 384, 2, stride=2)
        self.dec4 = conv_block(384 + 384, 384)
        self.up3 = nn.ConvTranspose2d(384, 256, 2, stride=2)
        self.dec3 = conv_block(256 + 256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(128 + 128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(64 + 64, 64)
        
        self.final = nn.Conv2d(64, 1, 1)

        with torch.no_grad():
            fan_in = self.final.weight.shape[1]  # No [0] needed!
            std = 10.0 / np.sqrt(fan_in)
            self.final.weight.normal_(0, std)
            self.final.bias.uniform_(-5.0, 5.0)

#         self.final = nn.Sequential(nn.Conv2d(64, 1, 1), nn.LeakyReLU(0.2, inplace=True))
    
#         with torch.no_grad():
#             fan_in = self.final[0].weight.shape[1]
#             std = 1.0 / np.sqrt(fan_in)  # Proper scaling
#             self.final[0].weight.normal_(0, std)
#             self.final[0].bias.uniform_(-1.0, 1.0)  # Random, not constant!
    
    def forward(self, x):
        input_h, input_w = x.size(2), x.size(3)
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))
        e6 = self.enc6(self.pool5(e5))
        b = self.bottleneck(self.pool6(e6))
        
        # Decoder with skip connections
        d6 = self.up6(b)
        if d6.shape[2:] != e6.shape[2:]:
            e6 = center_crop_to_match(e6, d6.shape[2:])
        d6 = self.dec6(torch.cat([d6, e6], dim=1))
        
        d5 = self.up5(d6)
        if d5.shape[2:] != e5.shape[2:]:
            e5 = center_crop_to_match(e5, d5.shape[2:])
        d5 = self.dec5(torch.cat([d5, e5], dim=1))
        
        d4 = self.up4(d5)
        if d4.shape[2:] != e4.shape[2:]:
            e4 = center_crop_to_match(e4, d4.shape[2:])
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            e3 = center_crop_to_match(e3, d3.shape[2:])
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            e2 = center_crop_to_match(e2, d2.shape[2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            e1 = center_crop_to_match(e1, d1.shape[2:])
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        logits = self.final(d1)
        if logits.size(2) != input_h or logits.size(3) != input_w:
            logits = F.interpolate(logits, size=(input_h, input_w), 
                             mode='bilinear', align_corners=False)
        building_mask = (x > 0.1) if self.use_masking else None
        
        if self.coord_method == 'soft_argmax':
            y, x_coord = soft_argmax(logits, building_mask, self.temperature, self.img_size)
        elif self.coord_method == 'hard_argmax':
            y, x_coord = hard_argmax(logits, building_mask, self.img_size)
        else:  # center_of_mass
            y, x_coord = center_of_mass(logits, building_mask, self.img_size)
        coords = torch.stack([y, x_coord], dim=1)
        

        return logits, coords




def convrelu_DCsingle(in_channels, out_channels, kernel=3, padding=1):
    """Conv + BatchNorm + ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def convreluT_DCsingle(in_channels, out_channels, output_padding=0):
    """ConvTranspose + BatchNorm + ReLU"""
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2,
                          padding=0, output_padding=output_padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class AOTBlock_DCsingle(nn.Module):
    """AOT Block with enhanced dilations and BatchNorm"""
    def __init__(self, dim, rates=[2, 4, 8, 16]):
        super(AOTBlock_DCsingle, self).__init__()
        self.rates = rates

        self.blocks = nn.ModuleList()
        for i, rate in enumerate(rates):
            self.blocks.append(nn.Sequential(
                nn.ReflectionPad2d(rate),
                nn.Conv2d(dim, dim // 4, kernel_size=3, padding=0, dilation=rate),
                nn.BatchNorm2d(dim // 4),
                nn.ReLU(inplace=True)
            ))

        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        outs = [block(x) for block in self.blocks]
        feat = torch.cat(outs, dim=1)
        feat = self.fuse(feat)
        gate = self.gate(feat)
        feat = feat * gate
        return x + feat




class TxLocator_DCNet_Single_150(nn.Module):
    """Single UNet DCNet for 150×150 - FULLY FIXED"""
    def __init__(self, coord_method='soft_argmax', temperature=1.0,
                 use_masking=True, img_size=150):
        super().__init__()
        self.coord_method = coord_method
        self.temperature = temperature
        self.use_masking = use_masking
        self.img_size = img_size

        # Encoder: 1 → 64 → 128 → 256 → 512
        self.conv0 = convrelu_DCsingle(1, 64)
        self.aot0 = AOTBlock_DCsingle(64, rates=[2, 4, 8, 16])
        self.pool0 = nn.MaxPool2d(2, stride=2)

        self.conv1 = convrelu_DCsingle(64, 128)
        self.aot1 = AOTBlock_DCsingle(128, rates=[2, 4, 8, 16])
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = convrelu_DCsingle(128, 256)
        self.aot2 = AOTBlock_DCsingle(256, rates=[2, 4, 8, 16])
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = convrelu_DCsingle(256, 512)
        self.aot3 = AOTBlock_DCsingle(512, rates=[2, 4, 8, 16])
        self.pool3 = nn.MaxPool2d(2, stride=2)

        # Decoder - FIXED channel counts!
        # Level 0: 9→18, concat with e3(512)
        self.up_conv0 = convreluT_DCsingle(512, 256, output_padding=0)
        self.dec0 = convrelu_DCsingle(256 + 512, 256)  # FIXED: 768 → 256

        # Level 1: 18→37, concat with e2(256)
        self.up_conv1 = convreluT_DCsingle(256, 128, output_padding=1)
        self.dec1 = convrelu_DCsingle(128 + 256, 128)  # FIXED: 384 → 128

        # Level 2: 37→75, concat with e1(128)
        self.up_conv2 = convreluT_DCsingle(128, 64, output_padding=1)
        self.dec2 = convrelu_DCsingle(64 + 128, 64)  # FIXED: 192 → 64

        # Level 3: 75→150, concat with e0(64)
        self.up_conv3 = convreluT_DCsingle(64, 64, output_padding=0)
        self.dec3 = convrelu_DCsingle(64 + 64, 64)  # FIXED: 128 → 64

        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        e0 = self.conv0(x)
        e0 = self.aot0(e0)
        e0_pooled = self.pool0(e0)  # 150→75

        e1 = self.conv1(e0_pooled)
        e1 = self.aot1(e1)
        e1_pooled = self.pool1(e1)  # 75→37

        e2 = self.conv2(e1_pooled)
        e2 = self.aot2(e2)
        e2_pooled = self.pool2(e2)  # 37→18

        e3 = self.conv3(e2_pooled)
        e3 = self.aot3(e3)
        e3_pooled = self.pool3(e3)  # 18→9

        # Decoder with skip connections
        d0 = self.up_conv0(e3_pooled)  # 9→18, 512→256
        d0 = torch.cat([d0, e3], dim=1)  # 256 + 512 = 768
        d0 = self.dec0(d0)  # 768→256

        d1 = self.up_conv1(d0)  # 18→37, 256→128
        d1 = torch.cat([d1, e2], dim=1)  # 128 + 256 = 384
        d1 = self.dec1(d1)  # 384→128

        d2 = self.up_conv2(d1)  # 37→75, 128→64
        d2 = torch.cat([d2, e1], dim=1)  # 64 + 128 = 192
        d2 = self.dec2(d2)  # 192→64

        d3 = self.up_conv3(d2)  # 75→150, 64→64
        d3 = torch.cat([d3, e0], dim=1)  # 64 + 64 = 128
        d3 = self.dec3(d3)  # 128→64

        heatmap = self.final(d3)

        # Coordinate extraction
        building_mask = (x > 0.1) if self.use_masking else None
        if self.coord_method == 'soft_argmax':
            y, x_coord = soft_argmax(heatmap, building_mask, self.temperature, self.img_size)
        elif self.coord_method == 'hard_argmax':
            y, x_coord = hard_argmax(heatmap, building_mask, self.img_size)
        else:
            y, x_coord = center_of_mass(heatmap, building_mask, self.img_size)
        coords = torch.stack([y, x_coord], dim=1)

        return heatmap, coords



    


class _ConvBnReLU_PM(nn.Sequential):
    """PMNet-style Conv-BN-ReLU block"""
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True):
        super(_ConvBnReLU_PM, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=1 - 0.999))
        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck_PM(nn.Module):
    """PMNet ResNet Bottleneck"""
    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck_PM, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU_PM(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU_PM(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU_PM(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU_PM(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else lambda x: x
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResLayer_PM(nn.Sequential):
    """PMNet ResNet Layer"""
    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer_PM, self).__init__()
        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck_PM(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


class _Stem_PM(nn.Sequential):
    """PMNet Stem - FIXED MaxPool"""
    def __init__(self, in_ch=1):
        super(_Stem_PM, self).__init__()
        self.add_module("conv1", _ConvBnReLU_PM(in_ch, 64, 7, 2, 3, 1, True))
        self.add_module("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


class _ASPPModule_PM(nn.Module):
    """ASPP module - FIXED: No inplace operations"""
    def __init__(self, in_ch, out_ch, pyramids):
        super(_ASPPModule_PM, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU_PM(in_ch, out_ch, 1, 1, 0, 1, True))
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU_PM(in_ch, out_ch, 3, 1, padding, dilation, True),
            )
        self.imagepool = nn.Sequential(
            OrderedDict([
                ("pool", nn.AdaptiveAvgPool2d((1, 1))),
                ("conv", _ConvBnReLU_PM(in_ch, out_ch, 1, 1, 0, 1, True)),
            ])
        )

    def forward(self, x):
        # FIXED: No inplace operations!
        h0 = self.stages.c0(x)
        h1 = self.stages.c1(x)
        h2 = self.stages.c2(x)
        h3 = self.stages.c3(x)
        h4 = F.interpolate(
            self.imagepool(x), size=x.shape[2:], mode="bilinear", align_corners=False
        )
        h = h0 + h1 + h2 + h3 + h4  # NO INPLACE
        return h


class ConRu_PM(nn.Module):
    """Conv + BN + ReLU"""
    def __init__(self, in_ch, out_ch):
        super(ConRu_PM, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_s1(x)
        x = self.bn_s1(x)
        return self.relu(x)


class ConRuT_PM(nn.Module):
    """ConvTranspose + BN + ReLU - LEARNABLE UPSAMPLING"""
    def __init__(self, in_ch, out_ch):
        super(ConRuT_PM, self).__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


# ==================== ACNET FOR SIP2NET ====================

class ACNet_SIP(nn.Module):
    """ACNet with proper padding for dilation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ACNet_SIP, self).__init__()

        pad_3x3 = dilation * (kernel_size // 2)
        pad_1x3_w = dilation
        pad_3x1_h = dilation

        self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=pad_3x3, dilation=dilation, bias=bias)
        self.conv_1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=stride,
                                    padding=(0, pad_1x3_w), dilation=(1, dilation), bias=bias)
        self.conv_3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride,
                                    padding=(pad_3x1_h, 0), dilation=(dilation, 1), bias=bias)

    def forward(self, x):
        out_3x3 = self.conv_3x3(x)
        out_1x3 = self.conv_1x3(x)
        out_3x1 = self.conv_3x1(x)

        # Safety check for size matching
        if out_3x3.shape[2:] != out_1x3.shape[2:] or out_3x3.shape[2:] != out_3x1.shape[2:]:
            target_size = out_3x3.shape[2:]
            if out_1x3.shape[2:] != target_size:
                out_1x3 = F.interpolate(out_1x3, size=target_size, mode='nearest')
            if out_3x1.shape[2:] != target_size:
                out_3x1 = F.interpolate(out_3x1, size=target_size, mode='nearest')

        return out_3x3 + out_1x3 + out_3x1




class TxLocator_PMNet_150(nn.Module):
    """
    PMNet for Tx Location - PROPERLY FIXED
    - Preserved ConvTranspose for learnable upsampling
    - F.interpolate only for size matching
    - No inplace operations in ASPP
    """
    def __init__(self, coord_method='soft_argmax', temperature=1.0,
                 use_masking=True, img_size=150, output_stride=8):
        super().__init__()
        self.coord_method = coord_method
        self.temperature = temperature
        self.use_masking = use_masking
        self.img_size = img_size

        if output_stride == 8:
            stride = [1, 2, 1, 1]
            dilation = [1, 1, 2, 2]
        elif output_stride == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]

        # Encoder
        self.layer1 = _Stem_PM(in_ch=1)
        self.layer2 = _ResLayer_PM(3, 64, 256, stride[0], dilation[0])
        self.layer3 = _ResLayer_PM(4, 256, 512, stride[1], dilation[1])
        self.layer4 = _ResLayer_PM(6, 512, 512, stride[2], dilation[2])
        self.layer5 = _ResLayer_PM(3, 512, 1024, stride[3], dilation[3], [1, 2, 1])

        # ASPP (no inplace)
        self.aspp = _ASPPModule_PM(1024, 256, [6, 12, 18])
        self.fc1 = _ConvBnReLU_PM(256, 512, 1, 1, 0, 1, True)

        # Decoder - KEEP ConvTranspose!
        self.conv_up5 = ConRu_PM(512, 512)
        self.conv_up4 = ConRu_PM(512 + 1024, 512)
        self.conv_up3 = ConRuT_PM(512 + 512, 256)  # ConvTranspose! Learnable 2× upsampling
        self.conv_up2 = ConRu_PM(256 + 256, 256)
        self.conv_up1 = ConRu_PM(256 + 64, 256)
        self.conv_up0 = ConRu_PM(256, 128)

        # Final output
        self.conv_up00 = nn.Sequential(
            nn.Conv2d(129, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        input_x = x
        input_h, input_w = x.size(2), x.size(3)

        # Encoder
        x1 = self.layer1(x)     # 38×38
        x2 = self.layer2(x1)    # 38×38
        x3 = self.layer3(x2)    # 19×19
        x4 = self.layer4(x3)    # 19×19
        x5 = self.layer5(x4)    # 19×19

        # ASPP
        feat = self.aspp(x5)
        feat = self.fc1(feat)   # 19×19, 512 ch

        # Decoder
        feat = self.conv_up5(feat)
        feat = torch.cat([feat, x5], dim=1)

        feat = self.conv_up4(feat)
        feat = torch.cat([feat, x4], dim=1)

        # LEARNABLE upsampling via ConvTranspose
        feat = self.conv_up3(feat)  # 19×19 → 38×38 (learnable!)

        # Match x2 size if needed (e.g., 38 vs 37)
        if feat.shape[2:] != x2.shape[2:]:
            feat = F.interpolate(feat, size=x2.shape[2:], mode='bilinear', align_corners=False)
        feat = torch.cat([feat, x2], dim=1)

        feat = self.conv_up2(feat)

        # Match x1 size if needed
        if feat.shape[2:] != x1.shape[2:]:
            feat = F.interpolate(feat, size=x1.shape[2:], mode='bilinear', align_corners=False)
        feat = torch.cat([feat, x1], dim=1)

        feat = self.conv_up1(feat)
        feat = self.conv_up0(feat)

        # Final upsampling to input size (fixed interpolation)
        if feat.shape[2:] != (input_h, input_w):
            feat = F.interpolate(feat, size=(input_h, input_w), mode='bilinear', align_corners=False)
        feat = torch.cat([feat, input_x], dim=1)

        heatmap = self.conv_up00(feat)

        # Coordinate extraction
        building_mask = (x > 0.1) if self.use_masking else None
        if self.coord_method == 'soft_argmax':
            y, x_coord = soft_argmax(heatmap, building_mask, self.temperature, self.img_size)
        elif self.coord_method == 'hard_argmax':
            y, x_coord = hard_argmax(heatmap, building_mask, self.img_size)
        else:
            y, x_coord = center_of_mass(heatmap, building_mask, self.img_size)
        coords = torch.stack([y, x_coord], dim=1)

        return heatmap, coords


# ==================== SIP2NET PROPERLY FIXED ====================



class TxLocator_SIP2Net_150(nn.Module):
    """
    SIP2Net for Tx Location - PROPERLY FIXED
    - Preserved ConvTranspose for learnable upsampling
    - Fixed ACNet padding
    - No inplace operations in ASPP
    """
    def __init__(self, coord_method='soft_argmax', temperature=1.0,
                 use_masking=True, img_size=150, output_stride=8):
        super().__init__()
        self.coord_method = coord_method
        self.temperature = temperature
        self.use_masking = use_masking
        self.img_size = img_size

        if output_stride == 8:
            stride = [1, 2, 1, 1]
            dilation = [1, 1, 2, 2]
        elif output_stride == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]

        # Encoder with ACNet
        self.layer1 = _Stem_PM(in_ch=1)
        self.layer2 = _ResLayer_SIP(3, 64, 256, stride[0], dilation[0], use_acnet=False)
        self.reduce = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            _BATCH_NORM(256, eps=1e-5, momentum=1 - 0.999),
            nn.ReLU()
        )
        self.layer3 = _ResLayer_SIP(4, 256, 512, stride[1], dilation[1], use_acnet=True)
        self.layer4 = _ResLayer_SIP(6, 512, 512, stride[2], dilation[2], use_acnet=True)
        self.layer5 = _ResLayer_SIP(3, 512, 1024, stride[3], dilation[3], [1, 2, 1], use_acnet=True)

        # ASPP
        self.aspp = _ASPPModule_PM(1024, 256, [2, 4, 6])
        self.fc1 = _ConvBnReLU_PM(256, 512, 1, 1, 0, 1, True)

        # Decoder - KEEP ConvTranspose!
        self.conv_up5 = ConRu_PM(512, 512)
        self.conv_up4 = ConRu_PM(512 + 1024, 512)
        self.conv_up3 = ConRuT_PM(512 + 512, 256)  # ConvTranspose! Learnable upsampling
        self.conv_up2 = ConRu_PM(256 + 256, 256)
        self.conv_up1 = ConRu_PM(256 + 256, 256)
        self.conv_up0 = ConRu_PM(256 + 64, 128)

        # Final with ACNet
        self.conv_up00 = nn.Sequential(
            ACNet_SIP(129, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ACNet_SIP(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        input_x = x
        input_h, input_w = x.size(2), x.size(3)

        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x2_reduced = self.reduce(x2)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        # ASPP
        feat = self.aspp(x5)
        feat = self.fc1(feat)

        # Decoder
        feat = self.conv_up5(feat)
        feat = torch.cat([feat, x5], dim=1)

        feat = self.conv_up4(feat)
        feat = torch.cat([feat, x4], dim=1)

        # LEARNABLE upsampling
        feat = self.conv_up3(feat)

        # Match x2_reduced size
        if feat.shape[2:] != x2_reduced.shape[2:]:
            feat = F.interpolate(feat, size=x2_reduced.shape[2:], mode='bilinear', align_corners=False)
        feat = torch.cat([feat, x2_reduced], dim=1)

        feat = self.conv_up2(feat)

        # Match x2 size
        if feat.shape[2:] != x2.shape[2:]:
            feat = F.interpolate(feat, size=x2.shape[2:], mode='bilinear', align_corners=False)
        feat = torch.cat([feat, x2], dim=1)

        feat = self.conv_up1(feat)

        # Match x1 size
        if feat.shape[2:] != x1.shape[2:]:
            feat = F.interpolate(feat, size=x1.shape[2:], mode='bilinear', align_corners=False)
        feat = torch.cat([feat, x1], dim=1)

        feat = self.conv_up0(feat)

        # Final upsampling
        if feat.shape[2:] != (input_h, input_w):
            feat = F.interpolate(feat, size=(input_h, input_w), mode='bilinear', align_corners=False)
        feat = torch.cat([feat, input_x], dim=1)

        heatmap = self.conv_up00(feat)

        # Coordinate extraction
        building_mask = (x > 0.1) if self.use_masking else None
        if self.coord_method == 'soft_argmax':
            y, x_coord = soft_argmax(heatmap, building_mask, self.temperature, self.img_size)
        elif self.coord_method == 'hard_argmax':
            y, x_coord = hard_argmax(heatmap, building_mask, self.img_size)
        else:
            y, x_coord = center_of_mass(heatmap, building_mask, self.img_size)
        coords = torch.stack([y, x_coord], dim=1)

        return heatmap, coords


# ==================== DCNET PROPERLY FIXED ====================



def create_model_deep(arch='deepxl_150', coord_method='soft_argmax', 
                      temperature=1.0, use_masking=True, img_size=150):
    """
    Factory function to create model architectures.
    
    Args:
        arch: Architecture name. One of:
            'deepxl_150'  — UNet encoder-decoder (DeepXL in paper)
            'pmnet_150'   — ResNet + ASPP (PMNet in paper)
            'sip2net_150' — PMNet + ACNet (SIP2Net in paper)
            'dcnet_150'   — UNet + AOT blocks (DC-Net in paper)
        coord_method: 'soft_argmax' or 'center_of_mass'
        temperature: Temperature for soft-argmax
        use_masking: Mask building pixels in output
        img_size: Spatial dimension of input (default 150 for center crop)
    
    Returns:
        model: PyTorch nn.Module
    """
    arch = arch.lower()
    
    models = {
        'deepxl_150': TxLocatorDeepXL_150,
        'pmnet_150': TxLocator_PMNet_150,
        'sip2net_150': TxLocator_SIP2Net_150,
        'dcnet_150': TxLocator_DCNet_Single_150,
    }
    
    if arch not in models:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {list(models.keys())}")
    
    model = models[arch](
        coord_method=coord_method,
        temperature=temperature,
        use_masking=use_masking,
        img_size=img_size
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {arch.upper()} created (img_size={img_size}):")
    print(f"  Parameters: {n_params:,}")
    print(f"  Method: {coord_method}")
    print(f"  Masking: {use_masking}")
    print(f"  Temperature: {temperature}")
    
    return model


# ==================== SHARED BUILDING BLOCKS ====================

from collections import OrderedDict

try:
    from encoding.nn import SyncBatchNorm
    _BATCH_NORM = SyncBatchNorm
except ImportError:
    _BATCH_NORM = nn.BatchNorm2d

_BOTTLENECK_EXPANSION = 4

