from torchvision import models
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Swinv2Model
from transformers import logging
logging.set_verbosity_error()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class ConvLayer(nn.Module):
    def __init__(self, inputfeatures, outputinter, kernel_size=7, stride=1, padding=3, dilation=1, output=64, layertype=1, droupout=False):
        super(ConvLayer, self).__init__()
        if droupout == False:
            self.layer1 = nn.Sequential(
                nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.BatchNorm2d(outputinter),
                nn.PReLU(num_parameters=1, init=0.25))
            self.layer2 = nn.Sequential(
                nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.BatchNorm2d(outputinter),
                nn.PReLU(num_parameters=1, init=0.25))
            self.layer3 = nn.Sequential(
                nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.BatchNorm2d(output),
                nn.PReLU(num_parameters=1, init=0.25))
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.BatchNorm2d(outputinter),
                nn.Dropout(p=0.30),
                nn.PReLU(num_parameters=1, init=0.25))
            self.layer2 = nn.Sequential(
                nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.BatchNorm2d(outputinter),
                nn.Dropout(p=0.30),
                nn.PReLU(num_parameters=1, init=0.25))
            self.layer3 = nn.Sequential(
                nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.BatchNorm2d(output),
                nn.Dropout(p=0.30),
                nn.PReLU(num_parameters=1, init=0.25))

        self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.layer5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.layertype = layertype

    def forward(self, x):
        out1 = self.layer1(x)
        if self.layertype == 1:
            out1 = self.layer3(out1)
            out1, inds = self.layer4(out1)
            return out1, inds
        elif self.layertype == 2:
            out1 = self.layer2(out1)
            out1 = self.layer3(out1)
            out1, inds = self.layer4(out1)
            return out1, inds
        elif self.layertype == 3:
            out1 = self.layer3(out1)
            return out1
        elif self.layertype == 4:
            out1 = self.layer3(out1)
            out1 = self.layer5(out1)
            return out1

class ClassifyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ClassifyBlock, self).__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.layer(x)

class PSPhead(nn.Module):
    def __init__(self, input_dim=1536, output_dims=384, final_output_dims=1536, pool_scales=[1,2,3,6]):
        super(PSPhead, self).__init__()
        self.ppm_modules = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool),
                nn.Conv2d(input_dim, output_dims, kernel_size=1),
                nn.BatchNorm2d(output_dims),
                nn.PReLU(num_parameters=1, init=0.25)
            )
            for pool in pool_scales
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(input_dim + output_dims*len(pool_scales), final_output_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_output_dims),
            nn.PReLU(num_parameters=1, init=0.25)
        )

    def forward(self, x):
        x = x.permute((0,3,1,2))
        ppm_outs = [x]
        for ppm in self.ppm_modules:
            ppm_out = F.interpolate(ppm(x), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            ppm_outs.append(ppm_out)
        ppm_outs = torch.cat(ppm_outs, dim=1)
        return self.bottleneck(ppm_outs)

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[384, 768, 1536, 1536], fpn_out=768):
        super(FPN_fuse, self).__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, fpn_out, kernel_size=1)
            for in_ch in feature_channels
        ])

        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)
            for _ in range(len(feature_channels) - 1)
        ])

        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out*2),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        features = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        P = [features[-1]]

        for i in reversed(range(len(features) - 1)):
            up = F.interpolate(P[-1], size=features[i].shape[2:], mode='bilinear', align_corners=True)
            fused = up + features[i]
            fused = self.smooth_convs[i](fused)
            P.append(fused)

        P = list(reversed(P))
        H, W = P[0].shape[2], P[0].shape[3]
        P = [P[0]] + [
            F.interpolate(p, size=(H, W), mode='bilinear', align_corners=True)
            for p in P[1:]
        ]

        x = torch.cat(P, dim=1)
        return self.conv_fusion(x)

class SwinUperNet(nn.Module):
    def __init__(self, num_classes=104):
        super(SwinUperNet, self).__init__()

        model_name = "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft"
        self.backbone = Swinv2Model.from_pretrained(model_name, ignore_mismatched_sizes=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.feature_channels = [384, 768, 1536, 1536]

        self.PPMhead = PSPhead(input_dim=1536, output_dims=384, final_output_dims=1536)
        self.FPN = FPN_fuse(self.feature_channels, fpn_out=768)
        self.head = ConvLayer(1536, 128, kernel_size=3, stride=1, padding=1, output=64, layertype=3, droupout=True)
        self.ClassifyBlock = ClassifyBlock(64, num_classes)

        self.PPMhead.apply(weights_init)
        self.FPN.apply(weights_init)
        self.head.apply(weights_init)
        self.ClassifyBlock.apply(weights_init)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        features = list(outputs.hidden_states[1:])

        for i in range(len(features)):
            h = int(np.sqrt(features[i].shape[1]))
            features[i] = features[i].view(features[i].shape[0], h, h, features[i].shape[2])
            if i != len(features) - 1:
                features[i] = features[i].permute(0,3,1,2)

        features[-1] = self.PPMhead(features[-1])

        x = self.FPN(features)
        x = self.head(x)
        x = F.interpolate(x, size=input_size, mode='bilinear')
        x = self.ClassifyBlock(x)

        return x