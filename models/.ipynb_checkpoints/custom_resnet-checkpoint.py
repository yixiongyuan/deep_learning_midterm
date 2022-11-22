from __future__ import absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .channel_selection import channel_selection
from collections import OrderedDict


__all__ = ['resnet']

"""
preactivation resnet with basicblock design.
"""

# reduction for covolution layer
def conv2d(filter_in, filter_out, kernel_size, stride=1):
    if kernel_size == 3:
        return nn.Sequential(OrderedDict([
            ("conv", nn.Sequential(nn.Conv2d(filter_in, filter_in, kernel_size=3,
                                             stride=stride, padding=1, groups=filter_in),
                                   nn.Conv2d(filter_in, filter_out, kernel_size=1))),
            ("BN", nn.BatchNorm2d(filter_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))
    else:
        pad = (kernel_size - 1) // 2
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
            ("BN", nn.BatchNorm2d(filter_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

class FusionBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,multiscale=None):
        super(FusionBlock, self).__init__()
        
        self.conv1 = conv2d(inplanes, planes, kernel_size=3,stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = conv2d(planes, planes, kernel_size=3,stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
       
          
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.multiscale = multiscale
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
       
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        
        if self.multiscale is not None:
            fusion = self.multiscale(x)
            out+=fusion
            
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3,stride=stride,padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], planes, kernel_size=3, stride=1,padding=1, bias=False)
             
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)


        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual

        return out
    
    
class resnet(nn.Module):
    def __init__(self, depth=20, cfg=None):
        super(resnet, self).__init__()
        
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9
        block = BasicBlock

        if cfg is None:
            # Construct config variable.
            cfg = [[128, 256], [256,256]*(n-1),[256,512],[512,512]*(n-1)]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1,stride=1,bias=False)
        
        self.fusion_layer1 = self._make_layer(FusionBlock, 64, n,stride=1)
        self.fusion_layer2 = self._make_layer(FusionBlock, 128, n, stride=2)
        self.basic_layer3 = self._make_layer(BasicBlock, 256, n, cfg = cfg[0:2*n], stride=2)
        self.basic_layer4 = self._make_layer(BasicBlock, 512, n, cfg = cfg[2*n:4*n], stride=2)
        
        
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.select = channel_selection(512 * block.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(4)

        self.fc = nn.Linear(cfg[-1], 10)
    
        #paramter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg = None, stride=1,fusion=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )
        
        layers = []
        
        if block == FusionBlock:
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks): 
                multiscale  = conv2d(self.inplanes, planes * block.expansion, kernel_size=5)
                layers.append(block(self.inplanes, planes,multiscale = multiscale))
        else:
            layers.append(block(self.inplanes, planes, cfg[0:2], stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, cfg[2*i: 2*(i+1)]))
           

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.fusion_layer1(x)  # 32x32
        x = self.fusion_layer2(x)  # 16x16
        x = self.basic_layer3(x)  # 8x8
        x = self.basic_layer4(x)  # 4x4
        
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    

# def test():
#     net = resnet()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()
