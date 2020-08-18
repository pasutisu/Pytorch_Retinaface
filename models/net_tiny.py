import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

# MobileNetV1(Tiny)
# Ultra-Light-Fast-Generic-Face-Detector-1MB uses this model.
# https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
class TinyMobileNetV1(nn.Module):
    def __init__(self):
        super(TinyMobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 16, 2),
            conv_dw(16, 32, 1),
            conv_dw(32, 32, 2),
            conv_dw(32, 32, 1),
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
            conv_dw(64, 64, 1),
            conv_dw(64, 64, 1),
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

# SSH(Tiny) - 7x7 filter removed SSH
# input -> conv3x3 -> relu -> conv3x3 -> concat -> relu -> output
#       -> conv3x3 -------------------->
# Single Stage Headless Face Detector: https://arxiv.org/abs/1708.03979
class TinySSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TinySSH, self).__init__()
        assert out_channel % 6 == 0
        
        self.conv3X3 = conv_bn_no_relu(in_channel, 4*out_channel//6, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, 2*out_channel//6, stride=1)
        self.conv5X5_2 = conv_bn_no_relu(2*out_channel//6, 2*out_channel//6, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        out = torch.cat([conv3X3, conv5X5], dim=1)
        out = F.relu(out)
        return out
