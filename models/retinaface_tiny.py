import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net_tiny import TinyMobileNetV1
from models.net_tiny import TinySSH


class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3, phase = 'train'):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.phase = phase
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        # PyTorch uses dimension 0 for batch dimension in training
        if self.phase == 'train':
            return out.view(out.shape[0], -1, 2)
        else:
            # use out.shape to set reshape parameter causes problem when export to ONNX for convert to kmodel
            return out.view(1, -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3, phase = 'train'):
        super(BboxHead,self).__init__()
        self.phase = phase
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        # PyTorch uses dimension 0 for batch dimension in training
        if self.phase == 'train':
            return out.view(out.shape[0], -1, 4)
        else:
            # use out.shape to set reshape parameter causes problem when export to ONNX for convert to kmodel
            return out.view(1, -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3, phase = 'train'):
        super(LandmarkHead,self).__init__()
        self.phase = phase
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        # PyTorch uses dimension 0 for batch dimension in training
        if self.phase == 'train':
            return out.view(out.shape[0], -1, 10)
        else:
            # use out.shape to set reshape parameter causes problem when export to ONNX for convert to kmodel
            return out.view(1, -1, 10)

class TinyRetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(TinyRetinaFace,self).__init__()
        self.phase = phase
        backbone = TinyMobileNetV1()

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.ssh1 = TinySSH(in_channels_list[0], out_channels)
        self.ssh2 = TinySSH(in_channels_list[1], out_channels)
        self.ssh3 = TinySSH(in_channels_list[2], out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num,self.phase))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num,self.phase))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num,self.phase))
        return landmarkhead

    def forward(self,inputs):
        # MobileNetV1(Tiny)
        out = self.body(inputs)
        out_list = list(out.values())

        # no FPN

        # SSH(Tiny)
        feature1 = self.ssh1(out_list[0])
        feature2 = self.ssh2(out_list[1])
        feature3 = self.ssh3(out_list[2])
        features = [feature1, feature2, feature3]

        # Head
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output