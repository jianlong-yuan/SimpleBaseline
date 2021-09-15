import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random, cv2
from torch.distributions.uniform import Uniform

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..config import cfg


@MODEL_REGISTRY.register(name='DeepLabV2')
class DeepLabV2(SegBaseModel):

    def __init__(self):
        super(DeepLabV2, self).__init__()
        if self.aux:
            self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], self.nclass)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], self.nclass)
        self.__setattr__('decoder', ['layer5', 'layer6'] if self.aux else ['layer6'])

    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.encoder(x)
        outputs = list()
        x = self.layer6(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.layer5(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)

        return tuple(outputs)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)



class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


