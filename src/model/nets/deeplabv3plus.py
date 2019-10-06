import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.nets.aspp import build_aspp
from src.model.nets.decoder import build_decoder
from src.model.nets.backbone import build_backbone
from src.model.nets.sync_batchnorm.batchnorm import SynchronizedBatchNorm3d

from src.model.nets.base_net import BaseNet


class DeepLabV3Plus(BaseNet):
    """
    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
    """
    def __init__(self, in_channels, out_channels, backbone_name='resnet3d', output_stride=16, sync_bn=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.backbone_name = backbone_name
        self.sync_bn = sync_bn
        self.output_stride = output_stride
            
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm3d
        else:
            BatchNorm = nn.BatchNorm3d
        
        self.backbone = build_backbone(self.backbone_name, self.output_stride, BatchNorm)
        self.aspp = build_aspp(self.backbone_name, self.output_stride, BatchNorm)
        self.decoder = build_decoder(self.out_channels, self.backbone_name, BatchNorm)
        
    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='trilinear', align_corners=True)
        x = F.softmax(x, dim=1)
        return x
    
    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm3d) \
                        or isinstance(m[1], nn.BatchNorm3d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm3d) \
                        or isinstance(m[1], nn.BatchNorm3d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
