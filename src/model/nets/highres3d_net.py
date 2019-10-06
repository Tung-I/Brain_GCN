import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.nets.base_net import BaseNet


class HighRes3DNet(BaseNet):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_block = _InBlock(in_channels, 8)
        self.res_block1 = nn.Sequential(_ResidualBlock(8, 8, dilation_rate=1),
                                        _ResidualBlock(8, 8, dilation_rate=1),
                                        _ResidualBlock(8, 16, dilation_rate=1))
        self.res_block2 = nn.Sequential(_ResidualBlock(16, 16, dilation_rate=2),
                                        _ResidualBlock(16, 16, dilation_rate=2),
                                        _ResidualBlock(16, 32, dilation_rate=2))
        self.res_block3 = nn.Sequential(_ResidualBlock(32, 32, dilation_rate=4),
                                        _ResidualBlock(32, 32, dilation_rate=4),
                                        _ResidualBlock(32, 64, dilation_rate=4))
        self.out_block = _OutBlock(64, out_channels)

    def forward(self, input):
        features = self.in_block(input)
        features = self.res_block1(features)
        features = self.res_block2(features)
        features = self.res_block3(features)
        features = self.out_block(features)
        output = F.softmax(features, dim=1)
        return output


class _InBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('conv1', nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        self.add_module('norm1', nn.BatchNorm3d(out_channels))
        self.add_module('active1', nn.ReLU(inplace=True))


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm_layer1 = nn.BatchNorm3d(in_channels)
        self.active_layer1 = nn.ReLU(inplace=True)
        self.conv_layer1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)

        self.norm_layer2 = nn.BatchNorm3d(out_channels)
        self.active_layer2 = nn.ReLU(inplace=True)
        self.conv_layer2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.projector = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, input):
        output = self.norm_layer1(input)
        output = self.active_layer1(output)
        output = self.conv_layer1(output)

        output = self.norm_layer2(output)
        output = self.active_layer2(output)
        output = self.conv_layer2(output)

        if self.projector:
            X = self.projector(input)
        else:
            X = input
        output = output + X
        return output


class _OutBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('deconv1', nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, 
                                                      padding=1, output_padding=1))
        self.add_module('norm1', nn.BatchNorm3d(in_channels))
        self.add_module('active1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(in_channels, out_channels, kernel_size=1))