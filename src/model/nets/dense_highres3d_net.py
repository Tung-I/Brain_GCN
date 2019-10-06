import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.nets.base_net import BaseNet


class DenseHighRes3DNet(BaseNet):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = nn.Sequential(nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
                                     nn.GroupNorm(num_groups=4, num_channels=8),
                                     nn.PReLU(num_parameters=8),
                                     DDC(downscale_factor=2))
        self.bottleneck = nn.Sequential(nn.Conv3d(64, 8, kernel_size=1),
                                        nn.GroupNorm(num_groups=4, num_channels=8),
                                        nn.PReLU(num_parameters=8))

        self.resblocks1 = nn.Sequential(HighResBlock(in_channels=64, num_features=8, dilated_rates=[1, 1]),
                                        HighResBlock(in_channels=8, num_features=8, dilated_rates=[1, 1]),
                                        HighResBlock(in_channels=8, num_features=8, dilated_rates=[1, 1]))
        self.resblocks2 = nn.Sequential(HighResBlock(in_channels=8, num_features=16, dilated_rates=[1, 3]),
                                        HighResBlock(in_channels=16, num_features=16, dilated_rates=[1, 3]),
                                        HighResBlock(in_channels=16, num_features=16, dilated_rates=[1, 3]))
        self.resblocks3 = nn.Sequential(HighResBlock(in_channels=16, num_features=32, dilated_rates=[1, 3, 5]),
                                        HighResBlock(in_channels=32, num_features=32, dilated_rates=[1, 3, 5]),
                                        HighResBlock(in_channels=32, num_features=32, dilated_rates=[1, 3, 5]))

        self.decoder = nn.Sequential(nn.Conv3d(64, 64, kernel_size=1),
                                     DUC(upscale_factor=2),
                                     nn.Conv3d(8, out_channels, kernel_size=1))

    def forward(self, input):
        features1 = self.encoder(input)
        features2 = self.resblocks1(features1)
        features3 = self.resblocks2(features2)
        features4 = self.resblocks3(features3)
        features1 = self.bottleneck(features1)
        features = torch.cat([features1, features2, features3, features4], dim=1)
        output = F.softmax(self.decoder(features), dim=1)
        return output


class DDC(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        n, c, d, h, w = input.size()
        r = self.downscale_factor
        out_c = c * (r**3)
        out_d = d // r
        out_h = h // r
        out_w = w // r

        input_view = input.contiguous().view(n, c, out_d, r, out_h, r, out_w, r)
        output = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous().view(n, out_c, out_d, out_h, out_w)
        return output


class DUC(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n, c, d, h, w = input.size()
        r = self.upscale_factor
        out_c = c // (r**3)
        out_d = d * r
        out_h = h * r
        out_w = w * r

        input_view = input.contiguous().view(n, out_c, r, r, r, d, h, w)
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous().view(n, out_c, out_d, out_h, out_w)
        return output


class HighResBlock(nn.Module):
    def __init__(self, in_channels, num_features, dilated_rates):
        super().__init__()
        self.conv = nn.Sequential()
        for i, dilated_rate in enumerate(dilated_rates):
            if i == 0:
                self.conv.add_module(f'norm{i}', nn.GroupNorm(num_groups=4, num_channels=in_channels))
                self.conv.add_module(f'acti{i}', nn.PReLU(num_parameters=in_channels))
                self.conv.add_module(f'conv{i}', nn.Conv3d(in_channels, num_features,
                                                           kernel_size=3, padding=dilated_rate, dilation=dilated_rate))
            else:
                self.conv.add_module(f'norm{i}', nn.GroupNorm(num_groups=4, num_channels=num_features))
                self.conv.add_module(f'acti{i}', nn.PReLU(num_parameters=num_features))
                self.conv.add_module(f'conv{i}', nn.Conv3d(num_features, num_features,
                                                           kernel_size=3, padding=dilated_rate, dilation=dilated_rate))
        self.projector = nn.Conv3d(in_channels, num_features, kernel_size=1) if in_channels != num_features else None

    def forward(self, input):
        features = self.conv(input)
        if self.projector:
            output = self.projector(input) + features
        else:
            output = input + features
        return output
