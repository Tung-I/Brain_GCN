import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.nets.base_net import BaseNet

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        d_diff = skipxdo.size(2) - out.size(2)
        h_diff = skipxdo.size(3) - out.size(3)
        w_diff = skipxdo.size(4) - out.size(4)
        out = F.pad(out, (w_diff // 2, w_diff - w_diff //2,
                          h_diff // 2, h_diff - h_diff //2,
                          d_diff // 2, d_diff - d_diff //2,))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(outChans)
        self.conv2 = nn.Conv3d(outChans, outChans, kernel_size=1)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        # convolve 32 down to 'outChans' channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = F.softmax(out, dim=1)

        return out


class VNet(BaseNet):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels, out_channels, elu=False):
        super(VNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.elu = elu
        self.in_tr = InputTransition(self.in_channels, 16, self.elu)
        self.down_tr32 = DownTransition(16, 1, self.elu)
        self.down_tr64 = DownTransition(32, 2, self.elu)
        self.down_tr128 = DownTransition(64, 3, self.elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, self.elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, self.elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, self.elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, self.elu)
        self.up_tr32 = UpTransition(64, 32, 1, self.elu)
        self.out_tr = OutputTransition(32, self.out_channels, self.elu)

    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out
