from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

import math


class DUNet(nn.Module):
    def __init__(self, in_channel, n_c, n_b1, n_b2):
        super(DUNet, self).__init__()
        self.unet1 = Unet(in_channel, n_c, n_b1, kernel_size=3, phase=3)
        self.fuse = default_conv(n_c * 3, n_c, kernel_size=3, padding=1, bias=True)
        self.unet2 = DRN(in_channel, n_c, n_b2, kernel_size=3, phase=3)

        self.conv_tail = default_conv(n_c, in_channel, kernel_size=3, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):

        b, c, n, h, w = x.size()
        x1 = x[:, :, 0:3, :, :]
        x2 = x[:, :, 1:4, :, :]
        x3 = x[:, :, 2:5, :, :]

        x1 = x1.reshape(b, -1, h, w)
        x2 = x2.reshape(b, -1, h, w)
        x3 = x3.reshape(b, -1, h, w)

        x1 = self.lrelu(self.unet1(x1))
        x2 = self.unet1(x2)
        x3 = self.lrelu(self.unet1(x3))

        y = torch.cat((x1, x2, x3), dim=1)

        y = self.fuse(y)

        y = self.conv_tail(self.unet2(y))

        return y

def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor
    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class Unet(nn.Module):
    def __init__(self, in_channel, n_feats, n_blocks, kernel_size=3, phase=None):
        super(Unet, self).__init__()

        self.conv_head = nn.Sequential(
            default_conv(in_channel * 3, n_feats, kernel_size, 1, True),
            nn.ReLU(inplace=True)
            )

        self.phase = phase
        act = nn.LeakyReLU(0.1, True)

        self.down = [
            DownBlock(
                scale=2, nFeat = n_feats * pow(2, p),
                in_channels=n_feats * pow(2, p),
                out_channels=n_feats * pow(2, p + 1)
            ) for p in range(self.phase)
        ]

        self.down = nn.ModuleList(self.down)

        up_body_blocks = [[
            RCAB(default_conv, n_feats * pow(2, p), kernel_size, act=act)
            for _ in range(n_blocks)
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks.insert(
            0, [
                RCAB(default_conv, n_feats * pow(2, self.phase), kernel_size, act=act)
                for _ in range(n_blocks)
            ]
        )

        # The fisrt upsample block
        up = [[
            Upsampler(default_conv, 2, n_feats * pow(2, self.phase), act=False),
            default_conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                Upsampler(default_conv, 2, 2 * n_feats * pow(2, p), act=False),
                default_conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        self.conv_tail = default_conv(n_feats * 2, n_feats, kernel_size, 1, True)


    def forward(self, x):

        x = self.conv_head(x)

        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        # up phases
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)

        x = self.conv_tail(x)

        return x

class DRN(nn.Module):
    def __init__(self, out_channel, n_feats, n_blocks, kernel_size=3, scale=4, phase=None):
        super(DRN, self).__init__()

        self.phase = phase
        act = nn.LeakyReLU(0.1, True)


        self.down = [
            DownBlock(
                scale=2, nFeat = n_feats * pow(2, p),
                in_channels=n_feats * pow(2, p),
                out_channels=n_feats * pow(2, p + 1)
            ) for p in range(self.phase)
        ]

        self.down = nn.ModuleList(self.down)

        up_body_blocks = [[
            RCAB(default_conv, n_feats * pow(2, p), kernel_size, act=act)
            for _ in range(n_blocks)
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks.insert(
            0, [
                RCAB(default_conv, n_feats * pow(2, self.phase), kernel_size, act=act)
                for _ in range(n_blocks)
            ]
        )

        # The fisrt upsample block
        up = [[
            Upsampler(default_conv, 2, n_feats * pow(2, self.phase), act=False),
            default_conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                Upsampler(default_conv, 2, 2 * n_feats * pow(2, p), act=False),
                default_conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # quality enhancement
        self.qe_begin = default_conv(n_feats * 2, n_feats, kernel_size, 1, True)

        qe_blocks = [
            RCAB(default_conv, n_feats, kernel_size, act=act)
            # RCAB_plus(default_conv, n_feats, kernel_size, act=act)
            for _ in range(n_blocks)
        ]

        self.qe_block = nn.Sequential(*qe_blocks)

    def forward(self, x):


        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        # up phases
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)

        x = self.qe_begin(x)
        head_qe = x
        x = self.qe_block(x)
        x = x + head_qe
        return x


########################
# Basic block
########################

def default_conv(in_channels, out_channels, kernel_size, padding=None, bias=False, init_sacle=0.1):
    if padding is None:
        padding = kernel_size // 2
    basic_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
    nn.init.kaiming_normal_(basic_conv.weight.data, a=0, mode='fan_in')
    basic_conv.weight.data *= init_sacle
    if basic_conv.bias is not None:
        basic_conv.bias.data.zero_()
    return basic_conv



class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class DownBlock(nn.Module):
    def __init__(self, scale, nFeat, in_channels, out_channels, negval=0.1):
        super(DownBlock, self).__init__()

        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True)
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=negval, inplace=True)
                )
            )

        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
