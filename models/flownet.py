import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_DECODER_CFG = {
    'small0.25':[
        # in, feat, rfeat, m
        [8, 64,  64, .25],
        [8, 72,  74, .5],
    ],
    'small0.75':[
        # in, feat, rfeat, m
        [24, 64,  64, .25],
        [16, 72,  74, .5],
    ]
    
}

_DECODER_CFG['small'] = _DECODER_CFG['small0.75']


class NetC(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.pyramids = layers

    def forward(self, x):
        feats = []
        for subnet in self.pyramids:
            x = subnet(x)
            feats.append(x)
        return feats[::-1]


class Upsample(nn.Module):
    """Upsampling module.

    Args:
        scale_factor (int): Scale factor of upsampling.
        channels (int): Number of channels of conv_transpose2d.
    """

    def __init__(self, scale_factor: int, channels: int) -> None:
        super().__init__()
        self.kernel_size = 2 * scale_factor - scale_factor % 2
        self.stride = scale_factor
        self.pad = math.ceil((scale_factor - 1) / 2.)
        self.channels = channels
        self.register_buffer('weight', self.bilinear_upsampling_filter())

    # caffe::BilinearFilter
    def bilinear_upsampling_filter(self) -> torch.Tensor:
        """Generate the weights for caffe::BilinearFilter.

        Returns:
            Tensor: The weights for caffe::BilinearFilter
        """
        f = math.ceil(self.kernel_size / 2.)
        c = (2 * f - 1 - f % 2) / 2. / f
        weight = torch.zeros(self.kernel_size**2)
        for i in range(self.kernel_size**2):
            x = i % self.kernel_size
            y = (i / self.kernel_size) % self.kernel_size
            weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return weight.view(1, 1, self.kernel_size,
                           self.kernel_size).repeat(self.channels, 1, 1, 1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward function for upsample.

        Args:
            data (Tensor): The input data.

        Returns:
            Tensor: The upsampled data.
        """
        return F.conv_transpose2d(
            data,
            self.weight,
            stride=self.stride,
            padding=self.pad,
            groups=self.channels)


class ConvBlock(nn.Sequential):
    """
    Stack (Conv+Act)
    """
    def __init__(self, in_ch, hidden_channels, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2
        self.in_ch = in_ch
        
        layers = []
        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, dilation=dilation),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            ])
            in_ch = out_ch
 
        super().__init__(*layers)
        self.out_ch = out_ch


class RefineBlock(nn.Module):
    def __init__(self, in_ch, hidden_channels=[64, 32], last_kernel_size=3, residual=True) -> None:
        super().__init__()
        self.residual = residual
        self.out_ch = hidden_channels[-1]

        self.conv_layers = ConvBlock(in_ch=in_ch, hidden_channels=hidden_channels)         
        self.pred_flow = nn.Conv2d(self.out_ch, 2, kernel_size=last_kernel_size, stride=1, padding=last_kernel_size // 2)

    def forward(self, feat, upflow):
        if upflow is None:
            upflow = torch.zeros_like(feat)[:, :2]
        else:
            feat = torch.cat((feat, upflow), dim=1)       
        feat = self.conv_layers(feat)
        res_flow = self.pred_flow(feat)
        if self.residual:
            res_flow = upflow + res_flow
        return res_flow


class Decoder(nn.Module):
    def __init__(self, in_ch, feat_ch, rfeat_ch, multiplier=1.) -> None:
        super().__init__()
        self.multiplier = multiplier

        self.feat_layer = nn.Sequential(
            nn.Conv2d(in_ch*2, feat_ch, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        ) if in_ch*2 != feat_ch else nn.Sequential()
        self.NetR = RefineBlock(rfeat_ch)

        self.upflow_layer = Upsample(scale_factor=2, channels=2)

    @staticmethod
    def _scale_img(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return F.interpolate(
            img, size=(h, w), mode='bilinear', align_corners=False)

    def forward(self, feat1, feat2, upflow):
        _feat = torch.cat((feat1, feat2), dim=1)
        _feat = self.feat_layer(_feat)
        flowR = self.NetR(_feat, upflow)
        upflow = self.upflow_layer(flowR)
        return flowR, upflow


class NetE(nn.Module):
    def __init__(self, cfgs, flow_div=20.) -> None:
        super().__init__()
        self.npyramid = len(cfgs)
        self.flow_div = flow_div

        self.blocks = nn.ModuleList()       
        for in_ch, feat_ch, rfeat_ch, multiplier in cfgs:
            multiplier *= self.flow_div
            self.blocks.append(
                Decoder(in_ch, feat_ch, rfeat_ch, multiplier))

    def forward(self, feat1, feat2):
        flow_pred = []
        
        upflow = None
        for i in range(self.npyramid):
            flow, upflow = self.blocks[i](feat1[i], feat2[i], upflow)
            flow_pred.append(flow)

        if self.training:
            return flow_pred
        return flow_pred[-1]


class FlowNetS(nn.Module):
    def __init__(self, pyramid_layers, cfgs=_DECODER_CFG['small']) -> None:
        super().__init__()
        self.npyramid = len(cfgs)

        self.encoder = NetC(pyramid_layers)
        self.decoder = NetE(cfgs)

    def extract_feat(self, imgs, inference):
        if inference:
            img1, img2 = imgs
        else:
            img1, img2 = imgs[:, :3], imgs[:, 3:]
        feats1, feats2 = self.encoder(img1), self.encoder(img2)
        return feats1, feats2
    
    def forward(self, imgs, inference=False):
        """
        Args:
            imgs (Tensor): [N, 2*C, H, W],The concatenated input images.
        """
        feats1, feats2 = self.extract_feat(imgs, inference)
        flow = self.decoder(feats1, feats2)
        res_dict = dict(
            flow=flow,
            features=[feats1[0], feats2[0]]
        )
        return res_dict
