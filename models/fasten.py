import math
import torch
import torch.nn as nn
from functools import partial

from .flownet import FlowNetS
from .mobilenetv3 import MobileNetV3
from .utils import HardSigmoid, HardSwish


_ARCH_CFG = {
    'netS':[
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 1],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ],
    'netT':[
        # k, t, c, SE, HS, s 
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]
    
}


class Aggregator(nn.Module):
    def __init__(self, feat_encoder, nframes, channel=1, expansion=24):
        super().__init__()
        
        self.nframes = nframes
        nweights = nframes * channel
        self.attn = nn.Sequential(
            nn.Conv2d((nframes-1)*2, expansion, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(expansion, channel*expansion, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel*expansion, nweights, kernel_size=1),
            HardSigmoid(),
        )

        self.feat_encoder = feat_encoder

    def forward(self, frames, flows, flow_feat):
        """
        Args:
            frames (Tensor)     : [N*k, C, H, W]
            flows (Tensor)      : [N, 2*(k-1), H, W]
            flow_feat (Tensor)  : [N, C, H, W]
        """
        # weight attention
        weight = self.attn(flows) # [N, k, 1, 1] or [N, k*C, 1, 1]
        Nxk, C, H, W = frames.shape
        N = Nxk // self.nframes
        k = self.nframes
        if weight.shape[1] == k*C:
            weight = weight.view(N, k, C, 1, 1)
        else:
            weight = weight.view(N, k, 1, 1, 1) # default
        frames = frames.view(N, k, C, H, W)
        frames = torch.sum(weight*frames, dim=1)

        feat = self.feat_encoder(frames)
        return torch.cat([feat, flow_feat], dim=1)

    def iforward(self, frames, flows, flow_feat):
        """
        Args:
            frames (List)     : k* [1, C, H, W]
            flows (Tensor)      : [1, 2*(k-1), H, W]
            flow_feat (Tensor)  : [1, C, H, W]
        """
        # weight attention
        weight = self.attn(flows) # k

        N, C, H, W = frames[0].shape
        _frames = [f.view(N, 1, C, H*W) for f in frames]
        _frames = torch.cat(_frames, dim=1)
        _frames = weight * _frames
        
        _frames = torch.sum(_frames, dim=1)
        _frames = _frames.view(N, C, H, W)

        feat = self.feat_encoder(_frames)
        return torch.cat([feat, flow_feat], dim=1)


class Fasten(nn.Module):
    def __init__(self, arch, nframes, num_classes, pretrained=None,
                    drop_rate=0.2) -> None:
        super().__init__()

        self.nframes = nframes
        self.num_classes = num_classes
        self.phase = 'full'
            
        # network module
        if 'mobilenet' in arch:
            width_mult = float(arch.split(',')[1])
            _MobileNetV3 = partial(MobileNetV3, mode='small', width_mult=width_mult, num_classes=num_classes)
            
            layers = _MobileNetV3(cfgs=_ARCH_CFG['netS']).features
            pyramid_layers = nn.ModuleList([layers[:2], layers[2:4]])
            self.flownet = FlowNetS(pyramid_layers)

            self.netT = _MobileNetV3(cfgs=_ARCH_CFG['netT'], input_planes=2*(nframes-1))
            self.encoderT = self.netT.features
            oupT = self.netT.inplanes

            self.netS = _MobileNetV3(cfgs=_ARCH_CFG['netS'])
            self.encoderSS = self.netS.features[:4]
            encoderS = self.netS.features[4:]
            oupS = self.netS.inplanes
            output_channel = self.netS.output_channel
        self.aggregator = Aggregator(encoderS, nframes)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(oupT+oupS),
            nn.Linear(oupT+oupS, output_channel),
            nn.BatchNorm1d(output_channel),
            HardSwish(),
            nn.Dropout(drop_rate),
            nn.Linear(output_channel, num_classes)
        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_() 
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def load_pretrained(self, pretrained, reweight=True):
        if pretrained is not None:
            net_data = torch.load(pretrained, map_location='cpu')
            if reweight:
                dict_keys = list(net_data.keys())
                for k in dict_keys:
                    if 'classifier' in k:
                        net_data.pop(k)
            self.netS.load_state_dict(net_data, strict=False)
  
    def set_phase(self, phase):
        self.phase = phase
    
    def estimate_flows(self, imgs):
        flows = []
        for i in range(self.nframes - 1):
            imgpair = imgs[:, 3*i:3*i+6]
            res_dict = self.flownet(imgpair)
            flow = res_dict['flow'][-1] if isinstance(res_dict['flow'], list) else res_dict['flow']
            flows.append(flow)

        return torch.cat(flows, dim=1)
    
    def extract_feat(self, imgs):
        N, kxC, H, W = imgs.shape
        C = kxC // self.nframes
        imgs = imgs.view(N*self.nframes, C, H, W)
        frames = self.encoderSS(imgs)
        return frames
    
    def forward(self, imgs):
        """
        Args:
            imgs (Tensor): [N, k*C, H, W],The concatenated input images.
        """
        # predict optical flows from multi frames
        flows = []
        for i in range(self.nframes - 1):
            imgpair = imgs[:, 3*i:3*i+6]
            res_dict = self.flownet(imgpair)
            flow = res_dict['flow'][-1] if isinstance(res_dict['flow'], list) else res_dict['flow']
            flows.append(flow)
        flows = torch.cat(flows, dim=1)

        N, kxC, H, W = imgs.shape
        C = kxC // self.nframes
        imgs = imgs.view(N*self.nframes, C, H, W)
        frames = self.encoderSS(imgs)

        flow_feat = self.encoderT(flows)
        # fuse spatial & temporal feat
        feat = self.aggregator(frames, flows, flow_feat)
        feat = self.pool(feat).view(feat.size(0), -1)
        cls = self.classifier(feat)
        
        return cls

    def inference_flow(self, imgs):
        return self.flownet(imgs)
    
    def inference_full(self, *imgs):
        """
        Args:
            imgs (List): k, [1, 3, H, W],The concatenated input images.
         """

        # predict optical flows from multi frames
        _flows = []
        for imgpair in zip(imgs[:-1], imgs[1:]):
            _flows.append(self.flownet(imgpair, True)['flow'])
        flows = torch.cat(_flows, dim=1)
        # encoder, extract feature
        frames = []
        for _img in imgs:
            _frame = self.encoderSS(_img)
            frames.append(_frame)
        flow_feat = self.encoderT(flows)
        # fuse spatial&temporal feat
        feat = self.aggregator.iforward(frames, flows, flow_feat)
        # classifier
        feat = self.pool(feat).view(feat.size(0), -1)
        cls = self.classifier(feat)
        
        return cls
