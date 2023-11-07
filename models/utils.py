import torch
from torch import nn, Tensor
import torch.nn.functional as F


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class FourierGenerator(nn.Sequential):
    def __init__(self, in_planes):
        super().__init__(
            nn.Conv2d(in_planes, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        # return self.relu(x + 3) / 6
        return self.relu(x + 3) * 0.16666666666666667


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.sigmoid = HardSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                HardSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEModuleConv(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(SEModuleConv, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
                nn.Conv2d(channel, _make_divisible(channel // reduction, 8), kernel_size=1, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(_make_divisible(channel // reduction, 8), channel, kernel_size=1, bias=bias),
                HardSigmoid()
        )

    def forward(self, x):
        # b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class FusedMBConv(nn.Module):
    """
    Implements Fused-MBConv from `"EfficientNetV2: Smaller Models and Faster Training"
    """
    def __init__(self, inp, oup, kernel_size, stride, use_se, use_hs, hidden_dim=0, expand_ratio=4,
                    se_bias=True, drop_rate=0.):
        super().__init__()
        assert stride in [1, 2]

        if hidden_dim <= 0:
            assert expand_ratio > 0
            hidden_dim = int(inp * expand_ratio)

        self.identity = stride == 1 and inp == oup
        # self.identity = False

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # conv kxk, no se
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                # conv kxk
                nn.Conv2d(inp, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, bias=False),
                nn.BatchNorm2d(hidden_dim),
                DropBlock2d(drop_rate, 3),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SEModuleConv(hidden_dim, bias=se_bias) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                DropBlock2d(drop_rate, 3),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Conv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_planes),
            HardSwish()
        )


class DropBlock2d(nn.Module):
    """
    Implements DropBlock2d from `"DropBlock: A regularization method for convolutional networks"
    <https://arxiv.org/abs/1810.12890>`.
    Args:
        p (float): Probability of an element to be dropped.
        block_size (int): Size of the block to drop.
        inplace (bool): If set to ``True``, will do this operation in-place. Default: ``False``
    """

    def __init__(self, p: float, block_size: int, inplace: bool = False) -> None:
        super().__init__()

        if p < 0.0 or p > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.block_size = block_size
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input feature map on which some areas will be randomly
                dropped.
        Returns:
            Tensor: The tensor after DropBlock layer.
        """
        if not self.training or self.p == 0:
            return input

        N, C, H, W = input.size()
        # compute the gamma of Bernoulli distribution
        gamma = (self.p * H * W) / ((self.block_size ** 2) * ((H - self.block_size + 1) * (W - self.block_size + 1)))
        mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=input.device))

        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.max_pool2d(mask, stride=(1, 1), kernel_size=(self.block_size, self.block_size), padding=self.block_size // 2)
        mask = 1 - mask
        normalize_scale = mask.numel() / (1e-6 + mask.sum())
        if self.inplace:
            input.mul_(mask * normalize_scale)
        else:
            input = input * mask * normalize_scale
        return input

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, block_size={self.block_size}, inplace={self.inplace})"
        return s

def set_drop_rate(p, module):
    if isinstance(module, DropBlock2d):
        module.p = p

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


class MBConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, use_se, use_hs, hidden_dim=0, expand_ratio=4, 
                    se_bias=True, drop_rate=0.):
        super().__init__()
        assert stride in [1, 2]

        if hidden_dim <= 0:
            assert expand_ratio > 0
            hidden_dim = int(inp * expand_ratio)

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                DropBlock2d(drop_rate, 3),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SEModuleConv(hidden_dim, bias=se_bias) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                DropBlock2d(drop_rate, 3),
            )
        else:
            self.conv = nn.Sequential(
                # pw channel expansion
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                DropBlock2d(drop_rate, 3),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                DropBlock2d(drop_rate, 3),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SEModuleConv(hidden_dim, bias=se_bias) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                DropBlock2d(drop_rate, 3),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MBConvBlock(nn.Module):
    def __init__(self, cfgs, inplanes, width_mult=1, min_planes=8):
        super().__init__()
        self.width_mult = width_mult
        self.min_planes = min_planes
        self.inplanes = inplanes

        layers = []
        for k, t, c, use_se, use_hs, s in cfgs:
            output_channel = _make_divisible(c * self.width_mult, self.min_planes)
            layers.append(MBConv(self.inplanes, output_channel, k, s, use_se, use_hs, hidden_dim=t))
            self.inplanes = output_channel

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
