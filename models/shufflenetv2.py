import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
import torch
from .anp_batchnorm import *
#'''
def shufflenetv2(num_classes=10, pretrained=1, norm_layer=nn.BatchNorm2d, **kwargs):
    '''
    net = models.shufflenet_v2_x1_0(pretrained=False)

    net.aux_logits = False

    if pretrained:
        for param in net.parameters():
            param.requires_grad = False
    '''
    net = ShuffleNetV2(num_classes=num_classes)
    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, 4096),
        nn.Linear(4096, num_classes)
    )



    return net
#'''

def shufflenet_reconstruct(ori_net):
    '''
    modules = list(ori_net.children())
    sub_modules = list(modules[-1])
    module0 = [modules[0]]
    module1 = modules[1:6]
    module2 = [sub_modules[0]]
    module3 = [sub_modules[1]]

    net = nn.Sequential(*[*module0, *module1, Avgpool2d_n(poolsize=7), Flatten(), *module2, *module3])

    # replace norm_layer
    '''
    children = list(ori_net.children())
    nchildren = []
    for c in children:
        if c.__class__.__name__ == 'Sequential':
            nchildren += list(c.children())
        else:
            nchildren.append(c)
    children = nchildren
    children.insert(-2, torch.nn.AvgPool2d(kernel_size=7))
    children.insert(-2, torch.nn.Flatten())
    # NoisyBatchNorm2d(24),
    net = nn.Sequential(*[*children[:1], NoisyBatchNorm2d(24), *children[2:21], NoisyBatchNorm2d(1024), *children[22:]])
    #net = nn.Sequential(*children)

    return net


class Relu(nn.Module):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):
        x = F.relu(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Avgpool2d_n(nn.Module):
    def __init__(self, poolsize=2):
        super(Avgpool2d_n, self).__init__()
        self.poolsize = poolsize
    def forward(self, x):
        x = F.avg_pool2d(x, self.poolsize)
        return x

'''
def test():
    net = shufflenetv2()

    summary(net, (3, 200, 200))
#test()
'''


# shufflenet implementation

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        if (self.stride == 1) and (inp != branch_features << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, inp {inp} and oup {oup} values. If stride == 1 then inp should be equal to oup // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        stages_repeats = [4, 8, 4],
        stages_out_channels = [24, 116, 232, 464, 1024],
        num_classes = 1000,
        inverted_residual = InvertedResidual,
    ) -> None:
        super().__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

