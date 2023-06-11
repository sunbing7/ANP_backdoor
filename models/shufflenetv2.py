import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
import torch
from .anp_batchnorm import *

def shufflenetv2(num_classes=10, pretrained=1, norm_layer=nn.BatchNorm2d, **kwargs):
    net = models.shufflenet_v2_x1_0(pretrained=pretrained)

    net.aux_logits = False

    if pretrained:
        for param in net.parameters():
            param.requires_grad = False

    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, 4096),
        nn.Linear(4096, num_classes)
    )

    # replace norm_layer
    children = list(net.children())
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
    net = nn.Sequential(*[*children[:2], *children[2:21], NoisyBatchNorm2d(1024), *children[22:]])

    return net

def test():
    net = shufflenetv2()

    summary(net, (3, 200, 200))
#test()