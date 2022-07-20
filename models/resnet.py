# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
from torch import nn as nn
from torch.autograd import Variable
import numpy.random as npr
import numpy as np
import torch.nn.functional as F
import random
from torch.autograd import Function
import random
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributions.normal as normal
from models import ADR
class MixStyle(nn.Module):
    """MixStyle.

    Reference:
    Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        """
        Args:
        p (float): probability of using MixStyle.
        alpha (float): parameter of the Beta distribution.
        eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha

        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})'

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix


class softmax_nograd(Function):
    @staticmethod
    def forward(self, x):
        return F.softmax(x, 2)

    @staticmethod
    def backward(self, grad):
        out = grad
        return out, None


class Featurer(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], jigsaw_classes=1000, classes=100):
        self.inplanes = 64
        super(Featurer, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.MaxPool2d(7, stride=1)

        self.mixstyle = MixStyle(p=0.5, alpha=0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def ccmp(self, input, kernel_size, stride):
        input = input.permute(0, 3, 2, 1)
        input = F.max_pool2d(input, kernel_size, stride)
        input = input.permute(0, 3, 2, 1).contiguous()
        return input

    def ccap(self, input, kernel_size, stride):
        input = input.permute(0, 3, 2, 1)
        input = F.avg_pool2d(input, kernel_size, stride)
        input = input.permute(0, 3, 2, 1).contiguous()
        return input

    def bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x, epoch=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.mixstyle(x)
        fm1 = x
        x = self.layer2(x)
        # x = self.mixstyle(x)
        fm2 = x
        x = self.layer3(x)
        # x = self.mixstyle(x)
        fm3 = x

        x = self.layer4(x)
        fm4 = x
        return x, [fm1, fm2, fm3, fm4]


class Classifier(nn.Module):
    def __init__(self, outp, step=4, stride=2, num_classes=1000):
        super(Classifier, self).__init__()
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.MaxPool2d(7, stride=1)
        self.classifier = nn.Linear(outp, num_classes)
        self.classifier2 = nn.Linear(outp, num_classes)

        self.step = step
        self.stride = stride

        self.intra_adr = ADR.Intra_ADR(outp)
        self.conv_mu = Parameter(torch.randn(outp, 1))
        self.conv_sigma = Parameter(torch.zeros(outp, 1))
        self.conv_bias = Parameter(torch.zeros(outp))
        self.lamb = nn.Parameter(0.01 * torch.ones(1))
        nn.init.kaiming_uniform_(self.conv_mu)
        self.reset_parameters()

    def ccmp(self, input, kernel_size, stride):
        input = input.permute(0, 3, 2, 1)
        input = F.max_pool2d(input, kernel_size, stride)
        input = input.permute(0, 3, 2, 1).contiguous()
        return input

    def ccap(self, input, kernel_size, stride):
        input = input.permute(0, 3, 2, 1)
        input = F.avg_pool2d(input, kernel_size, stride)
        input = input.permute(0, 3, 2, 1).contiguous()
        return input

    def bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x, adc=0, epoch=None):
        branch_out, branch2, x_adr = self.intra_adr(x)
        x_ce = x
        fm = x
        b2_out = self.gmp(branch2)
        b2_out = b2_out.view(b2_out.size(0), -1)
        x_adr = self.gap(x_adr)
        x_ce = self.gap(x_ce)
        x_adr = x_adr.view(x_adr.size(0), -1)
        x_ce = x_ce.view(x_ce.size(0), -1)
        x_ce = self.classifier(x_ce)
        x_adr = self.classifier2(x_adr)

        return [x_adr, x_ce], [branch_out, b2_out], fm

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.normal_(0.0, 0.01)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # nn.init.constant_(m.bias, 0)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def featurer_50():
    return Featurer(block=Bottleneck, layers=[3, 4, 6, 3])


def featurer_101():
    return Featurer(block=Bottleneck, layers=[3, 4, 23, 3])


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def load_state_dict(pretrain_state_dict, network):
    tmp = torch.load(pretrain_state_dict)
    if 'state' in tmp.keys():
        pretrained_dict = tmp['state']
    else:
        pretrained_dict = tmp

    network_dict = network.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in network_dict and v.size() == network_dict[k].size()}

    network_dict.update(pretrained_dict)
    nn.load_state_dict(network_dict)