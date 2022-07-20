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


def at(fms, **kwargs):
    ats = []
    for fm in fms:
        (N, C, H, W) = fm.shape
        ats.append(F.softmax(fm.reshape(N, C, -1), -1).mean(1))
    return ats

class Intra_ADR(nn.Module):
    def __init__(self, outp, step=4, stride=2, **kwargs):
        super(Intra_ADR, self).__init__()
        self.E_space = nn.Sequential(
            nn.ConvTranspose2d(outp, outp, 2, stride=stride, padding=0, output_padding=0,
                            bias=True, dilation=1, padding_mode='zeros'),
            nn.InstanceNorm2d(outp),
            nn.ReLU(inplace=True)
        )

    def cc_kth_p(self, input, kth=0):
        kth = input.size(1) // (input.size(2))
        input = torch.topk(input, kth, dim=1)[0]  # n,k,h,w

        input = input.mean(1, keepdim=True)
        return input

    def forward(self, x):
        branch = self.E_space(x)
        branch2 = branch
        x_adr = branch
        branch_ = branch.reshape(branch.size(0), branch.size(1), branch.size(2) * branch.size(3))
        branch = F.softmax(branch_, 2)
        branch_out = self.cc_kth_p(branch)
        return branch_out, branch2, x_adr

def Inter_ADR(t_cls_pred, cls_pred, t_fms, fms, label, device):
    t_mask = [(t_cls_pred[i] == label.data) * 1. for i in range(len(t_cls_pred))]
    t_mask = [t_mask[i].view(1, -1).permute(1, 0) for i in range(len(t_cls_pred))]
    mask = (cls_pred == label.data) * 1.
    mask = mask.view(1, -1).permute(1, 0)
    t_ats = [at(t_fms[i]) for i in range(len(t_cls_pred))]
    ats = at(fms)
    at_loss = 0
    for res_i in range(len(ats)):
        t_mask_dir = [t_mask[i].repeat(1, ats[res_i].size()[1]).to(device) for i in range(len(t_cls_pred))]
        mask_dir = mask.repeat(1, ats[res_i].size()[1]).to(device)

        t_mask_dvr = [1. - t_mask_dir[i] for i in range(len(t_cls_pred))]
        mask_dvr = 1. - mask_dir

        u_plus_temp = [t_ats[i][res_i].unsqueeze(2).contiguous() * t_mask_dir[i].unsqueeze(2).contiguous() for i in range(len(t_cls_pred))]
        u_plus_temp += [(ats[res_i].unsqueeze(2).contiguous() * mask_dir.unsqueeze(2).contiguous())]
        u_plus_temp = torch.cat(u_plus_temp, dim=2)
        u_plus = u_plus_temp.max(2)[0]

        u_minus_temp = [t_ats[i][res_i].unsqueeze(2).contiguous() * t_mask_dvr[i].unsqueeze(2).contiguous() for i in range(len(t_cls_pred))]
        u_minus_temp += [(ats[res_i].unsqueeze(2).contiguous() * mask_dvr.unsqueeze(2).contiguous())]
        u_minus_temp = torch.cat(u_minus_temp, dim=2)
        u_minus = u_minus_temp.max(2)[0]

        mask_plus_0 = torch.gt(u_plus, torch.zeros_like(u_plus)).to(device)
        mask_plus_1 = torch.gt(u_plus, ats[res_i]).to(device)

        mask_minus_0 = torch.gt(u_minus, torch.zeros_like(u_minus)).to(device)
        mask_minus_1 = torch.gt(u_minus, ats[res_i]).to(device)

        l2_dir = F.mse_loss(ats[res_i] * mask_plus_0 * mask_plus_1, u_plus * mask_plus_0 * mask_plus_1)
        l2_dvr = F.mse_loss(ats[res_i] * mask_minus_0 * mask_minus_1, u_minus * mask_minus_0 * mask_minus_1)
        at_loss = at_loss + 2 * l2_dir - l2_dvr

    return at_loss