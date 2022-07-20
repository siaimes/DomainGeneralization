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
from torch import optim

def sgd_step(network, epochs, lr, train_all, nesterov=False):
    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    # optimizer = optim.Adam(params, lr=lr, weight_decay=.0005)#, amsgrad=False)

    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d" % step_size)
    return optimizer, scheduler


def sgd_step2(network, epochs, lr, train_all, step_size, nesterov=False):
    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
    optimizer = optim.SGD(params, weight_decay=0.0005, momentum=0.9, nesterov=nesterov, lr=lr)

    # step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d" % step_size)
    return optimizer, scheduler


def sgd_cos(network, epochs, lr, train_all, weight_decay, nesterov=False):
    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
    optimizer = optim.SGD(params, weight_decay=weight_decay, momentum=.9, nesterov=nesterov, lr=lr)
    # optimizer = optim.Adam(params, lr=lr, weight_decay=.0005)#, amsgrad=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    return optimizer, scheduler

def sgd_warm(network, epochs, lr, train_all, nesterov=False):
    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
    # optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    optimizer = optim.Adam(params, lr=lr, weight_decay=.0005)#, amsgrad=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    return optimizer, scheduler
