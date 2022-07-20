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
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
import argparse
import os
import torch
from torch import nn
import torch.nn.functional as F
from data import data_pacs, data_oh, data_vlcs, data_dn
from models import model_factory
from models import resnet
from optimizer.optimizer_helper import *
import numpy as np
import shutil
import random
from utils.Logger import init_log
from datetime import datetime
import cv2
from models.ADR import *
from utils.build import *

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--target_idx", "-t", type=int, default=0, help="")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    '''----------------------data aug stuff---------------------------------'''
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.5, type=float, help="Chance of randomly greyscaling a tile")
    parser.add_argument("--limit_source", default=None, type=int, help=" ")
    parser.add_argument("--limit_target", default=None, type=int, help=" ")
    parser.add_argument("--learning_rate", "-l", type=float, default=.008, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=True, type=bool, help="Use nesterov")
    parser.add_argument("--s_adr", default=0, type=int,  help="Use adc")
    parser.add_argument("--t_adr", default=0, type=int,  help="Use adc")
    parser.add_argument("--w_intra", default=0.05, type=float,  help=" ")
    parser.add_argument("--loss_scale", default=0.005, type=float, help=" ")
    parser.add_argument("--weight_decay", default=0.0004, type=float,  help=" ")
    parser.add_argument("--drop", default=False, type=bool,  help=" ")
    parser.add_argument("--log_dir", default='.././exps', type=str,  help=" ")
    parser.add_argument("--name", default='log', type=str,  help=" ")
    parser.add_argument("--s_pretrained", default='/pretrain/resnet18.pth', type=str, help=" ")
    parser.add_argument("--warm_epochs", default=0, type=int, help="Use warm_epochs")
    parser.add_argument("--gpu", default='0', type=str, help=" ")
    return parser.parse_args()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        args.log_dir = os.path.join(args.log_dir, args.name)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        self.log_dir = args.log_dir
        log = init_log(args.log_dir, args.target)
        self._print = log.info
        self._print(args)

        # build domain-specific models and domain-aggregated model
        t_G_sp = [resnet.Featurer() for _ in range(len(args.single_model))]
        t_C_sp = [resnet.Classifier(512, num_classes=args.n_classes) for _ in range(len(args.single_model))]
        self.t_G_sp = [t_G_sp[i].to(device) for i in range(len(args.single_model))]
        self.t_C_sp = [t_C_sp[i].to(device) for i in range(len(args.single_model))]

        t_G, t_C = resnet.Featurer(), resnet.Classifier(512, num_classes=args.n_classes)
        self.t_G, self.t_C = t_G.to(device), t_C.to(device)
        s_G, s_C = resnet.Featurer(), resnet.Classifier(512, num_classes=args.n_classes)
        self.s_G, self.s_C = s_G.to(device), s_C.to(device)

        # data loader
        self.s_loader, self.source_loader, self.val_loader, self.target_loader = get_inter_loader(args)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}

        self.len_dataloader = len(self.source_loader)
        self._print("Dataset size: train %d, val %d, test %d" % (
            len(self.source_loader.dataset),
            len(self.val_loader.dataset),
            len(self.target_loader.dataset)))

        # optimizer and scheduler
        self.s_g_optimizer, self.s_g_scheduler = sgd_cos(self.s_G, args.epochs, args.learning_rate, args.train_all,
                                                         args.weight_decay, nesterov=args.nesterov)
        self.s_c_optimizer, self.s_c_scheduler = sgd_cos(self.s_C, args.epochs, args.learning_rate, args.train_all,
                                                         args.weight_decay * 0, nesterov=args.nesterov)
        self.s_c_optimizer_warm, _ = sgd_warm(self.s_C, args.epochs, args.learning_rate, args.train_all,
                                              nesterov=args.nesterov)
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None
        self.t_adr = args.t_adr
        self.s_adr = args.s_adr
        self.w_intra = args.w_intra
        self.loss_scale = args.loss_scale
        self.s_best_test = [0.] * (len(args.source) + 1)
        self.s_best_val = [0.] * (len(args.source) + 1)
        self.epochs = args.epochs
        self.warm_epochs = args.warm_epochs
        self.teacher_model = args.single_model
        self.student_model = args.student_model
        self.s_pretrained_addr = args.s_pretrained
        if self.s_pretrained_addr:
            try:
                self.s_pretrained_addr = os.path.dirname(os.getcwd()) + self.s_pretrained_addr
                load_params = torch.load(self.s_pretrained_addr)
                self.s_G.load_state_dict(load_params, strict=False)
                print('Load Pre-trained model from {}'.format(self.s_pretrained_addr))
            except FileNotFoundError:
                print("Pre-trained weights not found")

    def _do_student(self, epoch=None):
        # training fuction in one epoch
        criterion = nn.CrossEntropyLoss().to(self.device)
        self.s_G.train()
        self.s_C.train()
        cls_losses, adc_losses, correct = 0, 0, 0
        total = len(self.source_loader.dataset)
        loader = self.s_loader
        # simulate, divide and assemble
        for domain_id in range(len(loader)):
            for it, ((data, label, fname), _) in enumerate(loader[domain_id]):
                data, label = data.to(self.device), label.to(self.device)
                self.s_g_optimizer.zero_grad()
                self.s_c_optimizer.zero_grad()
                t_fms, t_cls_pred = [], []
                with torch.no_grad():
                    for model_id in range(len(self.t_G_sp)):
                        t_outputs_0, t_fms_0 = self.t_G_sp[model_id](data, label)
                        t_class_logit_0, _, _ = self.t_C_sp[model_id](t_outputs_0, self.t_adr)
                        _, t_cls_pred_0 = t_class_logit_0[1].max(dim=1)
                        t_fms.append(t_fms_0)
                        t_cls_pred.append(t_cls_pred_0)
                outputs, fms = self.s_G(data, label)
                class_logits, adc_out, fm = self.s_C(outputs, self.s_adr)
                _, cls_pred = class_logits[1].max(dim=1)
                at_loss = Inter_ADR(t_cls_pred, cls_pred, t_fms, fms, label, self.device)
                class_loss_adr = criterion(class_logits[0], label)
                class_loss_ce = criterion(class_logits[1], label)
                class_loss = (class_loss_ce + class_loss_adr) / 2
                if it % 3 == 0:
                    self.w_intra /= (epoch + 1)
                else:
                    self.w_intra = 0
                if self.s_adr == True:
                    adc_loss = 1.0 - torch.mean(torch.mean(adc_out[0], 2))
                    loss = self.loss_scale * (class_loss + self.w_intra * adc_loss + at_loss)
                else:
                    loss = class_loss + at_loss
                loss.backward()
                self.s_g_optimizer.step()
                self.s_c_optimizer.step()

        for it, ((data, label, fname), d_idx) in enumerate(self.source_loader):
            data, label, d_idx = data.to(self.device), label.to(self.device), d_idx.to(self.device)
            self.s_g_optimizer.zero_grad()
            self.s_c_optimizer.zero_grad()
            outputs, fms = self.s_G(data, label)
            class_logits, adc_out, fm = self.s_C(outputs, self.s_adr)
            _, cls_pred = class_logits[1].max(dim=1)
            class_loss_adr = criterion(class_logits[0], label)
            class_loss_ce = criterion(class_logits[1], label)
            class_loss = (class_loss_ce + class_loss_adr) / 2
            if it % 3 == 0:
                self.w_intra /= (epoch + 1)
            else:
                self.w_intra = 0
            if self.s_adr == True:
                adc_loss = 1.0 - torch.mean(torch.mean(adc_out[0], 2))
                loss = class_loss + self.w_intra * adc_loss
            else:
                adc_loss = torch.Tensor([0])
                loss = class_loss
            cls_losses += class_loss.item()
            adc_losses += adc_loss.item()
            correct += (cls_pred == label.data).sum()
            loss.backward()
            if epoch < self.warm_epochs:
                self.s_c_optimizer_warm.step()
            else:
                self.s_g_optimizer.step()
                self.s_c_optimizer.step()

        self._print('Eps:{}/{} | cls_loss={} | ad_loss={} | acc={} | BS:{} | Lr:{}'.format(
            epoch, self.args.epochs,
            round(cls_losses / total * label.size(0), 6),
            round(adc_losses / total * label.size(0), 6),
            round(100 * float(correct) / total, 3),
            data.size(0),
            self.s_c_scheduler.get_last_lr()[0]))
        self.s_G.eval()
        self.s_C.eval()
        # test
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct = self.do_test(self.s_G, self.s_C, loader, self.s_adr)
                class_acc = float(class_correct) / total
                self._print('{}_ACC is {}'.format(phase, round(100 * class_acc, 3)))

                self.s_results[phase][epoch] = class_acc
        # save checkpoints
        if self.s_results['test'][epoch] > self.s_best_test[self.args.target_idx]:
            self.s_best_test[self.args.target_idx] = self.s_results['test'][epoch]
            outfile = os.path.join(self.log_dir, 'best_{}.tar'.format(self.args.target))
            torch.save({'eps': epoch, 'test_acc': self.s_best_test[self.args.target_idx],
                        'g_state': self.s_G.state_dict(), 'c_state': self.s_C.state_dict()}, outfile)
        if self.s_results['val'][epoch] > self.s_best_val[self.args.target_idx]:
            self.s_best_val[self.args.target_idx] = self.s_results['test'][epoch]
            outfile = os.path.join(self.log_dir, 'val_{}.tar'.format(self.args.target))
            torch.save({'eps': epoch, 'test_acc': self.s_best_val[self.args.target_idx],
                        'g_state': self.s_G.state_dict(), 'c_state': self.s_C.state_dict()}, outfile)

        if epoch == self.epochs-1:
            last_outfile = os.path.join(self.log_dir, 'last_{}.tar'.format(self.args.target))
            torch.save({'eps': epoch, 'test_acc': self.s_results['test'][epoch], 'g_state': self.s_G.state_dict(),
                        'c_state': self.s_C.state_dict()}, last_outfile)

        self._print('-' * 10)

    def do_test(self, G, C, loader, adc):
        class_correct = 0
        for it, ((data, label, fname), _) in enumerate(loader):
            data, label = data.to(self.device), label.to(self.device)
            outputs, _ = G(data, label)
            class_logits, adc_out, _ = C(outputs, adc=adc)
            _, cls_pred = class_logits[1].max(dim=1)
            class_correct += torch.sum(cls_pred == label.data)
        return class_correct

    def do_training(self):
        self.s_results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        t_ckpt = [torch.load(self.teacher_model[i]) for i in range(len(self.teacher_model))]
        s_ckpt = torch.load(self.student_model)
        [self.t_G_sp[i].load_state_dict(t_ckpt[i]['g_state']) for i in range(len(self.teacher_model))]
        [self.t_C_sp[i].load_state_dict(t_ckpt[i]['c_state']) for i in range(len(self.teacher_model))]
        self.s_G.load_state_dict(s_ckpt['g_state'])
        self.s_C.load_state_dict(s_ckpt['c_state'])
        for ep in range(self.args.epochs):
            self._do_student(ep)
            self.s_g_scheduler.step()
            self.s_c_scheduler.step()

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args = get_inter_config(args)
    args.target = args.source[args.target_idx]
    args.source.remove(args.target)
    args.student_model = args.intra_model[args.target_idx]
    args.single_model.remove(args.single_model[args.target_idx])
    print("Target domain: {}".format(args.target))
    args.name = args.suffix + '/i2_adr'
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()