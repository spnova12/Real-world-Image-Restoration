
import os
import numpy as np

import common.module_losses as module_losses

import collections, functools, operator  # dict 더할때 사용
# (https://www.geeksforgeeks.org/python-sum-list-of-dictionaries-with-same-key/)

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
# cudnn.benchmark allows you to enable the inbuilt cudnn auto-tuner
# to find the best algorithm to use for your hardware.
cudnn.benchmark = True


class TrainModule(object):

    def __init__(self, cuda_num, DataParallel=False, edge_lambda=1, color_lambda=1, sigma=4):
        # training 과정 중 관찰할 log 들을 저장할 dict 만들어주기.
        self.logs_dict = {}
        self.logs_dict_sum = {}
        self.logs_dict_average = {}

        if DataParallel:
            cuda_num = 0
        self.DataParallel = DataParallel
        self.device = torch.device(f'cuda:{cuda_num}')

        self.criterion_char = module_losses.CharbonnierLoss()
        self.criterion_edge = module_losses.EdgeLoss(cuda_num=cuda_num)

        self.G_optimizer = None

        self.edge_lambda = edge_lambda
        self.color_lambda = color_lambda

        self.netA = None

    def set_init_lr(self, init_lr):
        # 초기 learning rate 설정해주기.
        self.init_lr = init_lr

    def set_net(self, net_dict):
        # 사용할 딥러닝 모델 설정해주기.
        self.netG = net_dict['G']
        self.netG.train()

        if 'A_pre' in net_dict:
            self.netA = net_dict['A_pre']
            self.netA.eval()
            for param in self.netA.parameters():
                param.requires_grad = False

    def adjust_learning_rate(self, decay_rate):
        # Sets the learning rate to the initial learning rate decayed by 'decay_rate' every 'iter_lr_decay' iteration
        for param_group in self.G_optimizer.param_groups:
            befo = param_group['lr']
            param_group['lr'] = param_group['lr'] * decay_rate
            after = param_group['lr']

            print('\n===> learning rate(befo, after) : ', befo, after)

    def get_logs_dict(self):
        # dict 형태로 반환
        return self.logs_dict

    def sum_logs_dict(self):
        # logs 들을 아래 구문을 이용하여 self.logs_dict_sum 에 다 더해준다.
        # 아래 구문을 아용하면 두개의 dict 가 같은 key 끼리 더해진다.
        self.logs_dict_sum = dict(functools.reduce(operator.add, map(collections.Counter,
                                                                     [self.logs_dict_sum, self.logs_dict])))

    def get_logs_dict_average(self, interval):
        # 축적된 log들을 interval (설정한 iteration 주기) 나눠줌 으로서 log 들의 평균을 구해준다.
        for key, value in self.logs_dict_sum.items():
            self.logs_dict_average[key] = value/interval

        # 평균을 낸 후 sum 을 축적할 dict 를 초기화 해준다.
        self.logs_dict_sum = {}
        return self.logs_dict_average

    def weight_saver(self, filename, iter_count, best_psnr):
        state_dict = self.netG.state_dict()
        # DataParallel 를 사용했다면, module prefix 를 제거한 후 저장해주자.
        if isinstance(self.netG, nn.DataParallel):
            state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}

        state = {
            'iter_count': iter_count,
            'best_psnr': best_psnr,

            'G': state_dict,

            'G_optimizer': self.G_optimizer.state_dict(),
        }
        torch.save(state, filename)

    def weight_loader(self, load_checkpoint_dir, load_checkpoint_A_pre_dir=None):
        # 이 함수에서 checkpoint
        # checkpoint 를 load 해줌.
        print(f"\n===> Load checkpoint")
        optimizer = None
        if os.path.isfile(load_checkpoint_dir):
            print("netG : loading checkpoint '{}'".format(load_checkpoint_dir))

            checkpoint = torch.load(load_checkpoint_dir)

            iter_count = checkpoint['iter_count']
            best_psnr = checkpoint['best_psnr']

            self.netG.load_state_dict(checkpoint['G'])
            optimizer = checkpoint['G_optimizer']
        else:
            print("netG : no checkpoint found at '{}'".format(load_checkpoint_dir))
            iter_count = 0
            best_psnr = None

        # if use gt alignment net.
        if load_checkpoint_A_pre_dir is not None and self.netA is not None:
            if os.path.isfile(load_checkpoint_A_pre_dir):
                print("netA_pre : loading checkpoint '{}'".format(load_checkpoint_A_pre_dir))
                checkpoint = torch.load(load_checkpoint_A_pre_dir)
                self.netA.load_state_dict(checkpoint['G'])
            else:
                raise SystemExit(": no checkpoint found at '{}'".format(load_checkpoint_A_pre_dir))

        # Set Device
        # 딥러닝이 적용될 device 를 os.environ["CUDA_VISIBLE_DEVICES"] 에서 할당된 대로 설정해준다.
        # multi gpu 사용을 위해 아래 pytorch 공식 링크의 방식 사용.
        # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#sphx-glr-beginner-blitz-data-parallel-tutorial-py
        # visible 한 gpu 가 2개 이상이라면 multi gpu 를 사용하게 한다.
        # visible 한 gpu 의 개수는 main_train.py 에서
        # os.environ["CUDA_VISIBLE_DEVICES"] 를 통해서 관리한다.
        if torch.cuda.device_count() > 1 and self.DataParallel:
            print("\n===> Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.netG = nn.DataParallel(self.netG)

            if self.netA is not None:
                self.netA = nn.DataParallel(self.netA)

        self.netG.to(self.device)

        if self.netA is not None:
            self.netA.to(self.device)

        # Set Optimizer
        # 모델마다 사용할 optimizer 설정.
        # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
        # Parameters of a model after .cuda() will be different objects with those before the call.
        # (https://pytorch.org/docs/stable/optim.html#constructing-it)
        self.G_optimizer = torch.optim.Adam(self.netG.parameters(), lr=self.init_lr, betas=(0.9, 0.999))
        if optimizer is not None:
            self.G_optimizer.load_state_dict(optimizer)
            for param_group in self.G_optimizer.param_groups:
                print('\n===> learning rate :',  param_group['lr'])

        return iter_count, best_psnr


    def optimize_parameters(self, batch):
        ##
        input = batch['input_img'].to(self.device)
        median = batch['median_img'].to(self.device)
        target = batch['target_img'].to(self.device)

        ##
        if self.netA is not None:
            with torch.no_grad():
                target = self.netA(median, target).clone().detach()

        ##
        restored = self.netG(input)

        ##
        # Compute loss at each stage
        loss_char = torch.stack([self.criterion_char(restored[j],target) for j in range(len(restored))]).sum()
        loss_edge = torch.stack([self.criterion_edge(restored[j],target) for j in range(len(restored))]).sum()
        errG = (loss_char) + (0.05*loss_edge)

        ##
        self.G_optimizer.zero_grad()
        errG.backward()
        self.G_optimizer.step()

        self.logs_dict['loss_char'] = loss_char.item()
        self.logs_dict['loss_edge'] = loss_edge.item()









