
from os import listdir
from os.path import join
import os

import math
import numpy as np
import torch.utils.data as data
import random
import cv2

from common.module_DB_manager import HumanForrestManager, get_sky, MedianImgs

import sys
import common.module_utils as utils

import torchvision.transforms as transforms
import torch
import torch.backends.cudnn as cudnn
# cudnn.benchmark allows you to enable the inbuilt cudnn auto-tuner
# to find the best algorithm to use for your hardware.
cudnn.benchmark = True


class DatasetForDataLoader(data.Dataset):
    """
    여러 데이터셋을 지정한 비율대로 섞어서 사용할 수 있다.
    가장 크기가 큰 데이터셋을 기준으로 이미지 수를 조정해준다.
    """
    def __init__(self, hf_DB_dir, noise_type, additional_info=None, median=False, noise_level=None):
        """
        :param img_dirs: [(input, target), (input, target) ... (input, target)]
        """
        # hf_DB_dir = '/hdd1/works/datasets/ssd2/human_and_forest/R_F_D_S_C'
        self.hf_DB = HumanForrestManager(hf_DB_dir, noise_type, show_details=False)
        self.ipsize = additional_info['input_patch_size']
        self.median = median

        # torchvision 에 있는 함수들은 될수있으면 그대로 가져다 사용하자.
        self.random_crop = transforms.RandomCrop(self.ipsize)
        # self.resize = transforms.Resize(self.tpsize//self.sfactor, Image.BICUBIC)
        self.totensor = transforms.ToTensor()  # [0,255] -> [0,1] 로 만들어줌.

        self.db_len = self.hf_DB.get_db_len()

        self.noise_level = noise_level

    def get_input_target_pairs(self, index, noise_level=None, noisy_num=None, median=False):
        input, target = self.hf_DB.get_input_target_pairs(index, noise_level, noisy_num, median=median)
        return input, target

    def make_noisy_and_new_gt(self, input, target, median=False, return_noise_and_median=False):
        if median:
            median_imgs = MedianImgs(input)
            my_median_img = median_imgs.get_median_result()

            my_json = os.path.splitext(input[0])[0] + '.json'
            drawImg = get_sky(my_json)

            # blur to sky label mask
            sd = 8
            truncate = 2
            radius = int(truncate * sd + 0.5)
            drawImg = cv2.GaussianBlur(drawImg * 255, (radius * 2 + 1, radius * 2 + 1), sd)
            drawImg = np.clip(drawImg, 0, 255)
            drawImg = drawImg / 255

            # target
            my_gt_img = cv2.imread(target)
            clouds = my_median_img * drawImg
            forground = my_gt_img * (1 - drawImg)
            new_gt_img = clouds + forground

            if return_noise_and_median:
                my_noisy_img = median_imgs.get_imgs_list()[3]
                return my_noisy_img, my_median_img, new_gt_img
            else:
                return my_median_img, new_gt_img

        else:
            my_json = os.path.splitext(input)[0] + '.json'
            drawImg = get_sky(my_json)

            # blur to sky label mask
            sd = 8
            truncate = 2
            radius = int(truncate * sd + 0.5)
            drawImg = cv2.GaussianBlur(drawImg * 255, (radius * 2 + 1, radius * 2 + 1), sd)
            drawImg = np.clip(drawImg, 0, 255)
            drawImg = drawImg / 255

            # noisy
            my_noisy_img = cv2.imread(input)

            # target
            my_gt_img = cv2.imread(target)
            if my_noisy_img.shape[0] != 1080:
                print(input, my_noisy_img.shape, drawImg.shape)
            clouds = my_noisy_img * drawImg
            forground = my_gt_img * (1 - drawImg)
            new_gt_img = clouds + forground

            return my_noisy_img, new_gt_img


    def get_db_len(self):
        return self.hf_DB.get_db_len()

    def __getitem__(self, index):
        median = self.median

        # 영상의 dir 받기.
        input, target = self.get_input_target_pairs(index, median=median)

        my_noisy_img, my_median_img, new_gt_img = self.make_noisy_and_new_gt(input, target, median=median, return_noise_and_median=True)

        h_input, w_input = my_noisy_img.shape[0], my_noisy_img.shape[1]

        # input_img 영역을 벗어나지 않는 범위 내에서, input_img 에서 crop 할 좌상단의 좌표를 구한다.
        left = np.random.randint(0, w_input - self.ipsize)  # Make margin as 0.
        top = np.random.randint(0, h_input - self.ipsize)

        input_img = my_noisy_img[top: top + self.ipsize, left: left + self.ipsize]
        median_img = my_median_img[top: top + self.ipsize, left: left + self.ipsize]
        target_img = new_gt_img[top: top + self.ipsize, left: left + self.ipsize]

        # random rotation
        mode = np.random.randint(4)
        input_img = utils.np_random_rotate(input_img, mode)
        median_img = utils.np_random_rotate(median_img, mode)
        target_img = utils.np_random_rotate(target_img, mode)

        # random flip
        mode = np.random.randint(2)
        input_img = utils.np_random_flip(input_img, mode)
        median_img = utils.np_random_flip(median_img, mode)
        target_img = utils.np_random_flip(target_img, mode)

        # tensor로 바꿔주기. and 각 영상들은 [0,1] 로 normalized.
        input_img = torch.from_numpy(np.array(input_img, np.float32, copy=True))
        median_img = torch.from_numpy(np.array(median_img, np.float32, copy=True))
        target_img = torch.from_numpy(np.array(target_img, np.float32, copy=True))
        input_img = input_img.permute((2, 0, 1)).contiguous() / 255
        median_img = median_img.permute((2, 0, 1)).contiguous() / 255
        target_img = target_img.permute((2, 0, 1)).contiguous() / 255

        return {'input_img': input_img, 'median_img': median_img, 'target_img': target_img}

    def __len__(self):
        # print('len', self.hf_DB.get_db_len())
        return self.db_len





