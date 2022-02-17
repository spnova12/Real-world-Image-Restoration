
from os import listdir
from os.path import join
import os

import math
import numpy as np
import torch.utils.data as data
import random
import cv2

from rawRGB.common.module_init_raw_DB_utils import *
from rawRGB.common.module_raw_utils import get_normalized_raw_from_dng, normalized_dng_postprocess

import sys
import rawRGB.common.module_utils as utils

import torchvision.transforms as transforms
import torch
import torch.backends.cudnn as cudnn
# cudnn.benchmark allows you to enable the inbuilt cudnn auto-tuner
# to find the best algorithm to use for your hardware.
cudnn.benchmark = True


def make_noisy_and_new_gt(input, target, json, input_metadata_dict, target_metadata_dict):
    cfa_data_patch_position = os.path.splitext(input)[0].split('___')[1].split('_')
    input_cfa_mask = read_obj(f"{os.path.splitext(input)[0]}_cfa_mask.bz2")
    target_cfa_mask = read_obj(f"{os.path.splitext(target)[0]}_cfa_mask.bz2")

    j, i, w, h = [int(x) for x in cfa_data_patch_position]

    # noisy
    my_noisy_img = raw_16bit_to_normalized_raw(read_obj(input), read_obj(input_metadata_dict), input_cfa_mask)
    # target
    my_gt_img = raw_16bit_to_normalized_raw(read_obj(target), read_obj(target_metadata_dict), target_cfa_mask)

    # get only one chennel.
    drawImg = get_screens(json)[:, :, 0]

    # resize
    drawImg = cv2.resize(drawImg, dsize=(5796, 3870),
                         interpolation=cv2.INTER_LINEAR)
    # crop
    drawImg = drawImg[i: (i + h), j: (j + w)]

    drawImg = cv2.resize(drawImg, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # blur to sky label mask
    sd = 8
    truncate = 2
    radius = int(truncate * sd + 0.5)
    drawImg = cv2.GaussianBlur(drawImg * 255, (radius * 2 + 1, radius * 2 + 1), sd)
    drawImg = np.clip(drawImg, 0, 255)
    drawImg = drawImg / 255

    # map to 4channels
    drawImg = np.repeat(drawImg.reshape(drawImg.shape[0], drawImg.shape[1], 1), 4, axis=2)

    # Apply the map
    clouds = my_noisy_img * drawImg
    forground = my_gt_img * (1 - drawImg)
    new_gt_img = clouds + forground

    return my_noisy_img, my_gt_img, new_gt_img


class DatasetForDataLoader(data.Dataset):
    """
    여러 데이터셋을 지정한 비율대로 섞어서 사용할 수 있다.
    가장 크기가 큰 데이터셋을 기준으로 이미지 수를 조정해준다.
    """
    def __init__(self, hf_patches_folder_dir, json_folder_dir, additional_info=None):
        """
        :param img_dirs: [(input, target), (input, target) ... (input, target)]
        """
        self.hf_DB = DB_dir_to_paired_list(hf_patches_folder_dir, json_folder_dir)
        self.ipsize = additional_info['input_patch_size']

        # torchvision 에 있는 함수들은 될수있으면 그대로 가져다 사용하자.
        self.random_crop = transforms.RandomCrop(self.ipsize)

    def get_the_dataset_list(self):
        return self.hf_DB

    def make_noisy_and_new_gt(self, input, target, json, input_metadata_dict, target_metadata_dict):
        my_noisy_img, my_gt_img, new_gt_img = make_noisy_and_new_gt(input, target, json,
                                                                    input_metadata_dict, target_metadata_dict)
        return my_noisy_img, my_gt_img, new_gt_img


    def __getitem__(self, index):
        # 영상의 dir 받기.
        input = self.hf_DB[index]['input']
        target = self.hf_DB[index]['target']
        json = self.hf_DB[index]['json']
        input_metadata_dict = self.hf_DB[index]['input_metadata_dict']
        target_metadata_dict = self.hf_DB[index]['target_metadata_dict']

        my_noisy_img, _, new_gt_img = self.make_noisy_and_new_gt(input, target, json,
                                                                 input_metadata_dict, target_metadata_dict)

        h_input, w_input = my_noisy_img.shape[0], my_noisy_img.shape[1]

        # input_img 영역을 벗어나지 않는 범위 내에서, input_img 에서 crop 할 좌상단의 좌표를 구한다.
        left = np.random.randint(0, w_input - self.ipsize)  # Make margin as 0.
        top = np.random.randint(0, h_input - self.ipsize)

        input_img = my_noisy_img[top: top + self.ipsize, left: left + self.ipsize]
        target_img = new_gt_img[top: top + self.ipsize, left: left + self.ipsize]


        # random rotation
        mode = np.random.randint(4)
        input_img = utils.np_random_rotate(input_img, mode)
        target_img = utils.np_random_rotate(target_img, mode)

        # random flip
        mode = np.random.randint(2)
        input_img = utils.np_random_flip(input_img, mode)
        target_img = utils.np_random_flip(target_img, mode)


        # tensor로 바꿔주기. and 각 영상들은 [0,1] 로 normalized.
        input_img = torch.from_numpy(np.array(input_img, np.float32, copy=True))
        target_img = torch.from_numpy(np.array(target_img, np.float32, copy=True))

        # The tensors are already normalize to 0~1 in noisy_gt_aligner_train.py
        input_img = input_img.permute((2, 0, 1)).contiguous()
        target_img = target_img.permute((2, 0, 1)).contiguous()

        return {'input_img': input_img,
                'target_img': target_img,
                'json': json,
                'input_metadata_dict': input_metadata_dict,
                'target_metadata_dict': target_metadata_dict}

    def __len__(self):

        return len(self.hf_DB)





