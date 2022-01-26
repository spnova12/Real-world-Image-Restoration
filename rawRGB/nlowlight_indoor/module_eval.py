
from os import listdir
from os.path import join
from glob import glob
import warnings

import tqdm
import pandas as pd
import os
import numpy as np

import rawRGB.common.module_utils as utils
import rawRGB.common.module_eval_tools as eval_tools
from rawRGB.common.module_init_raw_DB_utils import *

import cv2

import torch
import torchvision.transforms as transforms


class EvalModule(object):
    def __init__(self, train_set, net_dict, additional_info, cuda_num):
        # Use train set for eval!
        self.train_set = train_set

        # my eval set from train set
        self.eval_set = self.get_eval_set()

        # gpu 는 뭘 사용할지,
        self.device = torch.device(f'cuda:{cuda_num}')

        # dict 형태의 additional_info 를 통해 그 외 eval 에 필요한 추가 정보를 받는다.
        self.additional_info = additional_info

        # 모델 할당하기.
        self.netG = net_dict['G'].to(self.device)

    def get_eval_set(self):
        # how many imgs for eval
        imgs_for_eval_count = 20  # todo

        the_dataset_list = self.train_set.get_the_dataset_list()

        db_len = len(the_dataset_list)
        interval = db_len // imgs_for_eval_count

        eval_set = []
        for idx in range(imgs_for_eval_count):
            input_target_pairs = the_dataset_list[idx * interval]
            eval_set.append(input_target_pairs)

        print_wrap(f'imgs_for_eval_count : {imgs_for_eval_count}')
        return eval_set

        # RGB 영상으로 저장해서 관찰할 수 있게 해보았다. + psnr 도 측정해보도록 한다.

    def save_input_and_target(self, save_dir, visualize_eval_raw=False):

        psnr_dict = {}

        for eval_pair in tqdm.tqdm(self.eval_set):
            # get pair dir.
            input = eval_pair['input']
            target = eval_pair['target']
            json = eval_pair['json']
            input_metadata_dict = eval_pair['input_metadata_dict']
            target_metadata_dict = eval_pair['target_metadata_dict']

            # get base name to use.
            b_name = os.path.splitext(os.path.basename(input))[0]

            # Make new dir for save eval from input dir.
            new_dir = utils.make_dirs(f'{save_dir}/{b_name}')

            # read raw images with target_img_new.
            input_img, target_img, target_img_new = \
                self.train_set.make_noisy_and_new_gt(input, target,
                                                     json,
                                                     input_metadata_dict, target_metadata_dict)

            # get psnr
            psnr_dict[f'{str(b_name)}'] = [utils.get_psnr(input_img, target_img, min_value=0, max_value=1)]
            psnr_dict[f'{str(b_name)}'].append(utils.get_psnr(input_img, target_img_new, min_value=0, max_value=1))

            # visualize eval raws.
            if visualize_eval_raw:
                # (1) visualize input.
                input_uint8 = normalized_dng_postprocess_for_patch(normalized_raw=input_img,
                                                                   metadata_dict=read_obj(input_metadata_dict))
                cv2.imwrite(f'{new_dir}/{b_name}.png', input_uint8)

                # (2) visualize target.
                target_uint8 = normalized_dng_postprocess_for_patch(normalized_raw=target_img,
                                                                    metadata_dict=read_obj(target_metadata_dict))
                target_name = os.path.splitext(os.path.basename(target))[0]
                cv2.imwrite(f'{new_dir}/{target_name}.png', target_uint8)

                # (3) visualize target_new
                target_new_uint8 = normalized_dng_postprocess_for_patch(normalized_raw=target_img_new,
                                                                        metadata_dict=read_obj(target_metadata_dict))
                cv2.imwrite(f'{new_dir}/{b_name}_new_target.png', target_new_uint8)

        return psnr_dict

    def save_output(self, save_dir, iter, visualize_eval_raw=False):
        # dataset 별 psnr 을 저장할 dict, key:데이터 셋 name, value:psnr

        psnr_dict = {}

        for eval_pair in tqdm.tqdm(self.eval_set):
            # get pair dir.
            input = eval_pair['input']
            target = eval_pair['target']
            json = eval_pair['json']
            input_metadata_dict = eval_pair['input_metadata_dict']
            target_metadata_dict = eval_pair['target_metadata_dict']

            # get base name to use.
            b_name = os.path.splitext(os.path.basename(input))[0]

            # Make new dir for save eval from input dir.
            new_dir = utils.make_dirs(f'{save_dir}/{b_name}')

            # read raw images with target_img_new.
            input_img, target_img, target_img_new = \
                self.train_set.make_noisy_and_new_gt(input, target,
                                                     json,
                                                     input_metadata_dict, target_metadata_dict)

            # get the result
            h, w, _ = input_img.shape
            recon_img = eval_tools.recon_big_one_frame(
                input_img,
                (w, h), scale_factor=1, net=self.netG, minimum_wh=1000, device=self.device)

            # get psnr
            psnr_dict[f'{str(b_name)}'] = utils.get_psnr(recon_img, target_img_new, min_value=0, max_value=1)

            if visualize_eval_raw:
                # visualize recon_img
                recon_img_uint8 = normalized_dng_postprocess_for_patch(normalized_raw=recon_img,
                                                                       metadata_dict=read_obj(input_metadata_dict))
                cv2.imwrite(f'{new_dir}/{b_name}_transferred_{str(iter).zfill(10)}.png', recon_img_uint8)

        return psnr_dict