import numpy as np
import math
import cv2
import os

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
from torch.utils.data import DataLoader
import torch

import argparse

import torch.backends.cudnn as cudnn
# cudnn.benchmark allows you to enable the inbuilt cudnn auto-tuner
# to find the best algorithm to use for your hardware.
cudnn.benchmark = True

import rawRGB.common.module_utils as utils
import rawRGB.nlowlight_indoor.module_data as module_data
import rawRGB.nlowlight_indoor.module_train as module_train
import rawRGB.nlowlight_indoor.module_eval as module_eval

import datetime as pydatetime

# <><><> 사용할 net architecture 선택하기.
import rawRGB.common_net.MPRNet as MPRNet
import rawRGB.common.module_eval_tools as eval_tools
import rawRGB.nlowlight_indoor.module_data as module_data



def get_now():
    """
        현재 시스템 시간을 datetime형으로 반환
    """
    return pydatetime.datetime.now()

def get_now_timestamp():
    """
        현재 시스템 시간을 POSIX timestamp float형으로 반환
    """
    return get_now().timestamp()



def main(pretrain_net_dir_for_test, input_folder_dir, out_folder_name, cuda_num):
    # Set model
    my_model = MPRNet.MPRNet(in_c=4, out_c=4)
    myNet = eval_tools.NetForInference(my_model, cuda_num=cuda_num)
    myNet.weight_loader(pretrain_net_dir_for_test)

    # Set Device.
    device = torch.device(f'cuda:{cuda_num}')

    # Get Dataset
    print_wrap('Read all the patches.(.bz2)')
    # read only image not metadata.
    raw_dir_list = [x for x in glob.glob(f"{input_folder_dir}/*.bz2")
                    if 'metadata_dict' not in x and 'cfa_mask' not in x]

    for raw_dir in tqdm.tqdm(raw_dir_list):
        # get pair dir.
        input = read_obj(raw_dir)
        input_metadata_dict = read_obj(raw_dir.split('___')[0] + '___metadata_dict.bz2')
        input_cfa_mask = read_obj(f"{os.path.splitext(raw_dir)[0]}_cfa_mask.bz2")

        input_img = raw_16bit_to_normalized_raw(input, input_metadata_dict, input_cfa_mask)

        # get base name to use.
        b_name = os.path.splitext(os.path.basename(raw_dir))[0]

        # get the result
        h, w, _ = input_img.shape
        recon_img = eval_tools.recon_big_one_frame(
            input_img,
            (w, h), scale_factor=1, net=myNet.netG, minimum_wh=1000, device=device)


        # visualize recon_img
        recon_img_uint8 = normalized_dng_postprocess_for_patch(normalized_raw=recon_img,
                                                               metadata_dict=input_metadata_dict)
        # Make new dir for save eval from input dir.
        new_dir = utils.make_dirs(f'{out_folder_name}')
        cv2.imwrite(f'{new_dir}/{b_name}_reconstructed.png', recon_img_uint8)


def main2(pretrain_net_dir_for_test, hf_patches_folder_dir, json_folder_dir, cuda_num):

    # Set model
    my_model = MPRNet.MPRNet(in_c=4, out_c=4)
    myNet = eval_tools.NetForInference(my_model, cuda_num=cuda_num)
    myNet.weight_loader(pretrain_net_dir_for_test)

    # Set Device.
    device = torch.device(f'cuda:{cuda_num}')

    # Get Dataset
    hf_DB = DB_dir_to_paired_list(hf_patches_folder_dir, json_folder_dir)


    print('---------------------------------')
    print(len(hf_DB))

    ################################################################################################
    preprocessing_dir = os.path.dirname(os.path.realpath(__file__))
    test_DB_list_txt_dir = f'{preprocessing_dir}/test_DB_L_raw_list.txt'

    test_DB_list = None
    if os.path.isfile(test_DB_list_txt_dir):
        test_DB_list = sorted(read_text(test_DB_list_txt_dir))

    ################################################################################################
    hf_DB_new = []
    Test_rate = 10

    if test_DB_list is not None:
        for hf in tqdm.tqdm(hf_DB):
            if os.path.basename(hf['input']) in test_DB_list:
                hf_DB_new.append(hf)
    else:
        imgs_for_test_count = int(len(hf_DB) * (Test_rate / 100))

        db_len = len(hf_DB)
        interval = db_len // imgs_for_test_count

        for idx in tqdm.tqdm(range(imgs_for_test_count)):
            input_target_pairs = hf_DB[idx * interval]
            hf_DB_new.append(input_target_pairs)

        test_DB_list = [x['input'] for x in hf_DB_new]
        write_text(test_DB_list_txt_dir, test_DB_list, -1)

    ################################################################################################
    print('\n:: timestamp:', get_now_timestamp())
    print(f':: noise_type: L_raw')
    print(':: (Among all data, tems that are not in the test set are skipped.)')

    ################################################################################################

    psnr_dict = {}

    eval_set = hf_DB_new
    for eval_pair in tqdm.tqdm(eval_set):
        # get pair dir.
        input = eval_pair['input']
        target = eval_pair['target']
        json = eval_pair['json']
        input_metadata_dict = eval_pair['input_metadata_dict']
        target_metadata_dict = eval_pair['target_metadata_dict']

        # get base name to use.
        b_name = os.path.splitext(os.path.basename(input))[0]

        # read raw images with target_img_new.
        input_img, target_img, target_img_new = \
            module_data.make_noisy_and_new_gt(input, target,
                                              json,
                                              input_metadata_dict, target_metadata_dict)

        # get the result
        h, w, _ = input_img.shape
        recon_img = eval_tools.recon_big_one_frame(
            input_img,
            (w, h), scale_factor=1, net=myNet.netG, minimum_wh=1000, device=device)

        # get psnr
        psnr_dict[f'{str(b_name)}'] = utils.get_psnr(recon_img, target_img_new, min_value=0, max_value=1)


    ################################################################################################

    psnrs = sorted(psnr_dict.items(), key=lambda x: x[1], reverse=True)

    # to list
    psnrs = [[x, y] for (x, y) in psnrs]
    keys = [x for (x, y) in psnrs]
    v = [y for (x, y) in psnrs]

    result_average = sum(v) / len(v)

    print(f':: Test rate : {Test_rate}%')
    print(f':: Average PSNR : {result_average}')

    write_text(test_DB_list_txt_dir, keys, -1)

    make_dirs('test-out')
    csv_dir = f'test-out/test_DB_L_raw_psnrs.csv'
    print(f':: For detailed psnr values for each test image, refer to the following file: {csv_dir}')
    with open(csv_dir, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(psnrs)
        writer.writerow(['Average', result_average])