import os.path

import torch
import sRGB.common.module_eval_tools as eval_tools
from sRGB.common.module_utils import make_dirs
import cv2
import tqdm
import csv
import random

import sRGB.common_net.MPRNet as MPRNet
import sRGB.common_net.GRDN as GRDN

from sRGB.preprocessing.module_DB_manager import HumanForrestManager, get_sky, MedianImgs
import sRGB.common.module_utils as utils
import sRGB.preprocessing.module_data as module_data
from sRGB.R_S_L_F_D.module_eval import numpyPSNR
from sRGB.preprocessing.module_DB_manager import read_text, write_text


import datetime as pydatetime

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

    # Set model.
    my_model = MPRNet.MPRNet(in_c=3)
    myNet = eval_tools.NetForInference(my_model, cuda_num=cuda_num)
    myNet.weight_loader(pretrain_net_dir_for_test)

    # Set Device.
    device = torch.device(f'cuda:{cuda_num}')

    # Get data dir list.
    test_DB_dir_list = [os.path.join(input_folder_dir, x) for x in sorted(os.listdir(input_folder_dir))]

    # Set out dir.
    out_dir = make_dirs(f'test-out/{out_folder_name}', )

    # Infer and save.
    print('===> Denoise the images')
    for test_DB_dir in tqdm.tqdm(test_DB_dir_list):
        input_img = cv2.imread(test_DB_dir)

        h, w, _ = input_img.shape

        # Denoise images.
        recon_img = eval_tools.recon_big_one_frame(
            input_img,
            (w, h), scale_factor=1, net=myNet.netG, minimum_wh=2000, device=device)

        # Get basename and save reconstructed image.
        b_name = os.path.splitext(os.path.basename(test_DB_dir))[0]
        cv2.imwrite(f'{out_dir}/{b_name}_recon.png', recon_img)


def main2(pretrain_net_dir_for_test, pretrain_net_dir_for_align, DB_dir, noise_type, cuda_num=None):

    # Set model.
    my_model = MPRNet.MPRNet(in_c=3)
    myNet = eval_tools.NetForInference(my_model, cuda_num=cuda_num)
    myNet.weight_loader(pretrain_net_dir_for_test)

    my_model_A = GRDN.GRDN(input_channel=3)
    myNet_A = eval_tools.NetForInference(my_model_A, cuda_num=cuda_num)
    myNet_A.weight_loader(pretrain_net_dir_for_align)

    # Set Device.
    device = torch.device(f'cuda:{cuda_num}')

    # <><><> DB with Median depend on noise type.
    if noise_type == 'R' or noise_type == 'S':
        median = True
    else:
        median = False

    # Get Dataset
    train_set = module_data.DatasetForDataLoader(
        DB_dir,
        noise_type,
        median=median,
        return_noise_and_median=median
    )

    ################################################################################################
    db_len = train_set.get_db_len()
    eval_set = []
    for idx in range(db_len):
        # for each noise level
        my_scene = []

        # for loop for 4 levels
        for l in range(6):
            v = train_set.get_input_target_pairs(index=idx, noise_level=l + 1, noisy_num=0,
                                                 median=median)
            if v:
                my_scene.append(v)

            v2 = train_set.get_input_target_pairs(index=idx, noise_level=l + 1, noisy_num=2,
                                                 median=median)
            if v2:
                my_scene.append(v2)

        eval_set.append(my_scene)


    ################################################################################################
    print('\n:: timestamp:', get_now_timestamp())
    print(':: (Among all data, items that are not in the test set are skipped.)', noise_type)

    psnr_dict = {}
    total_pair_size = 0

    preprocessing_dir = os.path.dirname(os.path.realpath(__file__))
    test_DB_list_txt_dir = f'{preprocessing_dir}/test_DB_{noise_type}_list.txt'

    test_DB_list = None
    if os.path.isfile(test_DB_list_txt_dir):
        test_DB_list = sorted(read_text(test_DB_list_txt_dir))


    # todo : check length
    for eval_pairs in tqdm.tqdm(eval_set):
        # (0) Make new dir from target dir.
        target = eval_pairs[0][1]
        target_name = os.path.splitext(os.path.basename(target))[0]
        # new_dir = utils.make_dirs(f'{save_dir}/{target_name}')

        for eval_pair in eval_pairs:
            input, target = eval_pair
            total_pair_size += 1

            input_img = None
            target_img_new_aligned = None
            do_infer = False

            if median:
                b_name = os.path.splitext(os.path.basename(input[3]))[0]
                if test_DB_list is None:
                    do_infer = True
                elif b_name in test_DB_list:
                    do_infer = True
                if do_infer:
                    input_img, my_median_img, target_img_new = train_set.make_noisy_and_new_gt(input, target,
                                                                                               median=median,
                                                                                               return_noise_and_median=True)

                    # (2-0) Make aligned image through netA
                    h, w, _ = input_img.shape
                    target_img_new_aligned = eval_tools.recon_big_one_frame(
                        [my_median_img, target_img_new],
                        (w, h), scale_factor=1, net=myNet_A.netG, minimum_wh=2000, device=device)

            else:
                b_name = os.path.splitext(os.path.basename(input))[0]
                if test_DB_list is None:
                    do_infer = True
                elif b_name in test_DB_list:
                    do_infer = True
                if do_infer:
                    input_img, target_img_new = train_set.make_noisy_and_new_gt(input, target)


                    # (2-0) Make aligned image through netA
                    h, w, _ = input_img.shape
                    target_img_new_aligned = eval_tools.recon_big_one_frame(
                        [input_img, target_img_new],
                        (w, h), scale_factor=1, net=myNet_A.netG, minimum_wh=2000, device=device)

            if target_img_new_aligned is not None:
                # (2) get the result
                h, w, _ = input_img.shape
                recon_img = eval_tools.recon_big_one_frame(
                    input_img,
                    (w, h), scale_factor=1, net=myNet.netG, minimum_wh=2000, device=device)

                # cv2.imwrite(f'{new_dir}/{b_name}_recon_{str(iter).zfill(10)}.png', recon_img)

                # get psnr
                psnr_dict[f'{str(b_name)}'] = numpyPSNR(recon_img, target_img_new_aligned)

    ################################################################################################

    Test_rate = 10
    if test_DB_list is None:

        psnrs = sorted(psnr_dict.items(), key=lambda x: x[1], reverse=True)

        f = 70  # Actual valid rate
        test_size = int(len(psnr_dict) * (f / 100))
        psnrs = psnrs[:test_size]

        test_size = int(len(psnrs) * (Test_rate/100))
        psnrs = random.sample(psnrs, test_size)
    else:
        psnrs = sorted(psnr_dict.items(), key=lambda x: x[1], reverse=True)

    # to list
    psnrs = [[x, y] for (x, y) in psnrs]
    keys = [x for (x, y) in psnrs]
    v = [y for (x, y) in psnrs]

    result_average = sum(v)/len(v)

    print(f':: Test rate : {Test_rate}%')
    print(f':: Average PSNR : {result_average}')

    write_text(test_DB_list_txt_dir, keys, -1)

    make_dirs('test-out')
    csv_dir = f'test-out/test_DB_{noise_type}_psnrs.csv'
    print(f':: For detailed psnr values for each test image, refer to the following file: {csv_dir}')
    with open(csv_dir, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(psnrs)
        writer.writerow(['Average', result_average])

        # print(read_text(f'a.txt'))


    print(':: timestamp:', get_now_timestamp())
    # Set out dir.
    # out_dir = make_dirs(f'test-out/{out_folder_name}', )



