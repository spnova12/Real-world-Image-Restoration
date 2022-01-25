import os
import glob

import cv2

from rawRGB.common.module_init_raw_DB_utils import *
from skimage.metrics import structural_similarity as compare_ssim
from rawRGB.common.module_check_patches_findbestpairs_module import *
from rawRGB.common.module_init_raw_DB_utils import *


def get_ssim(a, b):
    a = a[:, :, :3] * 255
    b = b[:, :, :3] * 255
    a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    score, _ = compare_ssim(a, b, full=True)
    return score


def check_patches_findbestpairs(DB_dir):
    # Get this script's dir.
    preprocessing_dir = os.path.dirname(os.path.realpath(__file__))

    patch_DB_dir = DB_dir + '_patches'
    json_folder_dir = DB_dir

    # pair_based_gt = f'{preprocessing_dir}/pair_based_gt.pkl'
    #
    # if os.path.isfile(pair_based_gt):
    #     print_wrap('pair_based_gt.pkl is exist. Load the data.')
    #     raw_dir_paired_list = read_obj(pair_based_gt)
    # else:
    # Get the paired list from DB_dir.
    print_wrap(f'pair_based_gt.pkl is not exist. Make pair_based_gt.pkl')
    raw_dir_paired_list = DB_dir_to_paired_list(patch_DB_dir, json_folder_dir)
    # write_obj(pair_based_gt, raw_dir_paired_list, 'pkl')

    print_wrap('list to dict (key is GT)')
    gt_based_dict = {}
    for raw_dir_paired in tqdm.tqdm(raw_dir_paired_list):
        new_key = raw_dir_paired['target']
        if new_key not in gt_based_dict.keys():
            gt_based_dict[new_key] = []

        gt_based_dict[new_key].append(raw_dir_paired)

    gt_keys = list(gt_based_dict.keys())
    print_wrap(f'gt keys len : {len(gt_keys)}')

    real_mode = True
    if real_mode:
        print_wrap('Delete useless noisy patches')
        sample_save_dir = None
    else:
        print_wrap('save samples for visualization')
        sample_save_dir = make_dirs('toy_test/sorted_by_psnr_5')


    # multiprocess doesn't mean much to this for loop. Single process is more fast.
    find_good_noisy_pair = FindGoodNoisyPair(gt_based_dict, sample_save_dir=sample_save_dir, real_mode=True)
    for gt_key in tqdm.tqdm(gt_keys):
        find_good_noisy_pair.find_good_noisy_pair(gt_key)



