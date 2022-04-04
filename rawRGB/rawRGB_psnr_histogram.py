from rawRGB.common.module_init_raw_DB_utils import *
import torch.backends.cudnn as cudnn
# cudnn.benchmark allows you to enable the inbuilt cudnn auto-tuner
# to find the best algorithm to use for your hardware.
cudnn.benchmark = True

import rawRGB.common.module_utils as utils

import rawRGB.nlowlight_indoor.module_data as module_data
import matplotlib.pyplot as plt


def get_DB_psnr_histogram(DB_dir):
    hf_patches_folder_dir = DB_dir + '_patches'
    json_folder_dir = DB_dir

    # Get Dataset
    hf_DB = DB_dir_to_paired_list(hf_patches_folder_dir, json_folder_dir)

    hf_DB_new = []
    Test_rate = 10

    imgs_for_test_count = int(len(hf_DB) * (Test_rate / 100))

    db_len = len(hf_DB)
    interval = db_len // imgs_for_test_count

    for idx in tqdm.tqdm(range(imgs_for_test_count)):
        input_target_pairs = hf_DB[idx * interval]
        hf_DB_new.append(input_target_pairs)

    psnr_list = []

    eval_set = hf_DB_new
    for eval_pair in tqdm.tqdm(eval_set):
        # get pair dir.
        input = eval_pair['input']
        target = eval_pair['target']
        json = eval_pair['json']
        input_metadata_dict = eval_pair['input_metadata_dict']
        target_metadata_dict = eval_pair['target_metadata_dict']

        # read raw images with target_img_new.
        input_img, target_img, target_img_new = \
            module_data.make_noisy_and_new_gt(input, target,
                                              json,
                                              input_metadata_dict, target_metadata_dict)
        # get psnr
        psnr_list.append(utils.get_psnr(input_img, target_img, min_value=0, max_value=1))

    weight = psnr_list
    bins = 100
    plt.hist(weight, bins=bins, label=f'bins={bins}')
    plt.title(f"L-raw DB psnr histogram")
    plt.legend()
    plt.savefig(f"test-out/DB_psnr_histogram_L_raw.png")

