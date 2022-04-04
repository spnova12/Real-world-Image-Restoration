import os.path

import tqdm
import cv2

import sRGB.preprocessing.module_data as module_data
from sRGB.R_S_L_F_D.module_eval import numpyPSNR

import matplotlib.pyplot as plt



def get_DB_psnr_histogram(DB_dir, noise_type):
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
            sample_l = 10
            for noisy_num_i in range(sample_l):
                v = train_set.get_input_target_pairs(index=idx, noise_level=l + 1, noisy_num=noisy_num_i,
                                                     median=median)
                if v:
                    my_scene.append(v)

        eval_set.append(my_scene)


    ################################################################################################

    psnr_list = []
    for eval_pairs in tqdm.tqdm(eval_set):
        for eval_pair in eval_pairs:
            input, target = eval_pair
            input_img = cv2.imread(input)
            target_img = cv2.imread(target)

            # get psnr
            psnr_list.append(numpyPSNR(input_img, target_img))


    weight = psnr_list
    bins = 100
    plt.hist(weight, bins=bins, label=f'bins={bins}')
    plt.title(f"{noise_type} DB psnr histogram")
    plt.legend()
    plt.savefig(f"test-out/DB_psnr_histogram_{noise_type}.png")



