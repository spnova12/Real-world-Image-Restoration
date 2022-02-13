import os.path

import torch
import sRGB.common.module_eval_tools as eval_tools
from sRGB.common.module_utils import make_dirs
import cv2
import tqdm

def main(pretrain_net_dir_for_test, test_DB_dir_list, out_dir_name, cuda_num):
    out_dir = make_dirs(f'test-out/{out_dir_name}', )

    if cuda_num:
        device = torch.device(f'cuda:{cuda_num}')
    else:
        device = torch.device(f'cuda:0')

    myNet = eval_tools.NetForInference(input_channel=3, device=device)
    myNet.weight_loader(pretrain_net_dir_for_test)

    print('===> Denoise the images')
    for test_DB_dir in tqdm.tqdm(test_DB_dir_list):
        input_img = cv2.imread(test_DB_dir)

        h, w, _ = input_img.shape

        # Denoise images.
        recon_img = eval_tools.recon_big_one_frame(
            input_img,
            (w, h), scale_factor=1, net=myNet.netG, minimum_wh=2000, device=device)

        # Get basename and save reconstructed image.
        b_name = os.path.basename(test_DB_dir)
        cv2.imwrite(f'{out_dir}/{b_name}_recon.png', recon_img)








