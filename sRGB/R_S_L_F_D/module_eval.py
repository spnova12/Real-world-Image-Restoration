
from os import listdir
from os.path import join
from glob import glob
import warnings

import tqdm
import os
import numpy as np

import sRGB.common.module_utils as utils
import sRGB.common.module_eval_tools as eval_tools

import cv2

import torch
import torchvision.transforms as transforms


def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps


class EvalModule(object):
    def __init__(self, train_set, net_dict, additional_info, cuda_num, median=False, noise_level=None):
        # Use train set for eval!
        self.train_set = train_set

        self.median = median
        self.noise_level = noise_level

        # my eval set from train set
        self.eval_set = self.get_eval_set(self.median)

        # gpu 는 뭘 사용할지,
        self.device = torch.device(f'cuda:{cuda_num}')

        # dict 형태의 additional_info 를 통해 그 외 eval 에 필요한 추가 정보를 받는다.
        self.additional_info = additional_info
        self.totensor = transforms.ToTensor()  # [0,255] -> [0,1] 로 만들어줌.

        # 모델 할당하기.
        self.netG = net_dict['G'].to(self.device)

        self.netA = None
        if 'A_pre' in net_dict:
            self.netA = net_dict['A_pre']
            self.netA.eval()
            for param in self.netA.parameters():
                param.requires_grad = False


    def get_eval_set(self, median):
        # how many imgs for eval
        imgs_for_eval_count = 20

        db_len = self.train_set.get_db_len()
        interval = db_len // imgs_for_eval_count

        eval_set = []
        for idx in range(imgs_for_eval_count):
            # for each noise level
            my_scene = []

            if self.noise_level is not None:
                # fix the noise number(index) as 0
                my_scene.append(self.train_set.get_input_target_pairs(index=idx * interval, noise_level=self.noise_level, noisy_num=0, median=median))
            else:
                # for loop for 4 levels
                for l in range(4):
                    # fix the noise number(index) as 0
                    my_scene.append(self.train_set.get_input_target_pairs(index=idx * interval, noise_level=l+1, noisy_num=0, median=median))

            eval_set.append(my_scene)

        return eval_set

    # RGB 영상으로 저장해서 관찰할 수 있게 해보았다. + psnr 도 측정해보도록 한다.
    def save_input_and_target(self, save_dir):

        psnr_dict = {}

        for eval_pairs in self.eval_set:
            # (0) Make new dir from target dir.
            target = eval_pairs[0][1]
            target_name = os.path.splitext(os.path.basename(target))[0]
            new_dir = utils.make_dirs(f'{save_dir}/{target_name}')

            # (1) write target.
            target_img = cv2.imread(target)
            cv2.imwrite(f'{new_dir}/{target_name}.png', target_img)

            for eval_pair in eval_pairs:
                input, target = eval_pair

                target_img_new_aligned = None

                if self.median:
                    b_name = os.path.splitext(os.path.basename(input[3]))[0]
                    input_img, my_median_img, target_img_new = self.train_set.make_noisy_and_new_gt(input, target,
                                                                                                    median=self.median,
                                                                                                    return_noise_and_median=True)

                    if self.netA is not None:
                        # (2-0) Make aligned image through netA
                        h, w, _ = input_img.shape
                        target_img_new_aligned = eval_tools.recon_big_one_frame(
                            [my_median_img, target_img_new],
                            (w, h), scale_factor=1, net=self.netA, minimum_wh=2000, device=self.device)

                    # (2-1) write median
                    cv2.imwrite(f'{new_dir}/{b_name}_median.png', my_median_img)

                else:
                    b_name = os.path.splitext(os.path.basename(input))[0]
                    input_img, target_img_new = self.train_set.make_noisy_and_new_gt(input, target)

                    if self.netA is not None:
                        # (2-0) Make aligned image through netA
                        h, w, _ = input_img.shape
                        target_img_new_aligned = eval_tools.recon_big_one_frame(
                            [input_img, target_img_new],
                            (w, h), scale_factor=1, net=self.netA, minimum_wh=2000, device=self.device)

                target_img = cv2.imread(target)

                # (2) write input.
                cv2.imwrite(f'{new_dir}/{b_name}.png', input_img)

                # (3) write target_img_new
                cv2.imwrite(f'{new_dir}/{b_name}_target_img_new.png', target_img_new)

                # get psnr
                psnr_dict[f'{str(b_name)}'] = [numpyPSNR(input_img, target_img)]

                if self.netA is not None:
                    cv2.imwrite(f'{new_dir}/{b_name}_target_img_new_aligned.png', target_img_new_aligned)
                    psnr_dict[f'{str(b_name)}'].append(numpyPSNR(input_img, target_img_new_aligned))

                else:
                    psnr_dict[f'{str(b_name)}'].append(numpyPSNR(input_img, target_img_new))

        return psnr_dict

    def save_output(self, save_dir, iter):
        # dataset 별 psnr 을 저장할 dict, key:데이터 셋 name, value:psnr

        psnr_dict = {}

        for eval_pairs in tqdm.tqdm(self.eval_set):
            # (0) Make new dir from target dir.
            target = eval_pairs[0][1]
            target_name = os.path.splitext(os.path.basename(target))[0]
            new_dir = utils.make_dirs(f'{save_dir}/{target_name}')

            for eval_pair in eval_pairs:
                input, target = eval_pair

                target_img_new_aligned = None

                if self.median:
                    b_name = os.path.splitext(os.path.basename(input[3]))[0]
                    input_img, my_median_img, target_img_new = self.train_set.make_noisy_and_new_gt(input, target,
                                                                                                    median=self.median,
                                                                                                    return_noise_and_median=True)

                    if self.netA is not None:
                        # (2-0) Make aligned image through netA
                        h, w, _ = input_img.shape
                        target_img_new_aligned = eval_tools.recon_big_one_frame(
                            [my_median_img, target_img_new],
                            (w, h), scale_factor=1, net=self.netA, minimum_wh=2000, device=self.device)

                else:
                    b_name = os.path.splitext(os.path.basename(input))[0]
                    input_img, target_img_new = self.train_set.make_noisy_and_new_gt(input, target)

                    if self.netA is not None:
                        # (2-0) Make aligned image through netA
                        h, w, _ = input_img.shape
                        target_img_new_aligned = eval_tools.recon_big_one_frame(
                            [input_img, target_img_new],
                            (w, h), scale_factor=1, net=self.netA, minimum_wh=2000, device=self.device)

                # (2) get the result
                h, w, _ = input_img.shape
                recon_img = eval_tools.recon_big_one_frame(
                    input_img,
                    (w, h), scale_factor=1, net=self.netG, minimum_wh=2000, device=self.device)

                cv2.imwrite(f'{new_dir}/{b_name}_recon_{str(iter).zfill(10)}.png', recon_img)

                # get psnr
                if self.netA is not None:
                    psnr_dict[f'{str(b_name)}'] = numpyPSNR(recon_img, target_img_new_aligned)
                else:
                    psnr_dict[f'{str(b_name)}'] = numpyPSNR(recon_img, target_img_new)

        return psnr_dict
