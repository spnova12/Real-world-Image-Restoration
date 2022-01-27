import numpy as np
import math
import cv2

from torch.utils.data import DataLoader
import torch

import sRGB.common.module_utils as utils
import sRGB.preprocessing.module_data as module_data
import sRGB.preprocessing.module_eval as module_eval
import sRGB.preprocessing.module_train as module_train

import argparse

import torch.backends.cudnn as cudnn
# cudnn.benchmark allows you to enable the inbuilt cudnn auto-tuner
# to find the best algorithm to use for your hardware.
cudnn.benchmark = True

# <><><> 사용할 net architecture 선택하기.
import sRGB.common_net.GRDN as net


def noisy_gt_aligner_train(exp_name, hf_DB_dir, noise_type, cuda_num=None):
    # pytorch 버전 출력하기.
    print('\n===> Pytorch version :', torch.__version__)

    # <><><> 사용할 gpu 번호. (multi gpu 를 사용하려면 '0 으로 하고, DataParallel 을 True 로 해줌)
    # If cuda_num == None -> it means using DataParallel.
    if cuda_num == None:
        cuda_num = 0
        DataParallel = True
    else:
        DataParallel = False

    # <><><> 실험 이름.
    # exp_name = f'018'
    exp_dir = utils.make_dirs(f'train-out/{exp_name}')
    print(f'\n===> exp_name : {exp_name}')

    # <><><> noise type ('R', 'F', 'D', 'S', 'L' : Rain, Fog, Dust, Snow, Lowlight)
    # noise_type = 'R'

    # <><><> DB with Median depend on noise type.
    if noise_type == 'R' or noise_type == 'S':
        median = True
    else:
        median = False

    edge_lambda = 1
    color_lambda = 1.5
    sigma = 3


    # <><><> checkpoint version
    checkpoint_version = 'checkpoint.pth'



    ####################################################################################
    ####################################################################################
    ####################################################################################

    # <><><> training 과 valid 에 추가로 사용되는 정보를 입력해준다.
    additional_info = {
        'input_patch_size': 96,
        'batch_size': 16
    }

    # <><><> 사용할 딥러닝 모델들을 불러온다.
    net_dict = {
        'G': net.GRDN(input_channel=3),  # yuv 444 형태라면 3 이여야 한다.
    }

    # 불러온 모델의 사이즈를 출력해본다.
    print(f'\n===> model size')
    for key in net_dict.keys():
        print(f'Number of params ({key}): {sum([p.data.nelement() for p in net_dict[key].parameters()])}')


    # DataLoader 을 만들어준다.
    def init_train_loader(num_workers):
        train_set = module_data.DatasetForDataLoader(
            hf_DB_dir,
            noise_type,
            additional_info=additional_info,
            median=median
        )

        # 학습 전에 항상 train_loader 을 초기화 해준다.
        return train_set, DataLoader(
            dataset=train_set,
            num_workers=num_workers,
            batch_size=additional_info['batch_size'],
            shuffle=True,
            drop_last=True,
        )


    # 찾은 num_workers 를 갖고 loader 을 만들어본다.
    train_set, train_loader = init_train_loader(num_workers=6)
    # 만든 DataLoader 가 잘 작동하는지 확인하고자, sample 을 하나 뽑아본다.
    # (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 참고)
    dataiter = iter(train_loader)
    batch_sample = dataiter.next()

    # 뽑은 sample 을 tile 형태로 저장해본다.
    print(f'\n===> sample from train_loader')
    for key in batch_sample.keys():
        print(f'{key}.shape : {batch_sample[key].shape}')
        npimg = np.transpose(batch_sample[key].numpy(), (0, 2, 3, 1))

        # tile 로 만들어준다. 아직 0~1 사이의 값이다.
        sample_size = math.ceil(additional_info['batch_size'] ** 0.5)
        img = utils.batch2one_img((sample_size, sample_size), npimg)

        cv2.imwrite(f'{exp_dir}/{key}_samples.png', img * 255)


    ####################################################################################
    ####################################################################################
    ####################################################################################


    # <><><> Training 중 특히 iteration 과 관련된 옵션을 설정해준다.
    train_scheduler = {
        # 몇 iteration 마다 모델(check point)을 저장 할 것인가. (영상 저장, psnr 측정 및 기록, best psnr 갱신, 모델 저장)
        'iter_save_model': 1000,

        # 몇 iteration 마다 validation 영상을 저장할 것인가. (영상만 딱 저장)
        'iter_saving_image': 1000,

        # 몇 iteration 마다 vtm encoder 를 이용해서 eval 을 할 것인가.
        'iter_run_vtm': 5000,

        # 총 몇 iteration 학습시킬 것인가.
        'iter_total': 5000 * 200,

        # learning rate decay 할때 얼만큼 할것인가.
        'decay_rate': 0.7,

        # 초기 learning rate
        'init_lr': 0.0001
    }

    iter_save_model = train_scheduler['iter_save_model']
    iter_total = train_scheduler['iter_total']
    epoch_total = iter_total//iter_save_model


    # training 시킬 객체 만들어주기.
    Train = module_train.TrainModule(cuda_num, DataParallel,
                                     edge_lambda=edge_lambda, color_lambda=color_lambda, sigma=sigma)

    # 초기 lr 설정해주기.
    Train.set_init_lr(init_lr=train_scheduler['init_lr'])

    # 아까 불러온 딥러닝 모델 넣어주기.
    Train.set_net(net_dict=net_dict)

    # 미리 저장되 checkpoint 가 있으면 불러오기.
    # 저장된게 없다면 0 부터 시작.
    # device setting 하고, optimizer setting 도 같이 해줌.
    iter_count, best_psnr = Train.weight_loader_netG(f'{exp_dir}/{checkpoint_version}')

    ####################################################################################
    # <><><> valid 해줄 객체를 초기화 해준다. (mode 1 일 경우)
    evals = module_eval.EvalModule(
        train_set=train_set,
        net_dict=net_dict,
        additional_info=additional_info,
        cuda_num=cuda_num,
        median=median
    )
    print(f'\n===> test set setting... (This may take some time.)')
    psnr_dict = evals.save_input_and_target(utils.make_dirs(f'{exp_dir}/evals'))

    ####################################################################################
    # <><><> valid 결과를 저장해줄 cvs 파일의 내용을 초기화 해준다.
    # training 중 추력할 log 들을 저장할 객체만들어 준다.
    psnrs_logger = utils.LogCSV(log_dir=f'{exp_dir}/{exp_name}_psnrs.csv')

    valid_names = []
    valid_psnrs_input_and_gt = []
    valid_psnrs_input_and_gt_new = []

    for psnr_dict_key in psnr_dict.keys():
        valid_names.append(psnr_dict_key)
        valid_psnrs_input_and_gt.append(psnr_dict[psnr_dict_key][0])
        valid_psnrs_input_and_gt_new.append(psnr_dict[psnr_dict_key][1])

    # 학습을 처음 시작할 때 csv 파일을 다음과 같이 초기화 해준다.
    if iter_count == 0:
        # 뽑은 제목들을 csv 파일에 적는다.
        psnrs_logger.make_head(['iter', 'lr'] + valid_names)

        # 해당 항목의 psnr 을 파일을 적는다.
        psnrs_logger(['input_and_gt', ''] + valid_psnrs_input_and_gt)
        psnrs_logger(['input_and_gt_new', ''] + valid_psnrs_input_and_gt_new)


    ####################################################################################
    ####################################################################################
    ####################################################################################
    print(f'\n===> train start <=======================================')

    stop = False

    while True:
        if iter_total <= iter_count:
            stop = True

        if stop:
            print(f'\n===> train finish!')
            break

        for batch in train_loader:
            ####################################################################################
            # 모델을 1 iteration 만큼 학습 시킨다.
            Train.optimize_parameters(batch)

            iter_count += 1

            # log 값들을 더하여 축척시켜 놓는다.
            Train.sum_logs_dict()

            ####################################################################################
            # 현재 상태 출력
            interval = 20
            if iter_count % interval == 0:
                # logs_dict = Train.get_logs_dict()
                # print(logs_dict)

                # 축척시킨 log 들을 interval 로 나눠준다. -> interval 동안의 평균 log 값을 얻는다.
                logs_dict_average = Train.get_logs_dict_average(interval)
                print(
                    f'cuda:{cuda_num} | '
                    f'exp_name:{exp_name} | '
                    f'epoch:{(iter_count-1)//iter_save_model + 1}/{epoch_total} | '
                    f'iter:{iter_count} | '
                    f'logs_average:{logs_dict_average}')

            ####################################################################################
            # 모델 저장하기
            if iter_count % iter_save_model == 0:
                ################################################################################
                # 정지 영상에대해 eval 하는 코드.
                print(f'\n===> Eval ({exp_name})')
                psnr_dict = evals.save_output(f'{exp_dir}/evals', iter_count)

                valid_psnrs = []

                for psnr_dict_key in psnr_dict.keys():
                    valid_psnrs.append(psnr_dict[psnr_dict_key])  # 각 항목의 psnr 값.


                # 해당 항목의 psnr 을 파일을 적는다.
                psnrs_logger([iter_count, Train.G_optimizer.param_groups[0]['lr']] +  # csv 파일에 iter 를 원래 써주던 곳에 그냥 lr 을 넣어줬다.
                             valid_psnrs)


                # 가장 최신 iter 에 해당하는 모델을 저장해준다.
                Train.weight_saver(f'{exp_dir}/checkpoint_{str(iter_count).zfill(6)}.pth', iter_count, best_psnr)


            ####################################################################################
            # 영상 저장해서 관찰할 수 있게 해주기. (iter_save_model == True 일때는 굳이 두번 일 안하도록 해야된다)
            if iter_count % train_scheduler['iter_saving_image'] == 0 and iter_count % iter_save_model != 0:
                print(f'\n===> Save valid images ({exp_name})\n')
                evals.save_output(f'{exp_dir}/evals', iter_count)

            ####################################################################################
            if iter_total <= iter_count:
                stop = True
                break



