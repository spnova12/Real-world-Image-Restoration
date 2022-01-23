
import numpy as np
import cv2
import os
import tqdm
import common.module_utils as utils

import torch
import torchvision.transforms as transforms



def recon(input_batch_tensor_list, net, scale_factor, odd, device):
    """
    실제로 딥러닝 모델을 이용해서 inference 해주는 함수.
    """
    # network 에 downscaling 부분이 있으면 영상 사이즈가 downscaling 하는 수 만큼 영상에 padding 을 해줘야 한다.
    # padding 이 된 영상이 network 를 통과한 후 padding 을 지워준다.
    pad = utils.TorchPaddingForOdd(odd, scale_factor=scale_factor)

    for i, input_batch_tensor in enumerate(input_batch_tensor_list):
        input_batch_tensor_list[i] = pad.padding(input_batch_tensor).to(device)

    with torch.no_grad():
        net.eval()

        batch_tensor_out = net(*input_batch_tensor_list)

        # MPRnet's out is list. out[0] is the final output.
        if isinstance(batch_tensor_out, list):
            batch_tensor_out = batch_tensor_out[0]

        batch_tensor_out = pad.unpadding(batch_tensor_out)

        batch_tensor_out = batch_tensor_out.cpu()
        return batch_tensor_out


def recon_one_frame(input_img_list, net, device, scale_factor, downupcount):
    """
    한 frame 을 복원하는 함수이다.
    입력 bit 를 고려하여 0~1 사이로 normalize 해준 후 복원 함수를 이용해서 복원한다.
    복원된 후에는 원래 범위로 다시 normalize 해준다.
    그리고 rounding, clipping, 형 변환 까지 한 영상을 반환해준다.
    """
    def np_to_torch(input_img):
        # tensor로 바꿔주기.
        input_img = torch.from_numpy(np.array(input_img, np.float32, copy=False))
        input_img = input_img.permute((2, 0, 1)).contiguous()
        # conv2d 를 하려면 input 이 4 channel 이어야 한다!
        input_img = input_img.view(1, -1, input_img.shape[1], input_img.shape[2])
        return input_img

    input_tensor_list = []

    for input_img in input_img_list:
        input_tensor_list.append(np_to_torch(input_img))

    # 복원하기.
    output_img = recon(input_tensor_list, net, scale_factor, downupcount, device)

    # 복원하기 직적에 4 channel 로 만들었으니까 다시 3 channel 로 만들어 줘야 한다.
    output_img = output_img.view(-1, output_img.shape[2], output_img.shape[3])

    # 영상 후처리.
    npimg = output_img.numpy()
    npimg = npimg.clip(0, 1)
    npimg = npimg.transpose((1, 2, 0))

    return npimg


def recon_big_one_frame(input_big_list, wh, net, scale_factor, minimum_wh, device, odd=2):
    """
        큰 영상들을 분할하여 recon 하는 함수.
        input_bit is list
        """

    # if it is not list then change it to the list.
    input_big_list = input_big_list if isinstance(input_big_list, list) else [input_big_list]

    # 딥러닝에 모델을 넣을때 가장 먼저 할 일은 shape 를 3 channel 로 만들어 주는 것!
    if len(input_big_list[0].shape) == 2:
        input_big_list[0] = np.expand_dims(input_big_list[0], axis=0)

    ori_w_length, ori_h_length = wh

    img_out_np = np.zeros_like(input_big_list[0], dtype=np.float32)

    # 분할할 사이즈를 계산해준다.
    # print('분할 사이즈 계산하는 중...')
    w_length = ori_w_length
    h_length = ori_h_length
    w_split_count = 0
    h_split_count = 0

    while w_length > minimum_wh and h_length > minimum_wh:
        w_split_count += 1
        h_split_count += 1

        w_length = ori_w_length//w_split_count
        h_length = ori_h_length//h_split_count

    w_pos = 0
    h_pos = 0

    w_count = 0
    h_count = 0
    total_count = 0
    while ori_w_length - w_pos >= w_length:
        w_count += 1
        while ori_h_length - h_pos >= h_length:
            total_count += 1
            # print(f"{total_count}/{w_split_count * h_split_count} Forward Feeding...")
            h_count += 1

            wl = w_length
            hl = h_length

            if w_pos + w_length*2 > ori_w_length:
                wl = ori_w_length - w_pos
            if h_pos + h_length*2 > ori_h_length:
                hl = ori_h_length - h_pos

            j, i, w, h = w_pos, h_pos, wl, hl

            cropped_input_list = []
            for input_big in input_big_list:
                cropped_input_list.append(input_big[i:(i+h), j:(j + w)])

            img_out_np[i:(i + h), j:(j + w)] \
                = recon_one_frame(cropped_input_list,
                                  net, device, scale_factor=scale_factor, downupcount=odd)  # 복원 영상이 np 이다.

            h_pos += h_length  # //2

        h_pos = 0
        h_count = 0
        w_pos += w_length  # //2


    return img_out_np


######################################################################################################################
######################################################################################################################
######################################################################################################################


# <><><> 사용할 net architecture 선택하기.
import common_net.GRDN as net


class NetForInference(object):
    """
    학습이 다 끝난 후 모델을 inference 할 때 쓰이는 class.
    """
    def __init__(self, input_channel, device):
        # 사용할 gpu 번호.
        self.device = device

        # 사용할 딥러닝 모델들을 불러온다.
        net_dict = {'G': net.GRDN(input_channel=input_channel)}

        # 불러온 모델의 사이즈를 출력해본다.
        print(f'\n===> Model size')
        for key in net_dict.keys():
            print(f'Number of params ({key}): {sum([p.data.nelement() for p in net_dict[key].parameters()])}')

        self.netG = net_dict['G'].to(self.device)

    def weight_loader(self, load_checkpoint_dir):
        # checkpoint 를 load 해줌.
        print(f"\n===> Load checkpoint")
        if os.path.isfile(load_checkpoint_dir):
            print(": loading checkpoint '{}'".format(load_checkpoint_dir))
            checkpoint = torch.load(load_checkpoint_dir)
            iter_count = checkpoint['iter_count']
            best_psnr = checkpoint['best_psnr']
            self.netG.load_state_dict(checkpoint['G'])
        else:
            print(": no checkpoint found at '{}'".format(load_checkpoint_dir))




