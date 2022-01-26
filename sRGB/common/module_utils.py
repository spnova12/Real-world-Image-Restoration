import os
import cv2
import torch
import csv
import numpy as np
from math import log10
import time


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # png 읽을때 경고문 없애줌.


# def load_BGR(filepath):
#     img_BGR = cv2.imread(filepath, cv2.IMREAD_COLOR)
#     return img_BGR
#
#
# def load_grayscale(filepath):
#     img_grayscale = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#     return img_grayscale
#
#
# def load_pil(filepath):
#     # 가끔 흑백이미지가 섞이는 것을 방지하기 위해 'RGB' 로 변화함을 명시해줌.
#     img_pil = Image.open(filepath).convert('RGB')
#     return img_pil


def get_psnr(img1, img2, bitDepth=8):
    """
    psnr 을 계산해준다.
    A description of these metrics follows.
    JVET-V2016 : JVET common test conditions and evaluation procedures for neural network-based video coding technology
    (https://jvet-experts.org/)
    """
    if type(img1) == torch.Tensor:
        mse = torch.mean((img1 - img2) ** 2)
    else:
        mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * log10(((255<<(bitDepth-8))**2) / mse)


def make_dirs(path):
    """
    경로(폴더) 가 있음을 확인하고 없으면 새로 생성한다.
    :param path: 확인할 경로
    :return: path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def wait_file_writing(filename):
    print('wait_file_writing...')
    max_i = 10
    time.sleep(2)

    for i in range(max_i):
        try:
            with open(filename, 'rb') as _:
                break
        except IOError:
            time.sleep(1)
    else:
        raise IOError('Could not access {} after {} attempts'.format(filename, str(max_i)))


def np_random_rotate(input, mode):
    # rot90 설명 : https://stackoverflow.com/questions/63972190/understanding-numpy-rot90-axes
    # mode = np.random.randint(4)
    input = np.rot90(input, mode)
    return input


def np_random_flip(input, mode):
    # Flip array in the left/right direction.
    # mode = np.random.randint(2)
    if mode == 1:
        input = np.flip(input)
    return input


# def modcrop(img_in, scale):
#     mod = scale
#     w, h = img_in.size
#     j, i, w, h = 0, 0, w - w % mod, h - h % mod
#     img = img_in.crop((j, i, j + w, i + h))
#
#     return img


class LogCSV(object):
    def __init__(self, log_dir):
        """
        :param log_dir: log(csv 파일) 가 저장될 dir
        """
        self.head = False
        self.log_dir = log_dir
        f = open(self.log_dir, 'a')
        f.close()

    def make_head(self, header):
        """
        As of Python 3.6, for the CPython implementation of Python,
        dictionaries maintain insertion order by default.
        dict 에 key 생성한 순서가 그대로 유지됨을 확인.
        """
        self.head = True
        with open(self.log_dir, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(header)

    def __call__(self, log):
        """
        :param log: header 의 각 항목에 해당하는 값들의 list
        """
        with open(self.log_dir, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(log)


class TorchPaddingForOdd(object):
    """
    1/(2^downupcount) 크기로 Down-Sampling 하는 모델 사용시 이미지의 사이즈가 홀수 또는 특정 사이즈일 경우
    일시적으로 padding 을 하여 짝수 등으로 만들어 준 후 모델을 통과시키고,
    마지막으로 unpadding 을 하여 원래 이미지 크기로 만들어준다.
    """
    def __init__(self, downupcount=1, scale_factor=1):
        self.is_height_even = True
        self.is_width_even = True

        self.scale_factor = scale_factor
        self.downupcount = 2 ** downupcount
        self.pad1 = None
        self.pad2 = None

    def padding(self, img):
        # 홀수면 패딩을 체워주는 것을 해주자
        if img.shape[2] % self.downupcount != 0:
            self.is_height_even = False
            self.pad1 = (img.shape[2]//self.downupcount + 1) * self.downupcount - img.shape[2]
            img_ = torch.zeros(img.shape[0], img.shape[1], img.shape[2] + self.pad1, img.shape[3])
            img_[:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3]] = img
            for i in range(self.pad1):
                img_[:img.shape[0], :img.shape[1], img.shape[2] + i, :img.shape[3]] = img_[:img.shape[0], :img.shape[1], img.shape[2] - 1, :img.shape[3]]
            img = img_
        if img.shape[3] % self.downupcount != 0:
            self.is_width_even = False
            self.pad2 = (img.shape[3] // self.downupcount + 1) * self.downupcount - img.shape[3]
            img_ = torch.zeros(img.shape[0], img.shape[1], img.shape[2], img.shape[3] + self.pad2)
            img_[:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3]] = img
            for i in range(self.pad2):
                img_[:img.shape[0], :img.shape[1], :img.shape[2], img.shape[3] + i] = img_[:img.shape[0], :img.shape[1], :img.shape[2], img.shape[3] - 1]
            img = img_
        return img

    def unpadding(self, img):
        # 홀수였으면 패딩을 제거하는 것을 해주자
        if not self.is_height_even:
            img.data = img.data[:img.shape[0], :img.shape[1], :img.shape[2] - self.pad1 * self.scale_factor, :img.shape[3]]
        if not self.is_width_even:
            img.data = img.data[:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3] - self.pad2 * self.scale_factor]
        return img


def batch2one_img(size, images):
    """
    numpy의 batch 를 타일형태의 한장의 이미지로 만들어준다.
    size: (a, b) 형태의 튜플 a = 세로 타일 개수, b = 가로 타일 개수.
    images: input image 의 shape 은 (batch, h, w, channel) 이다.
    :return: color 일 경우 (h, w, 3), 흑백일 경우 (h, w) 인 한장의 이미지.
    """
    h, w = images.shape[1], images.shape[2]
    # color
    if len(images.shape) == 4:
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]  # 나누기 연산 후 몫이 아닌 나머지를 구함
            j = idx // size[1]  # 나누기 연산 후 소수점 이하의 수를 버리고, 정수 부분의 수만 구함
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    # gray scale
    elif len(images.shape) == 3:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')