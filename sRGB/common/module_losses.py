import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self, cuda_num):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cuda_num}')
            self.kernel = self.kernel.to(self.device)
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


def _gaussian_kernel1d(sigma=2, truncate=3):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    radius = int(truncate * sd + 0.5)

    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    return phi_x


class ColorLoss(nn.Module):
    def __init__(self, cuda_num, sigma=4):
        super(ColorLoss, self).__init__()
        # get kernel
        # Since we are calling correlate, not convolve, revert the kernel
        weights = _gaussian_kernel1d(sigma=sigma)[::-1]
        k = torch.Tensor([weights])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cuda_num}')
            self.kernel = self.kernel.to(self.device)
        self.loss = CharbonnierLoss()

    def conv_guss_with_sigma(self, img):
        # padding
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')

        # "groups == n_channels", because each channel should be seperated.
        return F.conv2d(img, self.kernel, groups=n_channels)

    def forward(self, x, y):
        loss = self.loss(self.conv_guss_with_sigma(x), self.conv_guss_with_sigma(y))
        return loss


if __name__ == "__main__":

    # get kernel
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma=10, truncate=6)[::-1]
    k = torch.Tensor([weights])
    kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)

    kernel_to_write = kernel[0][0].numpy()
    kernel_to_write = kernel_to_write * (1/np.max(kernel_to_write)) * 255
    cv2.imwrite('kernel.png', kernel_to_write)

    print('------')
    print(kernel.shape)

    # read image
    img_sample = cv2.imread('/hdd1/works/projects/human_and_forest/MPRNet/noisy_style_transformer_train/sample_test_imgs/D-210814_O9112R03_006_0033.jpg')
    img_sample = torch.from_numpy(img_sample.transpose((2, 0, 1))).unsqueeze(0).float()
    print(img_sample.shape)

    # padding
    n_channels, _, kw, kh = kernel.shape
    img_sample = F.pad(img_sample, (kw//2, kh//2, kw//2, kh//2), mode='replicate')

    # "groups == n_channels", because each channel should be seperated.
    my_result = F.conv2d(img_sample, kernel, groups=n_channels)
    print(my_result.shape)

    # torch to numpy
    my_result = my_result.squeeze().numpy()
    my_result = my_result.transpose((1,2,0))
    print(my_result.shape)

    # write result
    cv2.imwrite('blured.png', my_result)

