import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianBlur(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.weight = nn.Parameter(self._gaussian_kernel(self.kernel_size, self.sigma), requires_grad=False)

    def _gaussian_kernel(self, kernel_size, sigma):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*np.pi*variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) / \
                              (2*variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(self.channels, 1, 1, 1)

    def forward(self, x):
        # print(x.shape)
        # print(self.channels, self.weight, x)
        # print(self.weight.cuda().shape)
        out = F.conv2d(x, self.weight, stride=1, padding=int((self.kernel_size - 1)/2), groups=3)
        return out

class Sharpen(nn.Module):
    def __init__(self, sigma):
        super(Sharpen, self).__init__()
        self.sigma = sigma
        self.kernel = torch.stack([torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=torch.float32)])

    def forward(self, x):

        # Convolve the input tensor with the sharpen kernel
        kernel = self.kernel.to(x.device)
        kernel = kernel.unsqueeze(1).repeat(3, 1, 1, 1)
        # print(kernel)
        out = nn.functional.conv2d(x, kernel, bias=None, padding=1, stride=1, groups=3)

        out = self.sigma * torch.abs(out)

        return out
