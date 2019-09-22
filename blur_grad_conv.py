import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class BlurGrad(Function):

    @staticmethod
    def forward(ctx, x, gauss_kernel, padding, groups):
        ctx.gauss_kernel = gauss_kernel
        ctx.padding = padding
        ctx.groups = groups
        return x

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = F.pad(grad_out, ctx.padding, mode='constant')
        return F.conv2d(grad_in, ctx.gauss_kernel, stride=1, padding=0, groups=ctx.groups), None, None, None


class BlurGradConv(nn.Module):
    def __init__(self, gauss_kernel_size, conv):
        super(BlurGradConv, self).__init__()
        assert(gauss_kernel_size in [2, 3, 5])
        K = gauss_kernel_size
        self.K = K
        gauss_kernel_dict = {
            2: [1, 1],
            3: [1, 2, 1],
            5: [1, 4, 6, 4, 1]
        }
        pad_dict = {
            2: [0, 1, 0, 1],
            3: [1, 1, 1, 1],
            5: [2, 2, 2, 2],
        }
        self.pad = pad_dict[K]
        self.gauss = torch.FloatTensor(gauss_kernel_dict[K]).view(K, 1)
        self.gauss = self.gauss.matmul(self.gauss.view(1, K))
        self.gauss.div_(self.gauss.sum())
        self.gauss = self.gauss.view(1, 1, K, K).repeat(conv.in_channels, 1, 1, 1)
        self.gauss = nn.Parameter(self.gauss).requires_grad_(False)
        self.conv = conv
        self.blur_groups = conv.in_channels

    def forward(self, x):
        x = BlurGrad.apply(x, self.gauss, self.pad, self.blur_groups)
        x = self.conv(x)
        return x

    def __repr__(self):
        return self.conv.__repr__() + '\tblur_kernel_size={}'.format(self.K)


def test_blur_grad_conv():
    conv = nn.Conv2d(3, 6, 3, 1, 1)
    model = BlurGradConv(gauss_kernel_size=2, conv=conv).cuda()
    model.train()
    input = torch.randn(2, 3, 8, 8).cuda()
    output = model(input)
    loss = output.sum()
    loss.backward()
    print(model.gauss.requires_grad)
    print('end')


if __name__ == "__main__":
    test_blur_grad_conv()
