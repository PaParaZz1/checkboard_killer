import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cv2


class BlurGradResidual(Function):

    @staticmethod
    def forward(ctx, x, gauss_kernel, padding, groups):
        ctx.gauss_kernel = gauss_kernel
        ctx.padding = padding
        ctx.groups = groups
        return x

    @staticmethod
    def backward(ctx, grad_out):
        def viz(t):
            return (t - t.min()) / (t.max() - t.min() + 1e-12) * 255
        alpha = 0.2
        #cv2.imwrite('grad_out.png', viz(grad_out.data[0][0].abs().cpu().numpy()))
        grad_pad = F.pad(grad_out, [2, 2, 2, 2], mode='reflect')
        alpha_r = (grad_out - grad_pad[..., 4:, 2:-2]).abs()
        alpha_d = (grad_out - grad_pad[..., 2:-2, 4:]).abs()
        threshold = min(alpha_d.mean() + 2*alpha_d.std(), alpha_r.mean() + 2*alpha_r.std())
        alpha_d = torch.where(alpha_d > threshold, torch.ones_like(grad_out), torch.zeros_like(grad_out))
        alpha_r = torch.where(alpha_r > threshold, torch.ones_like(grad_out), torch.zeros_like(grad_out))
        alpha = alpha_d * alpha_r
        #alpha = F.pad(alpha, [0, 1, 0, 1], mode='reflect')
        #alpha = F.max_pool2d(alpha, kernel_size=2, stride=1)
        #alpha = (alpha_l.clamp(0, threshold) / threshold) * (alpha_r.clamp(0, threshold) / threshold)
        #cv2.imwrite('grad_grad.png', viz(alpha.data[0][0].abs().cpu().numpy()))
        #alpha = alpha.min(dim=1, keepdim=True)[0]
        #alpha1 = alpha ** 2
        alpha1 = 1 - alpha
        #cv2.imwrite('alpha.png', alpha.data[0][0].cpu().numpy()*255)
        grad_in = F.pad(grad_out, ctx.padding, mode='constant')
        grad_in = F.conv2d(grad_in, ctx.gauss_kernel, stride=1, padding=0, groups=ctx.groups)
        #grad_in = alpha1 * F.conv2d(grad_in, ctx.gauss_kernel, stride=1, padding=0, groups=ctx.groups) + alpha * grad_out
        return grad_in, None, None, None


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


class DownUpGrad(Function):

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_out):
        H, W = grad_out.shape[2:]
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            grad_out = F.pad(grad_out, [0, pad_h, 0, pad_w], mode='reflect')
        grad_in = F.avg_pool2d(grad_out, 2, 2)
        grad_in = F.interpolate(grad_in, scale_factor=2, mode='nearest')
        if pad_h or pad_w:
            grad_in = grad_in[..., :-1, :-1]
        return grad_in


class DownUpGradConv(nn.Module):
    def __init__(self, conv):
        super(DownUpGradConv, self).__init__()
        self.conv = conv

    def forward(self, x):
        x = DownUpGrad.apply(x)
        x = self.conv(x)
        return x

    def __repr__(self):
        return self.conv.__repr__() + '\tdown up grad'


class BlurGradConv(nn.Module):
    def __init__(self, gauss_kernel_size, conv, gauss_sigma=0):
        super(BlurGradConv, self).__init__()
        assert(gauss_kernel_size in [2, 3, 5])
        K = gauss_kernel_size
        self.K = K
        gauss_kernel_dict = {
            2: {0: [1, 1]},
            3: {0: [1, 2, 1], 0.5: cv2.getGaussianKernel(3, 0.5)},
            5: {0: [1, 4, 6, 4, 1]}
        }
        pad_dict = {
            2: [0, 1, 0, 1],
            3: [1, 1, 1, 1],
            5: [2, 2, 2, 2],
        }
        self.pad = pad_dict[K]
        self.gauss = torch.FloatTensor(gauss_kernel_dict[K][gauss_sigma]).view(K, 1)
        self.gauss = self.gauss.matmul(self.gauss.view(1, K))
        self.gauss.div_(self.gauss.sum())
        self.gauss = self.gauss.view(1, 1, K, K).repeat(conv.in_channels, 1, 1, 1)
        self.gauss = nn.Parameter(self.gauss).requires_grad_(False)
        self.conv = conv
        self.blur_groups = conv.in_channels

    def forward(self, x):
        x = BlurGradResidual.apply(x, self.gauss, self.pad, self.blur_groups)
        x = self.conv(x)
        return x

    def __repr__(self):
        return self.conv.__repr__() + '\tblur_kernel_size={}'.format(self.K)


def test_blur_grad_conv():
    conv = nn.Conv2d(3, 6, 3, 1, 1)
    model = BlurGradConv(gauss_kernel_size=3, conv=conv).cuda()
    model.train()
    input = torch.randn(2, 3, 8, 8).cuda()
    output = model(input)
    loss = output.sum()
    loss.backward()
    print(model.gauss)
    print('end')


if __name__ == "__main__":
    test_blur_grad_conv()
