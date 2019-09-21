import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_color_patch
from utils import save_image
#from blur_grad_conv2d import BlurGradConv2d
import sys
import os
sys.path.append('../CARAFE_pytorch')
from CARAFE_downsample import  CarafeDownsample
from torch.autograd import Function


class NoiseMaxPool(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoiseMaxPool, self).__init__()
        self.mp = nn.MaxPool2d(*args, **kwargs)

    def forward(self, x):
        min_val = x.clone().abs().min() * 1e-3
        return self.mp(x + min_val * torch.randn_like(x))


class _RectifiedMaxPool2d(Function):

    @staticmethod
    def forward(ctx, x, kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        assert(kernel_size == stride)
        assert(not return_indices)
        y = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                         return_indices=return_indices, ceil_mode=ceil_mode)
        y_dup = F.interpolate(y, scale_factor=stride, mode='nearest')
        ctx.max_location = x.eq(y_dup)
        ctx.stride = stride
        return y

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = F.interpolate(grad_out, scale_factor=ctx.stride, mode='nearest')
        ctx.max_location = torch.tensor(ctx.max_location.clone().detach(), device=grad_out.device, dtype=grad_out.dtype)
        return ctx.max_location * grad_in, None, None


class RectifiedMaxPool2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RectifiedMaxPool2d, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return _RectifiedMaxPool2d.apply(x, *self.args, **self.kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv_mp = nn.Conv2d(3, 3, 3, 1, 1)
        self.gauss = torch.FloatTensor([1, 2, 1]).view(3, 1)
        self.gauss = self.gauss.matmul(self.gauss.view(1, 3))
        self.gauss.div_(self.gauss.sum())
        self.gauss = self.gauss.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.conv1 = nn.Conv2d(3, 3, 3, 2, 1)
        #self.conv1 = nn.Conv2d(3, 3, 4, 2, [1, 2], bias=False)
        #self.conv1.weight.data = torch.ones(3, 3, 4, 4)
        #self.conv1.weight.data = torch.FloatTensor([1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]).view(1, 1, 4, 4).repeat(3, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        #self.mp = nn.MaxPool2d(2, 2)
        #self.mp = NoiseMaxPool(2, 2)
        self.mp = RectifiedMaxPool2d(2, 2)
        self.ap = nn.AvgPool2d(2, 2)
        self.carafe_down = CarafeDownsample(3, scale_factor=2, m_channels=3)

    def forward(self, x):
        #x = self.conv_mp(x)
        #x = self.conv1(x)
        x = self.mp(x)
        #x = self.carafe_down(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv2(x)
        return x

    def forward_viz_grad(self, x):
        self.grad = {}
        self.handle = {}

        def save_grad(name):
            def hook(grad):
                self.grad[name] = grad
            return hook

        def blur_grad(name):
            def hook(grad_in):
                grad_in = F.conv2d(grad_in, self.gauss, stride=1, padding=1, groups=3)
                self.grad[name] = grad_in
                return grad_in
            return hook

        def down_up_grad(name):
            def hook(grad_in):
                grad_in = F.avg_pool2d(grad_in, kernel_size=2, stride=2)
                grad_in = F.interpolate(grad_in, scale_factor=2, mode='nearest')
                self.grad[name] = grad_in
                return grad_in
            return hook

        #x = self.conv_mp(x)
        #x1 = self.conv1(x)
        x1 = self.mp(x)
        x2 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        x3 = self.conv2(x2)
        self.handle['x'] = x.register_hook(save_grad('x'))
        self.handle['x1'] = x1.register_hook(save_grad('x1'))
        #x2.register_hook(save_grad('x2'))
        #x3.register_hook(save_grad('x3'))
        return x3


def grad_viz():
    ITER = 500
    LR = 1e-2
    OUTPUT_DIR = 'mp_random_noise_blur/'
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    label = get_color_patch()
    label = label.permute(2, 0, 1).unsqueeze(0).div_(255.)
    input = torch.ones(*label.shape)
    input.requires_grad_(True)
    net = Net()
    net.eval()
    if torch.cuda.is_available():
        input = input.cuda()
        label = label.cuda()
        net.cuda()
    optimizer = torch.optim.Adam([input], lr=LR)
    criterion = nn.MSELoss()

    for idx in range(ITER):
        input_feature = net.forward_viz_grad(input)
        label_feature = net(label)
        loss = criterion(input_feature, label_feature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for k, v in net.handle.items():
            v.remove()
        print('idx: {}\tloss: {}'.format(idx, loss.item()))
        if idx % 10 == 0:
            H, W = net.grad['x1'].shape[2:]
            x1_g = torch.zeros(1, 3, 2*H, 2*W)
            x1_g[..., ::2, ::2] = net.grad['x1'][..., :, :]
            net.grad['est'] = F.conv2d(x1_g, net.conv1.weight, stride=1, padding=1)
            for k, v in net.grad.items():
                grad = v.data.squeeze(0).permute(1, 2, 0)
                grad = grad.abs_()
                grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-12) * 255.
                save_image(OUTPUT_DIR + 'grad_{}_{}'.format(k, idx), grad.numpy())
            grad = input.grad.data.squeeze(0).permute(1, 2, 0)
            grad = grad.abs_()
            grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-12) * 255.
            save_image(OUTPUT_DIR + 'input_grad_{}'.format(idx), grad.numpy())
            save_image(OUTPUT_DIR + 'input_{}'.format(idx), input.clone().data.squeeze(0).permute(1, 2, 0).mul_(255.).clamp(0, 255).numpy())

    if torch.cuda.is_available():
        input = input.cpu()
    save_image(OUTPUT_DIR + 'grad', input.data.squeeze(0).permute(1, 2, 0).mul_(255.).clamp(0, 255).numpy())


def test_rectified_max_pool():
    m = RectifiedMaxPool2d(2, 2)
    input = torch.FloatTensor([1 for _ in range(16)]).view(1, 1, 4, 4).requires_grad_(True)
    output = m(input)
    loss = output.mean()
    loss.backward()
    print(input.grad)


if __name__ == "__main__":
    grad_viz()
    #test_rectified_max_pool()
