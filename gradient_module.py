import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_color_patch, get_color
from utils import save_image
import os
import cv2
from torch.autograd import Function
from resnet import resnet50
from sp_op_conv import *


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
    def __init__(self, mode='normal'):
        super(Net, self).__init__()
        if mode == 'normal':
            self.conv1 = nn.Conv2d(3, 3, 3, 2, 1)
        elif mode == 'blur':
            self.conv1 = BlurGradConv(5, nn.Conv2d(3, 3, 3, 2, 1))
        elif mode == 'mask':
            self.conv1 = MaskNormConv(nn.Conv2d(3, 3, 3, 2, 1))
        elif mode == 'shift_expand':
            self.conv1 = ShiftConv(nn.Conv2d(3, 3, 3, 2, 1), 'expand')
        elif mode == 'shift_divide':
            self.conv1 = ShiftConv(nn.Conv2d(3, 3, 3, 2, 1), 'divide')
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
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
                self.gauss = self.gauss.to(grad_in.device)
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
        x1 = self.conv1(x)
        #x1 = self.mp(x)
        x2 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        x3 = self.conv2(x2)
        self.handle['x'] = x.register_hook(blur_grad('x'))
        self.handle['x1'] = x1.register_hook(save_grad('x1'))
        #x2.register_hook(save_grad('x2'))
        #x3.register_hook(save_grad('x3'))
        return x3


class TinyResNet(nn.Module):
    def __init__(self):
        super(TinyResNet, self).__init__()
        self.conv1 = BlurGradConv(3, nn.Conv2d(3, 3, 7, 2, 3, bias=False))
        self.avgpool = nn.AvgPool2d(2, 2)
        self.c11 = nn.Conv2d(3, 3, 1, 1, 0)
        self.c12 = BlurGradConv(3, nn.Conv2d(3, 3, 3, 2, 1))
        #self.c12 = nn.Conv2d(4, 4, 3, 2, 1)
        self.c13 = nn.Conv2d(3, 3, 1, 1, 0)
        #self.c1r = nn.Conv2d(8, 8, 1, 2, 0)
        #self.c1r = BlurGradConv(3, nn.Conv2d(8, 8, 1, 2, 0))

    def forward(self, x, viz=True):
        self.grad = {}
        self.handle = {}

        def save_grad(name):
            def hook(grad):
                self.grad[name] = grad
            return hook

        x0 = self.conv1(x)
        #x = self.avgpool(x)
        r0 = x0
        x1 = self.c11(x0)
        x2 = self.c12(x1)
        x3 = self.c13(x2)
        r1 = self.avgpool(r0)
        if viz:
            #self.handle['r1'] = r1.register_hook(save_grad('r1'))
            self.handle['x0'] = x0.register_hook(save_grad('x0'))
            self.handle['x'] = x.register_hook(save_grad('x'))

        #r1 = self.c1r(r1)
        y = x3 + r1
        #x = F.relu(x + r1)
        return y


def grad_viz():
    ITER = 500
    LR = 3e-4
    OUTPUT_DIR = '0900_exp_shift_divide/'
    PATH = '0900_x4_HR.png'
    viz_internal_grad = False
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    label = cv2.imread(PATH)
    label = cv2.resize(label, (0,0), fx=0.25, fy=0.25)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

    #input = torch.ones_like(label)
    input = cv2.GaussianBlur(label, (7, 7), 0.5)
    save_image(OUTPUT_DIR + 'input.png', input)
    save_image(OUTPUT_DIR + 'label.png', label)
    label = torch.FloatTensor(label).permute(2, 0, 1).unsqueeze(0).div_(255.).cuda()
    input = torch.FloatTensor(input).permute(2, 0, 1).unsqueeze(0).div_(255.).cuda()
    input.requires_grad_(True)
    net = Net(mode='shift_divide')
    #net = resnet50()
    #net.load_state_dict(torch.load('/mnt/lustre/niuyazhe/code/github/RCAN/RCAN_TrainCode/code/resnet50-19c8e357_grad_op.pth'), strict=False)
    net.eval()
    net.cuda()
    optimizer = torch.optim.Adam([input], lr=LR)
    criterion = nn.MSELoss()

    for idx in range(ITER):
        #input_feature = net.forward_viz_grad(input)
        label_feature = net(label)
        input_feature = net(input)
        loss = criterion(input_feature, label_feature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('idx: {}\tloss: {}'.format(idx, loss.item()))
        if idx % 10 == 0:
            if viz_internal_grad:
                #H, W = net.grad['x1'].shape[2:]
                #x1_g = torch.zeros(1, 3, 2*H, 2*W)
                #x1_g[..., ::2, ::2] = net.grad['x1'][..., :, :]
                #net.grad['est'] = F.conv2d(x1_g, net.conv1.weight, stride=1, padding=1)
                for k, v in net.grad.items():
                    grad = v.data.squeeze(0).permute(1, 2, 0)
                    grad = grad.abs_()
                    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-12) * 255.
                    save_image(OUTPUT_DIR + 'grad_{}_{}'.format(k, idx), grad.numpy())
            grad = input.grad.data.squeeze(0).permute(1, 2, 0)
            grad = grad.abs_()
            grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-12) * 255.
            save_image(OUTPUT_DIR + 'input_grad_{}'.format(idx), grad.cpu().numpy())
            save_image(OUTPUT_DIR + 'input_{}'.format(idx), input.clone().data.squeeze(0).permute(1, 2, 0).mul_(255.).clamp(0, 255).cpu().numpy())


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
