import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_color_patch
from utils import save_image
import sys
import os
sys.path.append('../CARAFE_pytorch')
from CARAFE_downsample import  CarafeDownsample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv1 = nn.Conv2d(3, 3, 3, 2, 1, bias=False)
        self.conv1.weight.data = torch.ones(3, 3, 3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.mp = nn.MaxPool2d(2, 2)
        self.ap = nn.AvgPool2d(2, 2)
        self.carafe_down = CarafeDownsample(3, scale_factor=2, m_channels=3)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.mp(x)
        #x = self.carafe_down(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv2(x)
        return x

    def forward_viz_grad(self, x):
        self.grad = {}

        def save_grad(name):
            def hook(grad):
                self.grad[name] = grad
            return hook

        x1 = self.conv1(x)
        #x1 = self.mp(x)
        x2 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        x3 = self.conv2(x2)
        x.register_hook(save_grad('x'))
        x1.register_hook(save_grad('x1'))
        #x2.register_hook(save_grad('x2'))
        #x3.register_hook(save_grad('x3'))
        return x3


def grad_viz():
    ITER = 500
    LR = 1e-2
    OUTPUT_DIR = 'conv_s2_no_bias_1init/'
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
        print('idx: {}\tloss: {}'.format(idx, loss.item()))
        if idx % 10 == 0:
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


if __name__ == "__main__":
    grad_viz()
