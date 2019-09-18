import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from data import get_color_patch
from utils import save_image


def conv_modules():
    conv_s1 = nn.Conv2d(3, 3, 3, 1, 1, bias=False)
    conv_s1.weight.data.abs_()
    conv_s1.weight.data.div_(conv_s1.weight.data.sum(dim=[1, 2, 3], keepdim=True))
    conv_s2 = nn.Conv2d(3, 3, 3, 2, 1, bias=True)
    conv_s2.weight.data.abs_()
    conv_s2.weight.data.div_(conv_s2.weight.data.sum(dim=[1, 2, 3], keepdim=True))
    convs = {'conv_s1': conv_s1, 'conv_s2': conv_s2}
    return convs


def deconv_modules():
    deconv_s2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
    deconv_s2.weight.data.abs_()
    deconv_s2.weight.data.div_(deconv_s2.weight.data.sum(dim=[1, 2, 3], keepdim=True))
    deconvs = {'deconv_s2': deconv_s2}
    return deconvs


def main():
    convs = conv_modules()
    deconvs = deconv_modules()
    modules = {}
    modules.update(convs)
    modules.update(deconvs)
    data_dict = {'color_patch': get_color_patch()}
    for name, data in data_dict.items():
        for module_name, module in modules.items():
            input = data.permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                output = module(input)
            output = output.squeeze(0).permute(1, 2, 0)
            save_image('_'.join([name, module_name]), output.numpy())


if __name__ == "__main__":
    main()
