import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from resnet import resnet50


class CDataset(Dataset):
    def __init__(self, root):
        super(CDataset, self).__init__()
        self.root = root
        self.lists = []
        for item in os.listdir(root):
            if 'png' in item:
                self.lists.append(item)

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, idx):
        path = self.lists[idx]
        img = cv2.imread(os.path.join(self.root, path))
        label = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        H, W = label.shape[:2]
        if H > W:
            label = label.transpose(1, 0, 2)
        if H < 280 or W < 480:
            label = cv2.resize(label, (480, 280), interpolation=cv2.INTER_CUBIC)
        label = label[:280, :480]
        input = np.copy(label)
        for i in range(3):
            input[..., i] = cv2.equalizeHist(input[..., i])
        input = cv2.GaussianBlur(input, (7, 7), 0)

        input = torch.FloatTensor(input).permute(2, 0, 1).div_(255.)
        label = torch.FloatTensor(label).permute(2, 0, 1).div_(255.)

        return input, label, path


def save_image_batch(imgs, names, output_dir, prefix):
    for img, name in zip(imgs, names):
        path = os.path.join(output_dir, prefix + '_' + name)
        cv2.imwrite(path, img.data.cpu().permute(1, 2, 0).numpy()*255)


def main():
    iters = 3000 + 2
    lr = 1e-3
    feature_name = ['layer3']
    ckpt_path = '/mnt/lustre/share/DSK/model_zoo/pytorch/imagenet/resnet50-19c8e357.pth'
    #ckpt_path = '/mnt/lustre/niuyazhe/code/gitlab/imagenet-example/experiments/res50_batch2k_epoch100_avg_down_skipk3_32G_NEW/ckpt_best.pth'
    output_dir = 'baseline'
    data_path = '/mnt/lustre/niuyazhe/data/benchmark/DIV2K100/HR'
    dataset = CDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    model = resnet50()
    model.load_state_dict(torch.load(ckpt_path))
    model.cuda()
    model.eval()
    criterion = nn.L1Loss()

    for idx, data in enumerate(dataloader):
        input, label, path = data
        input, label = input.cuda(), label.cuda()
        save_image_batch(input, path, output_dir, 'input')
        save_image_batch(label, path, output_dir, 'label')
        input.requires_grad_(True)
        optimizer = torch.optim.Adam([input], lr)
        with torch.no_grad():
            label_feature = model.forward_feature(label, feature_name)
        for i in range(iters):
            input_feature = model.forward_feature(input, feature_name)
            loss = criterion(input_feature[0], label_feature[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('idx:{}\ti:{}\tloss:{}'.format(idx, i, loss.item()))
        save_image_batch(input, path, output_dir, 'result')
        break


if __name__ == '__main__':
    main()
