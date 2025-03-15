import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os


class CAE(nn.Module):
    def __init__(self, in_planes, out_palnes):
        super(CAE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_palnes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_palnes),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.max_pool(y)
        z = torch.mul(x, y)
        return z


class GFE(nn.Module):
    def __init__(self, in_planes):
        super(GFE, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(in_planes),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(in_planes),
                                   nn.ReLU(inplace=True)
                                   )
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.w_pool = nn.AdaptiveAvgPool2d((None, 1))

    def three_d_hadamard(self, x1, x2):
        w = x1.size()[-1]
        h = x2.size()[2]
        x1.repeat(1, 1, h, 1)
        x2.repeat(1, 1, 1, w)
        y = torch.mul(x1, x2)
        return y

    def forward(self, x):
        x1 = x
        x2 = x
        x1 = self.conv1(x1)
        x1 = self.h_pool(x1)
        x2 = self.conv2(x2)
        x2 = self.w_pool(x2)
        res = self.three_d_hadamard(x1, x2)
        return res


def visiul(outputs, epoch, name, path, batch_size, mode):
    outputs = outputs.permute(0, 2, 3, 1)
    for i in range(batch_size):
        image_name = name[i].split("\\")[-1]
        image_name = image_name.split(".")[0]
        res = outputs[i]
        res = res.data.cpu().detach().numpy().argmax(axis=2)
        res_gray = res * (255 / 11)
        sg_grayimage = Image.fromarray(np.uint8(res_gray)).convert('L')
        w, h = sg_grayimage.size
        sg_rgbimage = np.zeros((h, w, 3)).astype("uint8")
        for n in range(w):
            for m in range(h):
                if res[m, n] == 0:
                    sg_rgbimage[m, n, 0] = 0
                    sg_rgbimage[m, n, 1] = 0
                    sg_rgbimage[m, n, 2] = 0
                elif res[m, n] == 1:
                    sg_rgbimage[m, n, 0] = 0
                    sg_rgbimage[m, n, 1] = 255
                    sg_rgbimage[m, n, 2] = 0
                elif res[m, n] == 2:
                    sg_rgbimage[m, n, 0] = 0
                    sg_rgbimage[m, n, 1] = 255
                    sg_rgbimage[m, n, 2] = 255
                elif res[m, n] == 3:
                    sg_rgbimage[m, n, 0] = 125
                    sg_rgbimage[m, n, 1] = 255
                    sg_rgbimage[m, n, 2] = 12
                elif res[m, n] == 4:
                    sg_rgbimage[m, n, 0] = 255
                    sg_rgbimage[m, n, 1] = 55
                    sg_rgbimage[m, n, 2] = 0
                elif res[m, n] == 5:
                    sg_rgbimage[m, n, 0] = 24
                    sg_rgbimage[m, n, 1] = 55
                    sg_rgbimage[m, n, 2] = 125
                elif res[m, n] == 6:
                    sg_rgbimage[m, n, 0] = 187
                    sg_rgbimage[m, n, 1] = 155
                    sg_rgbimage[m, n, 2] = 25
                elif res[m, n] == 7:
                    sg_rgbimage[m, n, 0] = 0
                    sg_rgbimage[m, n, 1] = 255
                    sg_rgbimage[m, n, 2] = 125
                elif res[m, n] == 8:
                    sg_rgbimage[m, n, 0] = 255
                    sg_rgbimage[m, n, 1] = 255
                    sg_rgbimage[m, n, 2] = 125
                elif res[m, n] == 9:
                    sg_rgbimage[m, n, 0] = 123
                    sg_rgbimage[m, n, 1] = 15
                    sg_rgbimage[m, n, 2] = 175
                elif res[m, n] == 10:
                    sg_rgbimage[m, n, 0] = 124
                    sg_rgbimage[m, n, 1] = 155
                    sg_rgbimage[m, n, 2] = 5
                elif res[m, n] == 11:
                    sg_rgbimage[m, n, 0] = 12
                    sg_rgbimage[m, n, 1] = 255
                    sg_rgbimage[m, n, 2] = 141
                else:
                    sg_rgbimage[m, n, 0] = 0
                    sg_rgbimage[m, n, 1] = 0
                    sg_rgbimage[m, n, 2] = 0
        sg_rgbimage = Image.fromarray(np.uint8(sg_rgbimage)).convert('RGB')
        if mode == 'labeled_train':
            save_name = '{}_epoch{}_{}.png'.format(mode, epoch, image_name)
            save_path = os.path.join(path, save_name)
            sg_rgbimage.save(save_path)
        elif mode == 'unlabeled_train':
            save_name = '{}_epoch{}_{}.png'.format(mode, epoch, image_name)
            save_path = os.path.join(path, save_name)
            sg_rgbimage.save(save_path)
        elif mode == 'vali':
            save_name = '{}_epoch{}_{}.png'.format(mode, epoch, image_name)
            save_path = os.path.join(path, save_name)
            sg_rgbimage.save(save_path)


def visiul_test(outputs, name, path, batch_size, mode):
    outputs = outputs.permute(0, 2, 3, 1)
    for i in range(batch_size):
        image_name = name[i].split("\\")[-1]
        image_name = image_name.split(".")[0]
        res = outputs[i]
        res = res.data.cpu().detach().numpy().argmax(axis=2)
        res_gray = res * (255 / 11)
        sg_grayimage = Image.fromarray(np.uint8(res_gray)).convert('L')
        w, h = sg_grayimage.size  # 图像的尺寸
        sg_rgbimage = np.zeros((h, w, 3)).astype("uint8")
        for n in range(w):
            for m in range(h):
                if res[m, n] == 0:
                    sg_rgbimage[m, n, 0] = 0
                    sg_rgbimage[m, n, 1] = 0
                    sg_rgbimage[m, n, 2] = 0
                elif res[m, n] == 1:
                    sg_rgbimage[m, n, 0] = 0
                    sg_rgbimage[m, n, 1] = 255
                    sg_rgbimage[m, n, 2] = 0
                elif res[m, n] == 2:
                    sg_rgbimage[m, n, 0] = 0
                    sg_rgbimage[m, n, 1] = 255
                    sg_rgbimage[m, n, 2] = 255
                elif res[m, n] == 3:
                    sg_rgbimage[m, n, 0] = 125
                    sg_rgbimage[m, n, 1] = 255
                    sg_rgbimage[m, n, 2] = 12
                elif res[m, n] == 4:
                    sg_rgbimage[m, n, 0] = 255
                    sg_rgbimage[m, n, 1] = 55
                    sg_rgbimage[m, n, 2] = 0
                elif res[m, n] == 5:
                    sg_rgbimage[m, n, 0] = 24
                    sg_rgbimage[m, n, 1] = 55
                    sg_rgbimage[m, n, 2] = 125
                elif res[m, n] == 6:
                    sg_rgbimage[m, n, 0] = 187
                    sg_rgbimage[m, n, 1] = 155
                    sg_rgbimage[m, n, 2] = 25
                elif res[m, n] == 7:
                    sg_rgbimage[m, n, 0] = 0
                    sg_rgbimage[m, n, 1] = 255
                    sg_rgbimage[m, n, 2] = 125
                elif res[m, n] == 8:
                    sg_rgbimage[m, n, 0] = 255
                    sg_rgbimage[m, n, 1] = 255
                    sg_rgbimage[m, n, 2] = 125
                elif res[m, n] == 9:
                    sg_rgbimage[m, n, 0] = 123
                    sg_rgbimage[m, n, 1] = 15
                    sg_rgbimage[m, n, 2] = 175
                elif res[m, n] == 10:
                    sg_rgbimage[m, n, 0] = 124
                    sg_rgbimage[m, n, 1] = 155
                    sg_rgbimage[m, n, 2] = 5
                elif res[m, n] == 11:
                    sg_rgbimage[m, n, 0] = 12
                    sg_rgbimage[m, n, 1] = 255
                    sg_rgbimage[m, n, 2] = 141
                else:
                    sg_rgbimage[m, n, 0] = 0
                    sg_rgbimage[m, n, 1] = 0
                    sg_rgbimage[m, n, 2] = 0
        sg_rgbimage = Image.fromarray(np.uint8(sg_rgbimage)).convert('RGB')
        if mode == 'labeled_train':
            save_name = '{}_{}.png'.format(mode, image_name)
            save_path = os.path.join(path, save_name)
            sg_rgbimage.save(save_path)
        elif mode == 'unlabeled_train':
            save_name = '{}_{}.png'.format(mode, image_name)
            save_path = os.path.join(path, save_name)
            sg_rgbimage.save(save_path)
        elif mode == 'vali':
            save_name = '{}_{}.png'.format(mode, image_name)
            save_path = os.path.join(path, save_name)
            sg_rgbimage.save(save_path)

