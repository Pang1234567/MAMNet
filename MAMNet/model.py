from torch import nn
from torchvision import models
import torch.nn.functional as F
from utils import CAE, GFE
import torch


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x


class MAMNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.w = 512
        self.h = 640
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.local = CAE(512, 512)
        self.overall = GFE(512)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=2, padding=2, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 20)))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=5, padding=5, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 10)))
        self.conv_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=9, padding=9, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 5)))
        self.conv_4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # local
        e4 = self.local(e4)
        # overall
        x1 = self.overall(e4)
        x1 = torch.mul(e4, x1)

        x2 = self.conv_1(x1)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)

        x3 = self.conv_2(x1)
        x3 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=True)

        x4 = self.conv_3(x1)
        x4 = F.interpolate(x4, scale_factor=8, mode='bilinear', align_corners=True)

        res = torch.concat((x1, x2, x3, x4), dim=1)
        res = self.conv_4(res)

        d4 = self.decoder4(res)
        b4 = torch.concat((d4, e3), dim=1)
        b4 = self.conv1(b4)

        d3 = self.decoder3(b4)
        b3 = torch.concat((d3, e2), dim=1)
        b3 = self.conv2(b3)

        d2 = self.decoder2(b3)
        b2 = torch.concat((d2, e1), dim=1)
        b2 = self.conv3(b2)

        d1 = self.decoder1(b2)

        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        return x_out


