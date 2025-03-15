import torch
import argparse
from torch.utils.data import DataLoader
from torch import optim
import os
import numpy as np
import tqdm
import datetime
import math
from vali import val_multi
import glob
from load_dataset import Load_Dataset
from tensorboardX import SummaryWriter
import torch.nn as nn
from loss import CELDice
from model import MAMNet
from utils import visiul

device_ids = [0]
parse = argparse.ArgumentParser()
num_classes = 12
lra = 0.0001


def adjust_learning_rate(optimizer, epoch):
    lr = lra * (0.8 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_filename():
    train_file_names = glob.glob('F:\\MAMNet\\data\\kidney\\train\\image\\*.png')
    val_file_names = glob.glob('F:\\MAMNet\\data\\kidney\\vali\\image\\*.png')
    return train_file_names, val_file_names


def train():
    mod = MAMNet(num_classes=num_classes, pretrained=True)
    model = mod.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)
    model.train()
    batch_size = args.batch_size
    criterion = CELDice(0.2, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lra)
    train_file, val_file = load_filename()
    liver_dataset = Load_Dataset(train_file)
    val_dataset = Load_Dataset(val_file)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_load = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    train_model(model, criterion, optimizer, dataloaders, val_load, num_classes)


def train_model(model, criterion, optimizer, dataload, val_load, num_classes, num_epochs=200):
    loss_list = []
    dice_list = []
    Iou_dict_list = []
    dice_dict_list = []
    iou_list = []
    logs_dir = 'Logs\\T{}\\'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.mkdir(logs_dir)
    writer = SummaryWriter(logs_dir)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        tq = tqdm.tqdm(total=math.ceil(dt_size/args.batch_size))
        tq.set_description('Epoch {}'.format(epoch))
        epoch_loss = []
        step = 0
        for x, y, name in dataload:
            step += 1
            inputs = x.cuda(device_ids[0])
            y = y.long()
            labels = y.cuda(device_ids[0])
            optimizer.zero_grad()
            outputs = model(inputs)
            if epoch % 10 == 9 and step == 1:
                mode = 'labeled_train'
                train_path = 'F:\\MAMNet\\pic\\train'
                visiul(outputs, epoch, name, train_path, args.batch_size, mode)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tq.update(1)
            epoch_loss.append(loss.item())
            epoch_loss_mean = np.mean(epoch_loss).astype(np.float64)
            tq.set_postfix(loss='{0:.3f}'.format(epoch_loss_mean))
        loss_list.append(epoch_loss_mean)
        tq.close()
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss_mean))
        # 计算评估指标
        dice, iou, iou_dict, dices_dict = val_multi(model, criterion, val_load, num_classes,
                                                    args.batch_size, device_ids, epoch)
        writer.add_scalar('Loss', epoch_loss_mean, epoch)
        writer.add_scalar('Dice', dice, epoch)
        writer.add_scalar('IoU', iou, epoch)
        Iou_dict_list.append(iou_dict)
        dice_dict_list.append(dices_dict)
        dice_list.append([dice, iou])
        iou_list.append(iou)
        adjust_learning_rate(optimizer, epoch)
        fileObject = open(logs_dir + 'mean_LossList.txt', 'w')
        for ip in loss_list:
            fileObject.write(str(ip))
            fileObject.write('\n')
        fileObject.close()
        fileObject = open(logs_dir + 'dice_list.txt', 'w')
        for ip in dice_list:
            fileObject.write(str(ip))
            fileObject.write('\n')
        fileObject.close()
        fileObject = open(logs_dir + 'Iou_dict_List.txt', 'w')
        for ip in Iou_dict_list:
            fileObject.write(str(ip))
            fileObject.write('\n')
        fileObject.close()
        fileObject = open(logs_dir + 'dice_dict_List.txt', 'w')
        for ip in dice_dict_list:
            fileObject.write(str(ip))
            fileObject.write('\n')
        fileObject.close()
        if epoch % 10 == 9:
            torch.save(model.module.state_dict(), logs_dir + 'weights_{}.pth'.format(epoch))
    writer.close()
    return model


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--mode", type=bool, default=True)
    args = parse.parse_args()
    train()

