import torch
import argparse
from torch.utils.data import DataLoader
from load_dataset import Load_Dataset
from vali import val_multi_test
import glob
from loss import CELDice
from model import MAMNet

device_ids = [0]
parse = argparse.ArgumentParser()
weight_load = 'F:\\MAMNet\\Logs\\weights_299.pth'
vali_path = 'F:\\MAMNet\\test data\\subset'
num_classes = 12


def test():
    model = MAMNet(num_classes=num_classes, pretrained=True)
    device = torch.device("cpu")
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight_load, map_location=device).items()})
    model = model.cuda(device_ids[0])
    model.eval()
    criterion = CELDice(0.2, num_classes=num_classes)
    test_file_names = glob.glob('F:\\MAMNet\\test data\\subset\\image\\*.png')
    test_dataset = Load_Dataset(test_file_names)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True,
                             num_workers=0, drop_last=True)
    average_dices, average_iou, _, _ = val_multi_test(model, criterion, test_loader, num_classes,
                                                      batch_size=2, device_ids=device_ids, vali_path=vali_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type=int, default=2)
    args = parse.parse_args()
    test()
