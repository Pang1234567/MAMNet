import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image as Image

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
y_trans = transforms.ToTensor()


class Load_Dataset(Dataset):
    def __init__(self, filenames):
        self.file_names = filenames

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        down_sample = 1
        img_file_name = self.file_names[idx]
        ori_image = load_image(img_file_name)
        image = x_transforms(ori_image)
        mask = load_mask(img_file_name)
        labels = torch.from_numpy(mask).float()
        return image, labels, img_file_name


def load_image(path):
    img_x = Image.open(path)
    return img_x


def load_mask(path):
    new_path = path.replace('image', 'labels')
    mask = cv2.imread(new_path, 0)
    mask = mask//20
    return mask.astype(np.uint8)

