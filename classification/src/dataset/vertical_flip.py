from typing import Dict, List

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VerticalFlipClassificationDataset(Dataset):
    """
    Vertical flip classification dataset.
    categories = ["normal", "vflip"]
    Args:
        data_list example: [
            {
                "data_path": "/dataset/raw/0914/台胞證/20569495559749.jpg",
                "gt_cls": 0
            }, ...
        ]
    """

    def __init__(self, data_list: List[Dict], img_size: int, do_flip: bool = True):
        if do_flip:
            self.transform = [self.get_transform(img_size, False), self.get_transform(img_size, True)]
        else:
            self.transform = [self.get_transform(img_size, False)]

        self.img_path_list = []
        self.labels = []
        self.transform_indices = []
        for data_dict in data_list:
            if do_flip:
                for i in range(2):
                    self.img_path_list.append(data_dict["data_path"])
                    self.labels.append((data_dict["gt_cls"] + i) % 2)
                    self.transform_indices.append(i)
            else:
                self.img_path_list.append(data_dict["data_path"])
                self.labels.append(data_dict["gt_cls"])
                self.transform_indices.append(0)

    @staticmethod
    def get_transform(img_size: int, flip: bool, use_color_jitter: bool = False):
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomVerticalFlip(p=1.0) if flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ]
        if use_color_jitter:
            transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        """
        Returns:
            x (torch.Tensor float): image, shape = (3, h, w)
            y (torch.Tensor int): class index, shape = ()
        """
        x = Image.open(self.img_path_list[index]).convert("RGB")
        x = ImageOps.exif_transpose(x)  # Must! Prevent default orientation affects label
        x = self.transform[self.transform_indices[index]](x)  # (h, w, c) -> (c, h, w)
        x = x.type(torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y
    