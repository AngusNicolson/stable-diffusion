
import json
from typing import Union, Any, Tuple, Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import albumentations


class IntergrowthDataset(Dataset):
    """Intergrowth dataset"""

    def __init__(self, json_file: Union[str, dict], split: str = "train", prefix: str = None, transform=None, load_img=True, size=256, downscale_f=4, min_crop_f=0.5, max_crop_f=1.0):
        """
        Args:
            json_file (string): Path to the json metadata file.
            prefix (string, optional): Add a prefix to file paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            split (string): Which data split to load
        """
        self.split = split
        self.load_img = load_img
        self.transform = transform
        self.metadata_labels = ["age_label"]
        self.size = size
        self.downscale_f = downscale_f
        assert (size / downscale_f).is_integer()
        self.lr_size = int(size / downscale_f)
        self.image_rescaler = albumentations.SmallestMaxSize(
            max_size=size,
            interpolation=cv2.INTER_AREA
        )

        self.degradation_process = albumentations.SmallestMaxSize(
            max_size=self.lr_size,
            interpolation=cv2.INTER_AREA,
        )

        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert (max_crop_f <= 1.)
        self.crop_fn = albumentations.RandomResizedCrop(
            size, size,
            (min_crop_f, max_crop_f)
        )

        if type(json_file) == dict:
            self.metadata = deepcopy(json_file)
        else:
            with open(json_file, "r") as fp:
                self.metadata = json.load(fp)

        if split is not None:
            self.metadata = self.metadata[split]

        if prefix is not None:
            for k in self.metadata.keys():
                self.metadata[k]["path"] = prefix + self.metadata[k]["path"]

        self.ids = list(self.metadata.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pid_metadata = self.metadata[self.ids[idx]]
        if self.load_img:
            img = self.load_img_(pid_metadata["path"])
            img = self.image_rescaler(image=img)["image"]
            if self.transform is not None:
                img = self.transform(img)
            lr_img = self.degradation_process(image=img)["image"]
        else:
            img = None
            lr_img = None

        sample = {"id": self.ids[idx], "image": img, "LR_image": lr_img}
        sample[0] = sample["image"]
        sample.update({k: pid_metadata[k] for k in self.metadata_labels})
        sample[1] = sample["age_label"]
        return sample

    @staticmethod
    def load_img_(path):
        img = Image.open(path)
        img = img.convert("RGB")
        img = np.array(img).astype("float32") / 127.5 - 1.0
        return img


class IntergrowthDatasetTrain(IntergrowthDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            split="train",
            **kwargs,
        )


class IntergrowthDatasetVal(IntergrowthDataset):
    def __init__(self, *args, min_crop_f=1.0, max_crop_f=1.0, **kwargs):
        super().__init__(
            *args,
            split="val",
            min_crop_f=min_crop_f,
            max_crop_f=max_crop_f,
            **kwargs,
        )
