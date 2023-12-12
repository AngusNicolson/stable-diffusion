
import json
from typing import Union, Any, Tuple, Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class IntergrowthDataset(Dataset):
    """Intergrowth dataset"""

    def __init__(self, json_file: Union[str, dict], split: str = "train", prefix: str = None, transform=None, load_img=True, size=256):
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
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = None

        sample = {"id": self.ids[idx], "image": img}
        sample[0] = sample["image"]
        sample.update({k: pid_metadata[k] for k in self.metadata_labels})
        sample[1] = sample["age_label"]
        return sample

    @staticmethod
    def load_img_(path):
        img = Image.open(path)
        img = np.array(img).astype("float32") / 127.5 - 1.0
        img = IntergrowthDataset.expand_img(img)
        return img

    @staticmethod
    def expand_img(img):
        img = np.expand_dims(img, -1)
        img = np.repeat(img, repeats=3, axis=-1)
        img = torch.from_numpy(img)
        return img
