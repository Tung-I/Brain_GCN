import csv
import glob
import torch
import random
import numpy as np
from box import Box
import nibabel as nib
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class VIPCUPSegDataset(BaseDataset):
    """
    Args:
        data_split_csv (str): The path of the training and validation data split csv file.
        train_preprocessings (list of Box): The preprocessing techniques applied to the training data before applying the augmentation.
        valid_preprocessings (list of Box): The preprocessing techniques applied to the validation data before applying the augmentation.
        transforms (list of Box): The preprocessing techniques applied to the data.
        augments (list of Box): The augmentation techniques applied to the training data (default: None).
    """
    def __init__(self, data_split_csv, train_preprocessings, valid_preprocessings, transforms, augments=None, **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        self.train_preprocessings = compose(train_preprocessings)
        self.valid_preprocessings = compose(valid_preprocessings)
        self.transforms = compose(transforms)
        self.augments = compose(augments)
        self.data_paths = []

        # Collect the data paths according to the dataset split csv.
        with open(self.data_split_csv, "r") as f:
            type_ = 'Training' if self.type == 'train' else 'Validation'
            rows = csv.reader(f)
            for case_name, split_type in rows:
                if split_type == type_:
                    image_path = self.data_dir / f'{case_name[:5]}-{case_name[5:]}image.nii.gz'
                    label_path = self.data_dir / f'{case_name[:5]}-{case_name[5:]}label_GTV1.nii.gz'
                    self.data_paths.append([image_path, label_path])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path, label_path = self.data_paths[index]
        image, label = nib.load(str(image_path)).get_fdata()[..., None], nib.load(str(label_path)).get_fdata()[..., None]

        if self.type == 'train':
            image, label = self.train_preprocessings(image, label, normalize_tags=[True, False], target=label, target_label=1)
            image, label = self.augments(image, label, elastic_deformation_orders=[3, 0])
        elif self.type == 'valid':
            image, label = self.valid_preprocessings(image, label, normalize_tags=[True, False], target=label, target_label=1)

        image, label = self.transforms(image, label, dtypes=[torch.float, torch.long])
        image, label = image.permute(3, 2, 0, 1).contiguous(), label.permute(3, 2, 0, 1).contiguous()
        return {"image": image, "label": label}
