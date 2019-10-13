import csv
import glob
import torch
import numpy as np
import nibabel as nib

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose
from src.data.transforms import SLIC_transform
from src.data.transforms import feature_extract

class BrainDataset(BaseDataset):
    """
    Args:
        data_split_csv (str): The path of the training and validation data split csv file.
        train_preprocessings (list of Box): The preprocessing techniques applied to the training data before applying the augmentation.
        valid_preprocessings (list of Box): The preprocessing techniques applied to the validation data before applying the augmentation.
        transforms (list of Box): The preprocessing techniques applied to the data.
        augments (list of Box): The augmentation techniques applied to the training data (default: None).
    """
    def __init__(self, data_split_csv, train_preprocessings, valid_preprocessings, transforms, augments=None, n_segments, compactness, feature_range, n_vertex, tao,  **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        self.train_preprocessings = compose(train_preprocessings)
        self.valid_preprocessings = compose(valid_preprocessings)
        self.transforms = compose(transforms)
        self.augments = compose(augments)
        
        self.n_segments = n_segments
        self.compactness = compactness
        self.feature_range = feature_range
        self.n_vertex = n_vertex
        self.tao = tao

        self.data_paths = []

        # Collect the data paths according to the dataset split csv.
        with open(self.data_split_csv, "r") as f:
            type_ = 'Training' if self.type == 'train' else 'Validation'
            rows = csv.reader(f)
            for case_name, split_type in rows:
                if split_type == type_:
                    image_paths = sorted(list((self.data_dir / case_name).glob('image*.nii.gz')))
                    label_paths = sorted(list((self.data_dir / case_name).glob('label*.nii.gz')))
                    self.data_paths.extend([(image_path, label_path) for image_path, label_path in zip(image_paths, label_paths)])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path, label_path = self.data_paths[index]
        image, label = nib.load(str(image_path)).get_data().astype(np.float32), nib.load(str(label_path)).get_data().astype(np.int64)
        image, label = image.transpose(1, 2, 0)[..., None], label.transpose(1, 2, 0)[..., None]

        if self.type == 'train':
            image, label = self.train_preprocessings(image, label)
            #image, label = self.augments(image, label, elastic_deformation_orders=[3, 0])
        elif self.type == 'valid':
            image, label = self.valid_preprocessings(image, label)

        segments = SLIC_transform(image, self.n_segments, self.compactness)
        features, adj_arr = feature_extract(image, segments, self.feature_range, self.n_vertex, self.tao)

        features, adj_arr, label ,segments = self.transforms(features, adj_arr, label, segments, dtypes=[torch.float, torch.float, torch.long, torch.long])
        features, adj_arr, label ,segments = features.contiguous(), adj_arr.contigugous(), label.contiguous(), segments.contiguous()
        return {"features": features, "adj_arr": adj_arr, "label": label, "segments": segments}
