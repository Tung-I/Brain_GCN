import csv
import glob
import torch
import numpy as np
import nibabel as nib

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose
from src.data.transforms import SLIC_transform
from src.data.transforms import feature_extract
from src.data.transforms import max_min_normalize
from src.data.transforms import label_transform

class BrainDataset(BaseDataset):
    """
    Args:
        data_split_csv (str): The path of the training and validation data split csv file.
        train_preprocessings (list of Box): The preprocessing techniques applied to the training data before applying the augmentation.
        valid_preprocessings (list of Box): The preprocessing techniques applied to the validation data before applying the augmentation.
        transforms (list of Box): The preprocessing techniques applied to the data.
        augments (list of Box): The augmentation techniques applied to the training data (default: None).
    """
    def __init__(self, data_split_csv, train_preprocessings, valid_preprocessings, transforms, n_segments, compactness, feature_range, n_vertex, tao, augments=None, **kwargs):
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
            type_ = 'train' if self.type == 'train' else 'valid'
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

        image = np.power(image, 2)
        image = max_min_normalize(image)

        if self.type == 'train':
            image, label = self.train_preprocessings(image, label)
            #image, label = self.augments(image, label, elastic_deformation_orders=[3, 0])
        elif self.type == 'valid':
            image, label = self.valid_preprocessings(image, label)

        segments = SLIC_transform(image, self.n_segments, self.compactness)
        features = feature_extract(image, segments, self.feature_range, self.n_vertex)
        label = label_transform(label, segments, self.n_vertex)


        features, label ,segments = self.transforms(features, label, segments, dtypes=[torch.float, torch.long, torch.long]) 
        #print(features.shape)
        #print(segments.device)
        #adj_arr = adj_generate(features, self.tao)
        #print(adj_arr.device)
        #adj_arr = self.transforms(adj_arr, dtypes=[torch.float])
        
        #features, adj_arr, label ,segments = features.contiguous(), adj_arr.contiguous(), label.contiguous(), segments.contiguous()
        features, label ,segments = features.contiguous(), label.contiguous(), segments.contiguous()
        #return {"features": features, "adj_arr": adj_arr, "label": label, "segments": segments}
        return {"features": features, "label": label, "segments": segments, "tao": self.tao}


