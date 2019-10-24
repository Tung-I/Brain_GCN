import csv
import glob
import torch
import numpy as np
import nibabel as nib

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose

class BrainDataset(BaseDataset):
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
        self.input_paths = []

        # Collect the data paths according to the dataset split csv.
        with open(self.data_split_csv, "r") as f:
            type_ = 'train' if self.type == 'train' else 'valid'
            rows = csv.reader(f)
            for case_name, split_type in rows:
                if split_type == type_:
                    image_paths = sorted(list((self.data_dir / case_name).glob('image*.npy')))
                    label_paths = sorted(list((self.data_dir / case_name).glob('label*.npy')))
                    gcn_label_paths = sorted(list((self.data_dir / case_name).glob('gcn_label*.npy')))
                    segments_paths = sorted(list((self.data_dir / case_name).glob('segments*.npy')))
                    features_paths = sorted(list((self.data_dir / case_name).glob('features*.npy')))
                    adj_paths = sorted(list((self.data_dir / case_name).glob('adj_arr*.npy')))
                   

                    self.data_paths.extend([(image_path, label_path, gcn_label_path) for image_path, label_path, gcn_label_path in zip(image_paths, label_paths, gcn_label_paths)])
                    self.input_paths.extend([(s_path, f_path, a_path) for s_path, f_path, a_path in zip (segments_paths, features_paths, adj_paths)])


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path, label_path, gcn_label_path = self.data_paths[index]
        #image, label = nib.load(str(image_path)).get_data().astype(np.float32), nib.load(str(label_path)).get_data().astype(np.int64)
        image = np.load(image_path).astype(np.float32)
        label = np.load(label_path).astype(np.int64)
        gcnlabel = np.load(gcn_label_path).astype(np.int64)
        s_path , f_path, a_path = self.input_paths[index]
        segments = np.load(s_path).astype(np.int64)
        features = np.load(f_path).astype(np.float32)
        adj_arr = np.load(a_path).astype(np.float32)

        image, label, gcnlabel = self.transforms(image, label, gcnlabel, dtypes=[torch.float, torch.long, torch.long])
        segments, features, adj_arr = self.transforms(segments, features, adj_arr, dtypes=[torch.long, torch.float, torch.float]) 
        #print(segments.device)
        #adj_arr = adj_generate(features, self.tao)
        #print(adj_arr.device)
        #adj_arr = self.transforms(adj_arr, dtypes=[torch.float])
        
        image, label, gcnlabel = image.contiguous(), label.contiguous(), gcnlabel.contiguous()
        segments, features, adj_arr = segments.contiguous(), features.contiguous(), adj_arr.contiguous()
        #return {"features": features, "adj_arr": adj_arr, "label": label, "segments": segments}
        return {"image": image, "label": label, "gcnlabel": gcnlabel, "segments": segments, "features": features, "adj_arr": adj_arr}


