import csv
import torch
import logging
import numpy as np
import nibabel as nib
from tqdm import tqdm

from src.data.transforms import compose


class VIPCUPSegPredictor(object):
    """The predictor for the VIPCUP 2018 segmentation task.
    Args:
        data_dir (Path): The directory of the saved data.
        data_split_csv (str): The path of the training and validation data split csv file.
        preprocessings (list of Box): The preprocessing techniques applied to the testing data.
        transforms (list of Box): The preprocessing techniques applied to the data.
        sample_size (tuple): The window size of each sampled data.
        shift (tuple): The shift distance between two contiguous samples.
        device (torch.device): The device.
        net (BaseNet): The network architecture.
        metric_fns (list of torch.nn.Module): The metric functions.
        saved_dir (str): The directory to save the predicted videos, images and metrics (default: None).
        exported (bool): Whether to export the predicted video, images and metrics (default: False).
    """
    def __init__(self, data_dir, data_split_csv, preprocessings, transforms, sample_size, shift,
                 device, net, metric_fns, saved_dir=None, exported=None):
        self.data_dir = data_dir
        self.data_split_csv = data_split_csv
        self.preprocessings = compose(preprocessings)
        self.transforms = compose(transforms)
        self.sample_size = sample_size
        self.shift = shift
        self.device = device
        self.net = net
        self.metric_fns = metric_fns
        self.saved_dir = saved_dir
        self.exported = exported
        self.log = self._init_log()

    def _init_log(self):
        """Initialize the log.
        Returns:
            log (dict): The initialized log.
        """
        log = {}
        for metric in self.metric_fns:
            if metric.__class__.__name__ == 'Dice':
                for i in range(self.net.out_channels):
                    log[f'Dice_{i}'] = 0
            elif metric.__class__.__name__ == 'FalseNegativeSize':
                for i in range(self.net.out_channels-1):
                    log[f'FalseNegativeSize_{i+1}'] = []
            else:
                log[metric.__class__.__name__] = 0
        return log

    def _update_log(self, log, metrics):
        """Update the log.
        Args:
            log (dict): The log to be updated.
            metrics (list of torch.Tensor): The computed metrics.
        """
        for metric, _metric in zip(self.metric_fns, metrics):
            if metric.__class__.__name__ == 'Dice':
                for i, class_score in enumerate(_metric):
                    log[f'Dice_{i}'] += class_score.item()
            elif metric.__class__.__name__ == 'FalseNegativeSize':
                for i, class_score in enumerate(_metric):
                    log[f'FalseNegativeSize_{i+1}'] += class_score
            else:
                log[metric.__class__.__name__] += _metric.item()

    def predict(self):
        data_paths = []
        count = 0
        self.net.eval()
        # Create the testing data path list
        with open(self.data_split_csv, "r") as f:
            rows = csv.reader(f)
            for case_name, split_type in rows:
                if split_type == 'Validation':
                    image_path = self.data_dir / f'{case_name[:5]}-{case_name[5:]}image.nii.gz'
                    label_path = self.data_dir / f'{case_name[:5]}-{case_name[5:]}label_GTV1.nii.gz'
                    data_paths.append([image_path, label_path])

        # Initital the list for saving metrics as a csv file
        header = ['name']
        for metric in self.metric_fns:
            if metric.__class__.__name__ == 'Dice':
                for i in range(self.net.out_channels):
                    header += [f'Dice_{i}']
            elif metric.__class__.__name__ == 'FalseNegativeSize':
                # From class 1 to class n
                for i in range(self.net.out_channels-1):
                    header += [f'FalseNegativeSize_{i+1}']
            else:
                header += [metric.__class__.__name__]
        results = [header]

        if self.exported:
            csv_path = self.saved_dir / 'results.csv'
            output_dir = self.saved_dir / 'prediction'
            if not output_dir.is_dir():
                output_dir.mkdir(parents=True)

        trange = tqdm(data_paths, total=len(data_paths), desc='testing')
        for data_path in trange:
            image_path, label_path = data_path
            image, label = nib.load(str(image_path)).get_fdata(), nib.load(str(label_path)).get_fdata()
            data_shape = list(image.shape)

            image, label = self.preprocessings(image[..., None], label[..., None], normalize_tags=[True, False])
            image, label = self.transforms(image, label, dtypes=[torch.float, torch.long])
            image, label = image.permute(3, 2, 0, 1).contiguous().to(self.device), label.permute(3, 2, 0, 1).contiguous().to(self.device)
            prediction = torch.zeros(1, self.net.out_channels, *image.shape[1:], dtype=torch.float32).to(self.device)
            pixel_count = torch.zeros(1, self.net.out_channels, *image.shape[1:], dtype=torch.float32).to(self.device)

            # Get the coordinated of each sampled volume
            starts, ends = [], []
            for k in range(0, data_shape[2], self.shift[2]):
                for j in range(0, data_shape[1], self.shift[1]):
                    for i in range(0, data_shape[0], self.shift[0]):
                        ends.append([min(i+self.sample_size[0], data_shape[0]), \
                                     min(j+self.sample_size[1], data_shape[1]), \
                                     min(k+self.sample_size[2], data_shape[2])])
                        starts.append([ends[-1][i]-self.sample_size[i] for i in range(len(data_shape))])

            # Get the prediction and calculate the average of the overlapped area
            for start, end in zip(starts, ends):
                input = image[:, start[2]:end[2], start[0]:end[0], start[1]:end[1]]
                with torch.no_grad():
                    output = self.net(input.unsqueeze(dim=0))
                prediction[:, :, start[2]:end[2], start[0]:end[0], start[1]:end[1]] += output
                pixel_count[:, :, start[2]:end[2], start[0]:end[0], start[1]:end[1]] += 1.0
            prediction = prediction / pixel_count

            count += 1
            metrics = [metric(prediction, label.unsqueeze(dim=0)) for metric in self.metric_fns]
            self._update_log(self.log, metrics)

            # Export the prediction
            if self.exported:
                filename = str(image_path.parts[-1]).replace('image', 'pred')
                prediction = prediction.argmax(dim=1).squeeze().cpu().numpy().transpose(1, 2, 0)
                nib.save(nib.Nifti1Image(prediction, np.eye(4)), str(output_dir / filename))

                result = [filename]
                for metric, _metric in zip(self.metric_fns, metrics):
                    if metric.__class__.__name__ == 'Dice':
                        for i, class_score in enumerate(_metric):
                            result.append(class_score.item())
                    elif metric.__class__.__name__ == 'FalseNegativeSize':
                        for i, class_score in enumerate(_metric):
                            if len(class_score) == 0:
                                result.append(0)
                            else:
                                result.append(np.mean(class_score))
                    else:
                        result.append(_metric.item())
                results.append([*result])

            dicts = {}
            for key, value in self.log.items():
                if 'FalseNegativeSize' in key:
                    dicts[key] = f'{np.mean(value): .3f}' if len(value) > 0 else str(0.000)
                else:
                    dicts[key] = f'{value / count: .3f}'
            trange.set_postfix(**dicts)

        for key in self.log:
            if 'FalseNegativeSize' in key:
                self.log[key] = np.mean(self.log[key]) if len(self.log[key]) > 0 else 0
            else:
                self.log[key] /= count

        if self.exported:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(results)

        logging.info(f'Test log: {self.log}.')
