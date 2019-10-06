import logging
import argparse
import csv
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage.interpolation import zoom


def main(args):
    with open(args.data_split_csv, "r") as f:
        rows = csv.reader(f)
        for case_name, _ in rows:
            logging.info(f'Process {case_name}')
            patient_dir = args.output_dir / f'{case_name}'
            if not patient_dir.is_dir():
                patient_dir.mkdir(parents=True)

            image_path = args.data_dir / f'{case_name}' / 'imaging.nii.gz'
            label_path = args.data_dir / f'{case_name}' / 'segmentation.nii.gz'
            image_metadata, label_metadata = nib.load(str(image_path)), nib.load(str(label_path))
            image, label = image_metadata.get_data().astype(np.float32), label_metadata.get_data().astype(np.uint8)
            z_resolution = image_metadata.header['pixdim'][1]
            image = zoom(image, (z_resolution / 5, 0.5, 0.5), order=3)
            label = zoom(label, (z_resolution / 5, 0.5, 0.5), order=0)
            nib.save(nib.Nifti1Image(image, np.eye(4)), str(patient_dir / 'imaging.nii.gz'))
            nib.save(nib.Nifti1Image(label, np.eye(4)), str(patient_dir / 'segmentation.nii.gz'))


def _parse_args():
    parser = argparse.ArgumentParser(description="The data preprocessing.")
    parser.add_argument('data_dir', type=Path, help='The directory of the data.')
    parser.add_argument('output_dir', type=Path, help='The output directory of the processed data.')
    parser.add_argument('data_split_csv', type=str, help='The path of the data split csv file.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)