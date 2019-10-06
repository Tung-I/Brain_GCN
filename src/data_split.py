import logging
import argparse
import csv
import numpy as np
from pathlib import Path


def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    paths = [path for path in data_dir.iterdir() if path.is_dir()]
    num_paths = len(paths)

    random_seed = 0
    np.random.seed(int(random_seed))
    np.random.shuffle(paths)

    csv_path = str(output_dir) + '/kits_train_val_test_split.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for idx, p in enumerate(paths):
            if idx < int(num_paths * 0.8):
                writer.writerow([str(p.parts[-1]), 'train'])
            elif (idx >= int(num_paths * 0.8) and
                    idx < int(num_paths * 0.9)):
                writer.writerow([str(p.parts[-1]), 'valid'])
            else:
                writer.writerow([str(p.parts[-1]), 'test'])


def _parse_args():
    parser = argparse.ArgumentParser(description="To create the split.csv.")
    parser.add_argument('data_dir', type=Path,
                        help='The directory of the data.')
    parser.add_argument('output_dir', type=Path,
                        help='The output directory of the csv file.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
