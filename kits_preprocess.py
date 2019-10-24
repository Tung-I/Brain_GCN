import numpy as np
import logging
import nibabel as nib
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
from skimage.segmentation import slic
from pathlib import Path
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from tqdm import tqdm


def main(data_dir, output_dir):
    paths = [path for path in data_dir.iterdir() if path.is_dir()]

    N_SEG = 5
    COMPACTNESS = 10
    N_VERTEX = 512
    N_FEATURES = 64
    TAO = 0.1
    THRESHOLD = 0.5

    for path in paths:
        logging.info(f'Process {path.parts[-1]}.')
        print(f'Process {path.parts[-1]}.')
        # Create output directory
        if not (output_dir / path.parts[-1]).is_dir():
            (output_dir / path.parts[-1]).mkdir(parents=True)

        # Read in the MRI scans
        image = nib.load(str(path / 'image.nii.gz')).get_data().astype(np.float32)
        image = np.squeeze(image, 3)
        image = image.transpose(2, 0, 1)
        label = nib.load(str(path / 'label.nii.gz')).get_data().astype(np.uint8)
        label = np.squeeze(label, 3)
        label = label.transpose(2, 0, 1)

        # Save each slice of the scan into single file
        for s in tqdm(range(image.shape[0])):
            if s>=5 and s<=115:
                _image = image[s]
                _label = label[s]
                
                img_crop, label_crop = various_crop(_image, _label, IMG_MIN_SIZE)
                img_crop = max_min_normalize(img_crop)
                img_3c = np.expand_dims(img_crop, axis=2)
                img_3c = np.concatenate((img_3c, img_3c, img_3c), 2)
                
                segments = slic(img_3c.astype('double'), n_segments=N_SEG, compactness=COMPACTNESS)
                features = feature_extract(img_crop, segments, N_VERTEX, N_FEATURES)
                adj_arr = adj_generate(features, segments, TAO, THRESHOLD)
                
                gcn_label = label_transform(label_crop, segments, N_VERTEX, N_FEATURES)

                np.save(str(output_dir / path.parts[-1] / f'image_{s}.npy'), img_crop)
                np.save(str(output_dir / path.parts[-1] / f'label_{s}.npy'), label_crop)
                np.save(str(output_dir / path.parts[-1] / f'gcn_label_{s}.npy'), gcn_label)
                np.save(str(output_dir / path.parts[-1] / f'segments_{s}.npy'), segments)
                np.save(str(output_dir / path.parts[-1] / f'features_{s}.npy'), features)
                np.save(str(output_dir / path.parts[-1] / f'adj_arr_{s}.npy'), adj_arr)


def various_crop(img, label, least_size):
    """
    Args:
        img: (numpy.ndarray)
    Returns:
        cropped_img: (numpy.ndarray)
    """
    h_size = img.shape[0]
    w_size = img.shape[1]
    anchor = [0, h_size, 0, w_size]
    line_w = img.sum(0)
    line_h = img.sum(1)
    flag = False
    for i in range(h_size):
        if flag==False and line_h[i]!=0:
            anchor[0] = i
            flag = True
        elif flag==True and line_h[i]==0:
            anchor[1] = i
            flag = False
            break;
    flag = False
    for i in range(w_size):
        if flag==False and line_w[i]!=0:
            anchor[2] = i
            flag = True
        elif flag==True and line_w[i]==0:
            anchor[3] = i
            flag = False
            break;
    h_up, h_down, w_left, w_right = anchor
    if (h_down - h_up) < least_size:
        h_down = h_up + least_size
    if (w_right - w_left) < least_size:
        w_right = w_left + least_size
    return img[h_up:h_down, w_left:w_right], label[h_up:h_down, w_left:w_right]


def max_min_normalize(img):
    img = np.asarray(img)
    return (img-np.min(img)) / ((np.max(img)-np.min(img) + 1e-10))


def value_count(arr, num):
    """
    Args:
        arr: value 0~1 (numpy.ndarray)
    Returns:
        arr of value count: (numpy.ndarray)
    """
    arr = arr.flatten()
    arr = (arr * (num-1)).astype(np.int32)
    cnt_arr = np.zeros(num).astype(np.int32)
    for i in range(arr.shape[0]):
        cnt_arr[arr[i]] += 1
    return cnt_arr / (cnt_arr.max() + 1e-10)


def feature_extract(img, segments, v_num, f_num):
    features = np.zeros((v_num, f_num))
    for i in range(v_num):
        area = img[np.where(segments==i)]
        features[i] = value_count(area, f_num)
    return features 


def adj_generate(features, segments, tao, threshold):
    centroid = []
    range_num = features.shape[0] if features.shape[0] < (segments.max()+1) else (segments.max()+1)
    for i in range(range_num):
        centroid.append((np.where(segments==i)[0].mean(), np.where(segments==i)[1].mean()))
    centroid = np.asarray(centroid)
    adj_arr = np.zeros((features.shape[0], features.shape[0]))
    for i in range(range_num):
        for j in range(i+1, range_num):
            if LA.norm(centroid[i] - centroid[j]) > (segments.shape[0]*threshold):
                adj_arr[i, j] = 0
            else:
                e_dist = LA.norm((features[i] - features[j]))
                tmp = -1 * e_dist * e_dist / (2*tao*tao)
                adj_arr[i, j] = math.exp(tmp)
            adj_arr[j, i] = adj_arr[i, j]
    adj_arr += np.eye(features.shape[0])
    return adj_arr


def label_transform(label, segments, v_num, f_num):
    gcn_label = np.zeros((v_num, 1))
    n_range = int(np.minimum(v_num, segments.max()+1))
    for i in range(n_range):
        gcn_label[i] = int(np.median(label[np.where(segments==i)]))
    return gcn_label


if __name__ == "__main__":
    data_dir = Path('/home/tony/Documents/IBSR_preprocessed/')
    output_dir = Path('/home/tony/Documents/IBSR_features_01/')
    main(data_dir, output_dir)

