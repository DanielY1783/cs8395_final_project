# test.py
# Matthew C. Sedam
# CS 8395 Deep Learning in Medical Image Processing

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.models import densenet201
from torchvision import transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from PIL import Image
from src.unet import UNet
from src.util import CTImageDataset

GOOD_DIR = 'good'
MODEL_PATH = GOOD_DIR + '/best_model.pt'
TEST_IMAGE_DIR = 'data/Testing/img'
TR_IMAGE_DIR = 'data/Training/img'
TR_SEG_DIR = 'data/Training/label'
VAL_IMAGE_DIR = 'data/Validation/img'
VAL_SEG_DIR = 'data/Validation/label'


def load_model(use_cpu=False):
    """
    Loads the model.
    :param use_cpu: if True, use cpu and not cuda
    :return: the model, the device
    """

    device = torch.device('cpu' if use_cpu else 'cuda')
    model = UNet(1, 1, bilinear=False)
    model = model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


def main():
    model, device = load_model()

    test_data = CTImageDataset(image_dir=TEST_IMAGE_DIR, seg_image_dir=None, testing=True)
    tr_data = CTImageDataset(image_dir=TR_IMAGE_DIR, seg_image_dir=TR_SEG_DIR)
    val_data = CTImageDataset(image_dir=VAL_IMAGE_DIR, seg_image_dir=VAL_SEG_DIR)

    with torch.no_grad():
        dice_all = []
        for image_dir, seg_dir, dataset in [(TEST_IMAGE_DIR, None, test_data),
                                            (TR_IMAGE_DIR, TR_SEG_DIR, tr_data),
                                            (VAL_IMAGE_DIR, VAL_SEG_DIR, val_data)]:
            seg_2dimages = []
            for data, target in dataset:
                data = np.array([data])
                data = torch.from_numpy(data).to(device)
                seg = model(data)
                seg = seg.cpu().numpy().transpose(0, 2, 3, 1)[0]
                seg = seg.reshape((seg.shape[0], seg.shape[1]))
                seg_2dimages.append(seg)
            seg_2dimages = np.array(seg_2dimages)
            seg_2dimages = (seg_2dimages >= 0.5) * 1

            image_fns = list(filter(lambda x: '.nii.gz' in x, os.listdir(image_dir)))
            image_fns.sort()
            seg_fns = list(filter(lambda x: '.nii.gz' in x, os.listdir(seg_dir))) if seg_dir else None
            if seg_fns:
                seg_fns.sort()
            num_proc_slices = 0
            for i, (image_fn, seg_fn) in enumerate(zip(image_fns,
                                                       seg_fns if seg_fns else [None for _ in range(len(image_fns))])):
                image = nib.load(image_dir + '/' + image_fn)
                image_shape = image.get_fdata().shape
                image_num_slices = image_shape[2]

                # calculate output segmentation image and save
                out_seg_image = seg_2dimages[num_proc_slices:num_proc_slices + image_num_slices, :, :]
                out_seg_image = out_seg_image.transpose(1, 2, 0)
                out_seg_image = nib.Nifti1Image(out_seg_image, image.affine)

                out_seg_image_fn = image_fn.replace('img', 'label')
                nib.save(out_seg_image, out_seg_image_fn)

                # calculate DICE
                if seg_fn:
                    seg_image = nib.load(seg_dir + '/' + seg_fn).get_fdata()
                    seg_image = np.multiply((seg_image >= 0.5) * 1, (seg_image <= 1.5) * 1)
                    out_seg_image = nib.load(out_seg_image_fn).get_fdata()
                    out_seg_image = (out_seg_image >= 0.5) * 1
                    assert seg_image.shape == out_seg_image.shape

                    dice = ((2.0 * np.sum(seg_image[out_seg_image == 1] == 1)) /
                            (np.sum(seg_image) + np.sum(out_seg_image)))
                    dice_all.append((image_fn, dice))

                num_proc_slices += image_num_slices
                print(i + 1, '/', len(image_fns), ':', (i + 1) * 100 / len(image_fns), '%')

        with open('dice.txt', 'w') as file:
            dice_all = [x[0] + ': ' + str(x[1]) for x in dice_all]
            dice_all = '\n'.join(dice_all) + '\n'
            file.write(dice_all)


if __name__ == '__main__':
    main()
