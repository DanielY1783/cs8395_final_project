# util.py
# Matthew C. Sedam
# CS 8395 Deep Learning in Medical Image Processing
# Some code from: https://github.com/pytorch/examples/blob/master/mnist/main.py

import argparse

import nibabel as nib
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_MIN_VALUE = -1000
IMAGE_MAX_VALUE = 1000
IMAGE_INPUT_SIZE = 512

DATA_DIR = './data'
TR_DATA = DATA_DIR + '/Training/img'
VAL_DATA = DATA_DIR + '/Validation/img'
TEST_DATA = DATA_DIR + '/Testing/img'
TR_DATA_LABELS = DATA_DIR + '/Training/label'
VAL_DATA_LABELS = DATA_DIR + '/Validation/label'
TEST_DATA_LABELS = DATA_DIR + '/Testing/label'

TARGET_LABEL = 1


class CTImageDataset(Dataset):
    """CT Image Dataset"""

    def __init__(self, image_dir=TR_DATA, seg_image_dir=TR_DATA_LABELS, testing=False):
        """
        Initialize dataset. Load into memory for faster access.
        :param image_dir: the image directory
        :param seg_image_dir: the segmentation image directory
        :param testing: if True, in testing mode (do not pull segmentations)
        """

        print('START: Loading images')

        self.image_dir = image_dir
        self.seg_image_dir = seg_image_dir
        self.testing = testing

        image_fns = os.listdir(self.image_dir)
        image_fns.sort()
        self.images = []
        for image_fn in image_fns:
            image = nib.load(self.image_dir + '/' + image_fn).get_fdata().transpose(2, 0, 1)
            for image2d in image:
                image2d_proc = image2d.reshape((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1))
                image2d_proc -= IMAGE_MIN_VALUE
                image2d_proc /= (IMAGE_MAX_VALUE - IMAGE_MIN_VALUE)
                image2d_proc = np.clip(image2d_proc, 0, 1)
                image2d_proc = image2d_proc.transpose(2, 0, 1)
                self.images.append(image2d_proc)
        self.images = np.array(self.images, dtype=np.float32)

        if not self.testing:
            seg_image_fns = os.listdir(self.seg_image_dir)
            seg_image_fns.sort()
            self.seg_images = []
            for seg_image_fn in seg_image_fns:
                seg_image = nib.load(self.seg_image_dir + '/' + seg_image_fn).get_fdata().transpose(2, 0, 1)
                for seg_image2d in seg_image:
                    seg_image2d_proc = seg_image2d.reshape((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1))
                    seg_image2d_proc = seg_image2d_proc.transpose(2, 0, 1)
                    self.seg_images.append(seg_image2d_proc)
            self.seg_images = np.array(self.seg_images)
            self.seg_images = (self.seg_images == TARGET_LABEL) * 1  # use only target label (spleen)
            self.seg_images = np.array(self.seg_images, dtype=np.float32)

        print('END: Loading images')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], (self.seg_images[idx] if not self.testing else self.images[idx])


def get_args():
    """
    Returns the arguments.
    :return: the args
    """

    parser = argparse.ArgumentParser(description='Homework 3')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='M',
                        help='Learning rate step gamma (default: 0.95)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=np.random.rand(), metavar='S',
                        help='random seed (default: RANDOM)')
    return parser.parse_args()
