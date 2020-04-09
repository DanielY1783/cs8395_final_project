# Author: Daniel Yan
# Slice nii files into 2D numpy arrays for GAN use. Save as png file.

# Imports
import nibabel as nib
import numpy as np
from PIL import Image
import os

# Constants
# File path to original nii volumes and labels
IMG_PATH = "../../data/Training/img/"
LABEL_PATH = "../../data/Training/label/"
# File path to save the 2d slices of images and labels
IMG_SLICE_PATH = "../../data/GAN_2d/img/"
LABEL_SLICE_PATH = "../../data/GAN_2d/label/"

def main():
    # Process all labels
    for file_name in os.listdir(LABEL_PATH):
        # Load the image
        image = nib.load(LABEL_PATH + file_name)
        # Get the array of values
        image_data = image.get_fdata()
        # Filter by spleen and convert to uint8. Set spleen as 128 since there are pixel range is 0-255
        image_data = np.where(image_data==1, 128, 0)
        image_data = image_data.astype(np.uint8)
        # Slice along z axis and save
        for slice in range(image.shape[2]):
            image_2d = image_data[:, :, slice]
            # Take out the "label" part from original image and save slice with PIL
            im = Image.fromarray(image_2d)
            im.save(LABEL_SLICE_PATH + file_name[5:-7] + "_" + str(slice) + ".png")
    # Process the image volumes
    for file_name in os.listdir(IMG_PATH):
        # Load the image
        image = nib.load(IMG_PATH + file_name)
        # Get the array of values
        image_data = image.get_fdata()
        # Resize to 0-255 range. Images are CT scans so clip to 1000 first.
        image_data = np.clip(image_data, -1000, 1000)
        image_data = image_data + 1000.0
        image_data = image_data * 255.0 / 2000
        image_data = image_data.astype(np.uint8)
        # Slice along z axis and save
        for slice in range(image.shape[2]):
            image_2d = image_data[:, :, slice]
            image_2d = image_2d.astype(np.uint8)
            # Take out the "img" part from original image and save slice with PIL
            im = Image.fromarray(image_2d)
            im.save(IMG_SLICE_PATH + file_name[3:-7] + "_" + str(slice) + ".png")



if __name__ == '__main__':
    main()