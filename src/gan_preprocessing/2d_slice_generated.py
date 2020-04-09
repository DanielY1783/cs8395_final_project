# Author: Daniel Yan
# Slice nii files into 2D numpy arrays for GAN use. Save as png file.

# Imports
import nibabel as nib
import numpy as np
from PIL import Image
import os

# Constants
# File paths
LABEL_PATH = "../../data/generated_segmentation/"
LABEL_SLICE_PATH = "../../data/generated_segmentation_2d/"

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
            im.save(LABEL_SLICE_PATH + file_name[:-7] + "_" + str(slice) + ".png")


if __name__ == '__main__':
    main()