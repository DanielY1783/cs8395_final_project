# Author: Daniel Yan
# Generates new segmentations from existing segmenetations by either increasing or decreasing the size of the spleen.
# For each original segmentation, generate two new segmentations by increasing the size of the spleen, and
# two new segmentations by decreasing the size of the spleen.

# Imports
import nibabel as nib
import numpy as np
import os

# Constants
# File path to original training labels
ORIGINAL_SEGS = "../../data/Training/label/"
# File path to generated segmentations
GENERATED = "../../data/"

def grow_images(image, iterations=5, number=2, weight=0.4, std=0.25):
    """

    :param image:
    :param iterations:
    :param number:
    :return:
    """
    # Get the original image data
    original_image_data = image.get_fdata()
    # Get the x, y, z dimensions
    x_shape = original_image_data.shape[0]
    y_shape = original_image_data.shape[1]
    z_shape = original_image_data.shape[2]
    # Use a loop to generate the number of images specified
    for n in range(number):
        # Create copy of original data to modify
        image_data = np.copy(original_image_data)
        # Grow images by adding number of adjacent voxels with spleen
        # for a set number of iterations
        for i in range(iterations):
            # Shift images by 1 voxel in each direction
            x_plus = np.zeros(image_data.shape)
            x_plus[1:x_shape, :, :] = image_data[0:x_shape-1, :, :]
            x_minus = np.zeros(image_data.shape)
            x_minus[0:x_shape-1, :, :] = image_data[1:x_shape, :, :]
            y_plus = np.zeros(image_data.shape)
            y_plus[:, 1:y_shape, :] = image_data[:, 0:y_shape-1, :]
            y_minus = np.zeros(image_data.shape)
            y_minus[:, 0:y_shape-1, :] = image_data[:, 1:y_shape, :]
            z_plus = np.zeros(image_data.shape)
            z_plus[:, :, 1:z_shape] = image_data[:, :, 0:z_shape-1]
            z_minus = np.zeros(image_data.shape)
            z_minus[:, :, 0:z_shape-1] = image_data[:, :, 1:z_shape]
            # Add to get the sum of the values for the neighboring voxels.
            image_data = x_plus + x_minus + y_plus + y_minus + z_plus + z_minus
            # Multiply by weight for each neighboring pixel
            image_data = image_data * weight
            # 

def main():
    # Iterate through all files in the folder for original training
    # segmentations to generate new segmentations for one
    for file_name in os.listdir(ORIGINAL_SEGS):
        print("Generating segmentations for: ", file_name)
        # Load the image
        image = nib.load(ORIGINAL_SEGS + file_name)
        # Get the array of values
        image_data = image.get_fdata()
        # Filter for only spleen labels
        image_data = np.where(image_data == 1, 1, 0)
        # Save the original spleen segmentation
        image = nib.Nifti1Image(image_data, image.affine)
        nib.save(image, GENERATED + file_name)
        # Generate new segmentations that are larger than original segmentations
        grow_images(image)
        shrink_images(image)

if __name__ == '__main__':
    main()