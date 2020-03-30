# CS8395 Final Project
Final project for medical image processing with deep learning. GAN for data augmentation for spleen segmentation.

## Data Layout
The existing code expects the data to be organized in a `data/` folder with sub-folders
`Testing/`, `Training/`, and `Validation/`. In each of the sub-folders, there should be
`img/` and `label/` which contains the images and the labels.

NOTE: The current `DataLoader` loads every image into RAM at the same time. This requires ~32 GB RAM total.

Training: 6-40
Validation: 1-5
