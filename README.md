# Crack-segmentation-using-DeepLabv3-

The code belongs to a crack segmentation task using the DeepLabv3+ architecture and Squeeze-and-Excitation (S&E) attention module.

"""
DeepLabV3+ Crack Segmentation with Xception Backbone
=====================================================

This script trains a binary crack-segmentation model using a DeepLabV3+-style
architecture with an Xception encoder, ASPP module, and squeeze-and-excitation
blocks.

Expected dataset structure
--------------------------
crack_segmentation_dataset/
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── masks/
    ├── image_001.png
    ├── image_002.png
    └── ...

Notes
-----
- Images are resized to IMAGE_SIZE x IMAGE_SIZE.
- Masks are converted to binary values: 0 for background and 1 for crack.
- The model uses Dice loss and reports IoU, Dice coefficient, recall, and precision.
"""
