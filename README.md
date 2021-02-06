# PyTorch for Image segmentation
 This project implements several convolutional neural network models for semantic segmentation in PyTorch. It uses glomerulus identification as test subject to compare the performance of different neural networks, as well as loss functions.

## Models
 1. ResNet ([Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385))
 2. SE-ResNet ([Squeeze-and-Excitation Networks (2017)](https://arxiv.org/abs/1709.01507))
 3. PSPNet ([Pyramid Scene Parsing Network (2016)](https://arxiv.org/abs/1612.01105))
 4. U-Net ([U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)](https://arxiv.org/abs/1505.04597))

## Loss functions
 1. binary cross-entropy loss
 2. Sørensen–Dice loss
 3. Jaccard loss

## Directories
 1. *models* - contains all the neural network definitons
 2. *losses* - contains all the loss function definitons
 3. *data* - contains the python script for processing human kidney tissue images from [HuBMAP - Hacking the Kidney](https://www.kaggle.com/c/hubmap-kidney-segmentation/data)
 4. *training* - contains the python code for neural network training

## Comparison between models
Using semantic segmentation of human kidney tissue images as a benchmark test, PSPNet shows the best performance:

| Model     | Dice coefficient  |
| --------- | ----------------- |
| ResNet    | Content Cell      |
| SE-ResNet | Content Cell      |
| PSPNet    | Content Cell      |
| U-Net     | Content Cell      |