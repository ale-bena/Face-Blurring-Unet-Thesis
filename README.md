<h1 align="center">Face blurring project</h1>
  
## Table of contents
- [Introduction](#introduction)
- [Dataset structure](#dataset-structure)
- [Graphs](#graphs)
- [Model architecture](#model-architecture)
- [Training](#training)

# Introduction

This repository containes the code and the information from a thesis work about face blurring. The aim is to provide a comparison between two differen ways to blur faces: one is using pretrained face detection models and implementing a post processing face blurring function that blur the pixel inside the output bounding box and the other way is a pipeline approach, using a unet that learns to blur the faces directly. The second approach is more kean to errors and less precise, but is useful to see how a structure such as a unet can be reduced to fit on an embedded device, which is the main focus of the reserch.
2 dataset versions, one with a custom built dataset with roughly 3000 images from celebA-HQ and 2900 from a roboflow dataset; the other one is trained using a partition of the vggFace2 found on kaggle, because the original one was way too big.
---
# Dataset structure

```
┣ dataset
┃ ┣ train
┃ ┃ ┣00001.jpg
┃ ┃ ┣ ...
┃ ┣ train_blur
┃ ┃ ┣ 00001.jpg
┃ ┃ ┣ ...
┃ ┣ val
┃ ┃ ┣ 00001.jpg
┃ ┃ ┣...
┃ ┣ val_blur
┃ ┃ ┣ 00001.jpg
┃ ┃ ┣ ...
```
The dataset is organized in three different type of folders: training, validation and testing. The dataset has a total number of 13000 images which are splitted 80% for training, 20% for validation and the remaining 10% is dedicated to testing. The 80% of the images of the dataset comes from the VVGFace2 (256x256 version) LINK!!!, this has been done to be coherent to an inspiring research, XimSwap(BIB). These images have a high quality, the face is frontal, visible and has a big dimension(???).
For simplicity reasons all the images contain only one face, so the model might not work well when multiple faces are present in the input image.
For each step there are 3 folders: one with the original images, one with the blurred faces and one with the face masks for the metrics. 

Comment on dataset:
The fact that the major part of the images are frontal and the faces are quite big can lead to overfitting and especially when using an image-to-image learning technique, if the model blurs always near the center part of the image it might blur that region even when the face appears on the side. It also might misunderstand objects in the central region with faces and blur them.
 

[Back to top](#table-of-contents)

# Model architecture
```
The model architecture is based on a simple unet structure, which is a convolutial network with a downsample(encoder) and an upsample(decoder) path. This type of path is common in image reconstruction or detection tasks.
In specific the architecture od the model in analysis is a 3 layer encoder and 3 layer decoder architecture wirh the following filters: 32-64-128 for encoder, and opposite for the decoder. The bottleneck(deppest point of the network) has 256 filters.
(To provide more generalization batch normalization has been added to each layer an also a dropout has been added to the deepest layer of the encoder(0,05) and to the bottleneck(0,2).)
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (None, 128, 128,  │          0 │ -                 │
│ (InputLayer)        │ 3)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d (Conv2D)     │ (None, 128, 128,  │        896 │ input_layer[0][0] │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_1 (Conv2D)   │ (None, 128, 128,  │      9,248 │ conv2d[0][0]      │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d       │ (None, 64, 64,    │          0 │ conv2d_1[0][0]    │
│ (MaxPooling2D)      │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_2 (Conv2D)   │ (None, 64, 64,    │     18,496 │ max_pooling2d[0]… │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_3 (Conv2D)   │ (None, 64, 64,    │     36,928 │ conv2d_2[0][0]    │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_1     │ (None, 32, 32,    │          0 │ conv2d_3[0][0]    │
│ (MaxPooling2D)      │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_4 (Conv2D)   │ (None, 32, 32,    │     73,856 │ max_pooling2d_1[… │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_5 (Conv2D)   │ (None, 32, 32,    │    147,584 │ conv2d_4[0][0]    │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_2     │ (None, 16, 16,    │          0 │ conv2d_5[0][0]    │
│ (MaxPooling2D)      │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_6 (Conv2D)   │ (None, 16, 16,    │    295,168 │ max_pooling2d_2[… │
│                     │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_7 (Conv2D)   │ (None, 16, 16,    │    590,080 │ conv2d_6[0][0]    │
│                     │ 256)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_transpose    │ (None, 32, 32,    │    131,200 │ conv2d_7[0][0]    │
│ (Conv2DTranspose)   │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate         │ (None, 32, 32,    │          0 │ conv2d_transpose… │
│ (Concatenate)       │ 256)              │            │ conv2d_5[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_8 (Conv2D)   │ (None, 32, 32,    │    295,040 │ concatenate[0][0] │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_9 (Conv2D)   │ (None, 32, 32,    │    147,584 │ conv2d_8[0][0]    │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_transpose_1  │ (None, 64, 64,    │     32,832 │ conv2d_9[0][0]    │
│ (Conv2DTranspose)   │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_1       │ (None, 64, 64,    │          0 │ conv2d_transpose… │
│ (Concatenate)       │ 128)              │            │ conv2d_3[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_10 (Conv2D)  │ (None, 64, 64,    │     73,792 │ concatenate_1[0]… │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_11 (Conv2D)  │ (None, 64, 64,    │     36,928 │ conv2d_10[0][0]   │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_transpose_2  │ (None, 128, 128,  │      8,224 │ conv2d_11[0][0]   │
│ (Conv2DTranspose)   │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_2       │ (None, 128, 128,  │          0 │ conv2d_transpose… │
│ (Concatenate)       │ 64)               │            │ conv2d_1[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_12 (Conv2D)  │ (None, 128, 128,  │     18,464 │ concatenate_2[0]… │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_13 (Conv2D)  │ (None, 128, 128,  │      9,248 │ conv2d_12[0][0]   │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_14 (Conv2D)  │ (None, 128, 128,  │         99 │ conv2d_13[0][0]   │
│                     │ 3)                │            │                   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 1,925,667 (7.35 MB)
 Trainable params: 1,925,667 (7.35 MB)
 Non-trainable params: 0 (0.00 B)
[Back to top](#table-of-contents)
```
# Training 
```
```
# Graphs
```
The graphs in the picture indicates the values for the parameters MSE(loss) and MAE for the teacher model. The comparison is between the training values and the ones calculated during the validation phase, which are more true to tell the capacity of the model and suffer less of overfitting. As can be seen the training line continues to decrease, but the validation line converges earlier. This indicates that the model has reached it's maximum performance and will probably start to overfit if the training goes on too much. The only way to improve the performance is by applying structural modifications to the model, the dataset, or applying more specifical parameters.

```

---
