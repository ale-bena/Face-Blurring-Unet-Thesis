<h1 align="center">Face blurring project</h1>
  
## Table of contents
- [Introduction](#introduction)
- [Dataset structure](#dataset-structure)
- [Graphs](#graphs)

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

# Graphs
```
The graphs in the picture indicates the values for the parameters MSE(loss) and MAE for the teacher model. The comparison is between the training values and the ones calculated during the validation phase, which are more true to tell the capacity of the model and suffer less of overfitting. As can be seen the training line continues to decrease, but the validation line converges earlier. This indicates that the model has reached it's maximum performance and will probably start to overfit if the training goes on too much. The only way to improve the performance is by applying structural modifications to the model, the dataset, or applying more specifical parameters.
<img src="images/MSE%20vs%20val_mse.png" alt="MSE training vs validation values for teacher model" width="400">
<img src="images/MAE%20and%20val_MAE.png" alt="MSE training vs validation values for teacher model" width="400">


---
