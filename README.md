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

[Back to top](#table-of-contents)

# Graphs
![Testo alternativo](images/MAE and val_MAE.png)

---
