<h1 align="center">Face blurring project</h1>
  
## Table of contents
- [Introduction](#introduction)
- [Dataset structure](#dataset structure)

# introduction

2 dataset versions, one with a custom built dataset with roughly 3000 images from celebA-HQ and 2900 from a roboflow dataset; the other one is trained using a partition of the vggFace2 found on kaggle, because the original one was way too big.
---
# dataset structure

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

---
