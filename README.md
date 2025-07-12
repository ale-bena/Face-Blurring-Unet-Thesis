# face_blur_project

2 dataset versions, one with a custom built dataset with roughly 3000 images from celebA-HQ and 2900 from a roboflow dataset; the other one is trained using a partition of the vggFace2 found on kaggle, because the original one was way too big.

# Structure of the dataset
'''
|- dataset
|  |- train
|  |  |- 00001.jpg
|  |  |-...
|  |- train_blur
|  |  |- 00001.jpg
|  |  |-...
|  |- val
|  |  |- 00001.jpg
|  |  |-...
|  |- val_blur
|  |  |- 00001.jpg
|  |  |-...

'''
