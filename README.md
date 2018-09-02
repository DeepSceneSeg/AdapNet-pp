# AdapNet++:  Self-Supervised Model Adaptation for Multimodal Semantic Segmentation
AdapNet++ is a deep learning model for semantic image segmentation, where the goal is to assign semantic labels to every
pixel in the input image. 

This repository contains our TensorFlow implementation. We provide codes allowing users to train the model, evaluate
results in terms of mIoU(mean intersection-over-union). 

If you find the code useful for your research, please consider citing our paper:
```
@article{valada18SSMA,
author = "Abhinav Valada, Rohit Mohan and Wolfram Burgard",
title = "Self-Supervised Model Adaptation for Multimodal Semantic Segmentation",
journal = "arXiv preprint arXiv:1808.03833",
month = "August",
year = "2018",
}
```

## Some segmentation results:

| Dataset       | RGB_Image     | Segmented_Image|
| ------------- | ------------- | -------------  |
| Cityscapes    |<img src="images/city2.png" width=400> |  <img src="images/city2_pred_v2.png" width=400>|
| Forest  | <img src="images/forest2.png" width=400>  |<img src="images/forest2_pred_v2.png" width=400> |
| Sun  | <img src="images/sun1.png" width=400>  | <img src="images/sun1_pred_v2.png" width=400>|
| Synthia  | <img src="images/synthia1.png" width=400>  | <img src="images/sun1_pred_v2.png" width=400> |
| Scannetv2  | <img src="images/scannet1.png" width=400>  |<img src="images/scannet1_pred_v2.png" width=400> |


## System requirement

#### Programming language
```
Python 2.7
```
#### Python Packages
```
tensorflow-gpu 1.4.0
```
## Configure the network

Use checkpoint in init_checkpoint for network intialization

#### Training
```
    gpu_id: id of gpu to be used
    model: name of the model
    num_classes: number of classes
    intialize:  path to pre-trained model
    checkpoint: path to save model
    train_data: path to dataset .tfrecords
    dataset: name of dataset (cityscapes, forest, scannet, synthia or sun)
    batch_size: training batch size
    type: type of data  (rgb, jet(depth), hha(depth))
    skip_step: how many steps to print loss 
    height: height of input image
    width: width of input image
    max_iteration: how many iterations to train
    learning_rate: initial learning rate
    save_step: how many steps to save the model
    power: parameter for poly learning rate
    
```
#### Evaluation
```
    gpu_id: id of gpu to be used
    model: name of the model
    num_classes: number of classes
    checkpoint: path to saved model
    test_data: path to dataset .tfrecords
    dataset: name of dataset (cityscapes, forest, scannet, synthia or sun)
    batch_size: evaluation batch size
    type: type of data  (rgb, jet(depth), hha(depth))
    skip_step: how many steps to print mIoU
    height: height of input image
    width: width of input image
    
```
#### Data
```
Augment the default dataset -> augmented-dataset.
Convert it into .tfrecords format. (Use features identical to the one given in dataset/helper.py parser function)
             
(Input to model is in BGR and 'NHWC' form)

```
## Training and Evaluation

#### Start training
Create the config file for training in config folder.
Run
```
python train.py -c config cityscapes_train.config or python train.py --config cityscapes_train.config

```

#### Eval

Select a checkpoint to test/validate your model in terms of mean IoU.
Create the config file for evaluation in config folder.

```
python evaluate.py -c config cityscapes_test.config or python evaluate.py --config cityscapes_test.config
```
