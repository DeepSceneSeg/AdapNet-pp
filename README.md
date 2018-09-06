# AdapNet++:  Self-Supervised Model Adaptation for Multimodal Semantic Segmentation
AdapNet++ is a deep learning model for semantic image segmentation, where the goal is to assign semantic labels to every
pixel in the input image. 

This repository contains our TensorFlow implementation. We provide the codes allowing users to train the model, evaluate
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

| Dataset       | RGB Image     | Segmented Image|
| ------------- | ------------- | -------------  |
| Cityscapes    |<img src="images/city2.png" width=400> |  <img src="images/city2_pred_v2.png" width=400>|
| Forest  | <img src="images/forest2.png" width=400>  |<img src="images/forest2_pred_v2.png" width=400> |
| Sun RGB-D  | <img src="images/sun1.png" width=400>  | <img src="images/sun1_pred_v2.png" width=400>|
| Synthia  | <img src="images/synthia1.png" width=400>  | <img src="images/synthia1_pred_v2.png" width=400> |
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

* Use checkpoint in init_checkpoint for network intialization
* The initial BN layer of the network learns the appropriate transformation of input data during training. Therefore, no mean subtraction needed.
#### Data

* Augment the default dataset -> augmented-training-dataset.
  In our case, we first resized the dataset to (768,384) and then augmented it.
  (random_flip, random_scale and random_crop)

* Convert augmented-training-dataset/val-dataset/test-dataset into .tfrecords format.
  Prepare a .txt file as follows:
  ```
     path_to_modality1/0.png path_to_modality2/0.png path_to_label/0.png
     path_to_modality1/1.png path_to_modality2/1.png path_to_label/1.png
     path_to_modality1/2.png path_to_modality2/2.png path_to_label/2.png
     ...
  ```
  Run from dataset folder:
  ```
     python convert_to_tfrecords.py --file path_to_.txt_file --record tf_records_name 
  ```
  (Input to model is in BGR and 'NHWC' form)
#### Training
```
    gpu_id: id of gpu to be used
    model: name of the model
    num_classes: number of classes
    intialize:  path to pre-trained model
    checkpoint: path to save model
    train_data: path to dataset .tfrecords
    batch_size: training batch size
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
    batch_size: evaluation batch size
    skip_step: how many steps to print mIoU
    height: height of input image
    width: width of input image
    
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
## Additional Notes:
   * We provide only single scale evaluation script. Multi-Scales+Flip evaluation will further imporve the model's performance.
   * We provide only single gpu training script. Training on multiple gpus using synchronized batch normalization with larger batch size will furthur improve the model's performance.
