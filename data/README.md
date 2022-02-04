# Data

In this part of the readme we describe the datasets we used and how to create them.

## ImageNet Data

- when we say ImageNet data, we refer to the data that is used for the ImageNet Large Scale Visual Recognition
  Challenge (ILSVRC)
- an overview of the Challenges can be found [here](http://image-net.org/challenges/LSVRC/)
- [paper](https://arxiv.org/pdf/1409.0575.pdf)
- **object categories**
    - total of 1000 synsets
    - synsets follow the WordNet hierarchy (2014)
    - since 2012 the used categories remained consistent
- **data collection**
    - images are retrieved by querying multiple search engines
- **image classification**
    - humans label the images (using Amazon Mechanical Turk) using Wikipedia definition
    - multiple users label each image (at least 10 per image until confidence threshold is passed)
- **statistics**
    - 1000 object classes
    - ~ 1.2 Million training images
    - ~ 50 Thousand validation images
    - ~ 100 Thousand test images

## COCO

- [coco dataset](https://cocodataset.org/)
- COCO - Common Objects in Context
- large-scale object detection, segmentation, and captioning dataset
- general info
    - total of 80 categories
    - each picture can have multiple categories
    - train, val, and test set: more than 200 000 pictures
- definition of multiple challenges:
    - Object Detection
    - Keypoint Detection
    - Stuff Segmentation
    - Panoptic Segmentation
    - Image Captioning
    - DensePose

### Focus

- for us most interesting is the data provided for the **Object Detection** use case
- the data ([can be downloaded here](https://cocodataset.org/#download)) is structured in two parts:
    - the images
    - the annotations
- we need the *train_val annotations* that contain 6 files of 3 categories (train and val split)
    - person_keypoints
    - captions
    - instances
- interesting for us are the *instances* annotations, the annotations data contains the following data
    - general info for file (JSON object)
    - list of licences
    - list of annotations
    - list of categories
- of this data, the *annotations*, and the *categories* are of interest
- they have the following format ([see here](https://cocodataset.org/#format-data)):

 ```json
annotation{
  "id": int,
  "image_id": int,
  "category_id": int,
  "segmentation": RLE
  or [
  polygon
],
"area": float,
"bbox": [x, y, width, height],
"iscrowd": 0 or 1,
}

categories[
{
"id": int,
"name": str,
"supercategory" : str,
}
]
```

## Customized COCO dataset

- the goal of creating our customized COCO dataset is to create a dataset that is similar to/compatible with the
  ImageNet dataset but from a different distribution than the ImageNet data
- also, we want to split the subset of the COCO data that we use into multiple classes

### Creation

- The images in the ImageNet data have only one defined category
- Also in most images, you can see only the one object defining the category

- To create a similar dataset, we filter the images:
    - first, we extract all images that have only one assigned category
    - out of these images, we then extract the images that have a category that is also part of the Imagenet dataset
- finally, we take only the filtered images and store them only with the relevant data

- the custom coco dataset is created by executing the [extract-custom-coco](custom/extract_custom_coco.py) script using
  the following arguments
    - `--coco-train-root-path <coco-root>/train2017`
    - `--coco-train-annotations <coco-root>/annotations/instances_train2017.json`
    - `--coco-val-root-path <coco-root>/val2017`
    - `--coco-val-annotations <coco-root>/annotations/instances_val2017.json`
    - `--imagenet-root <imagenet-root>`
    - `--target-root <target-dir>`

### Download

- **download initial data**
    - [coco-train2017](http://images.cocodataset.org/zips/train2017.zip)
    - [coco-val2017](http://images.cocodataset.org/zips/val2017.zip)
    - [coco-annotations2017](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
      (contains `instances_train2017.json` and `instances_val2017.json`)
    - the `<imagenet-root>` needs to contain the following files
        - `ILSVRC2012_devkit_t12.tar.gz`
        - `ILSVRC2012_img_val.tar`
        - unfortunately, we can not include a download link here because a login is required
            - the root link is [here](http://www.image-net.org/challenges/LSVRC/2012/downloads)
            - **WARNING**: after login for download you get redirected to the data for 2010, change the link to 2012
              (http://www.image-net.org/challenges/LSVRC/2012/downloads) before download!
- **customized coco data**
    - [full-dataset](https://owncloud.hpi.de/s/TRCzfvxwyHCRIQr)

## Custom Coco Datasets 512

- for our experiments we create 4 custom coco datasets
    - custom-coco-food-512,custom-coco-outdoor-512,custom-coco-indoor-512,custom-coco-zebra-512
- each dataset contains the first 512 items of its categories
- we created them using the script `create_coco_n.py`

### Download

- [custom-coco-food-512](https://owncloud.hpi.de/s/Pp2f4hdKFvUrYMm)
- [custom-coco-outdoor-512](https://owncloud.hpi.de/s/T083xp5fBt5S7OI)
- [custom-coco-indoor-512](https://owncloud.hpi.de/s/m5XjelcaVm577i4)
- [custom-coco-zebra-512](https://owncloud.hpi.de/s/MgfxezMgdWbvOYu)

## Pretrained Models: Used Data

We refer to the results and models listed in the
[official PyTorch documentation](https://pytorch.org/docs/stable/torchvision/models.html) (last accessed, 01.12.2020)

Essential questions to answer for pre-trained models (AlexNet, VGG-19, ResNet18, ResNet50, Resnet152):

- What data was used to pre-train the models? (short answer: **data of ImageNet challenge 2012**)
- Was the validation dataset used to train the models? (short answer: **NO** )

What data was exactly used to train the models can not be answered by the information given in the documentation.
Nevertheless, we can make some qualified guesses.

What data was used to train the model?

- for all models, the documentation says pre-trained on ImageNet
- furthermore, the provided
  [dataloader uses the ImageNet data from 2012](https://github.com/pytorch/vision/blob/6e7ed49a93a1b0d47cef7722ea2c2f525dcb8795/torchvision/datasets/imagenet.py#L11-L15)
    - thus we can be pretty sure that the data used to pre-train the models is the **data of the ImageNet challenge
      2012**. The data can be downloaded [here](http://image-net.org/challenges/LSVRC/2012/downloads.php#images)

Was the validation set used to train the models?

- good overview of common definitions of train, test, and validation set can be
  found [here](https://machinelearningmastery.com/difference-test-validation-datasets/)
- they give the following definitions:
    - Training Dataset: The sample of data used to fit the model.
    - Validation Dataset: The sample of data used to provide an unbiased evaluation of a model fit on the training
      dataset while tuning model hyperparameters. The evaluation becomes more biased as skill on the validation dataset
      is incorporated into the model configuration.
    - Test Dataset: The sample of data used to provide an unbiased evaluation of a final model fit on the training
      dataset.
- following these definitions, the validation dataset shouldn't be used to train the model, but it is also said:
    - *the final model could be fit on the aggregate of the training and validation datasets*
- taking a look at the [GitHub issue](https://github.com/pytorch/vision/issues/2469) asking for the code that was used
  to generate the pre-trained models, we can find out that:
    - the validation split is used to generate the *dataset
      test* ([code](https://github.com/pytorch/vision/blob/6e7ed49a93a1b0d47cef7722ea2c2f525dcb8795/references/classification/train.py#L110-L138))
    - this data is **only** used to generate the *data_loader_test*
      ([code](https://github.com/pytorch/vision/blob/6e7ed49a93a1b0d47cef7722ea2c2f525dcb8795/references/classification/train.py#L164))
    - the *data_loader_test* is **only** used in the *evaluate*
      method ([code](https://github.com/pytorch/vision/blob/6e7ed49a93a1b0d47cef7722ea2c2f525dcb8795/references/classification/train.py#L48-L71))
    - the *evaluate* method calculates no gradients and also makes no use of the optimizer
    - thus **the validation set was not used to pretrain the models we use**
