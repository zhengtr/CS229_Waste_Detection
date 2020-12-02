
# CS 229 (Machine Learning, Fall 2020) Final Project.
Title: Waste Detection Using Different Deep Kearning Methods 

This is the repository cs229 final project.
Created by Brynne Hurst, Tanran Zheng and Gabriela Bravo-Illanes.



# Setup
You can install or conda enviroment using the enviroment.yml file


# Data procesing scripts:
**Notebooks:**
To run different data processing you can use the available notebooks

Augment_data.ipynb: You will need a .json file in coco format and the images asociated to that data set. This notebook will execute functions using the imgaug library to augment the data set and the mask for each object. Images from different categories can be augmented a different number of times. The output are the aumented images and a new .json file containing the annotations of the original data set plus the new one.

Create_merged_categories.ipynb: Code that allow to create new labels for images with more than one element on it. (This code wasn't used in the final project)

Crop_images.ipynb: Crop the objects from the images using their bounding boxes. Then creates a .json file in coco formar of the data set composed of the cropped images. Used to create the data set for the SVM.

Delete_categories.ipynb: Will edit a .json file removing the annotations of deleted categories.

Inspec_data.ipynb: Allow to inspect specific images from a data set (visualizong the masked objects), plot the taxonomy of the data set, and report the number of annotations per category.

map_categories.ipynb: Allow to relabel categories. Use for mapping .csv files.

remove_small_objects.ipynb: funtions allows to remove annotations from the .json file from objects smaller than a threshold. Objects are deleted if image_width/bounding_box_width && image_height/bounding_box_height < threshold.

split_datasets.ipynb: Split data set using split_dataset.py. By default is 80% training, 10% validation and 10% test. It also can create multiple splits to perform k-fold cross-validation
<br />
<br />


**Scripts:**
The previous notebook uses functions from the files:

data_procesing.py: containing different functions to process the data set

images_to_coco.py: contain functions to create the new .json files

pycococreatortools.py: Adapted file from  https://github.com/waspinator/pycococreator/. contain functions to transform images with mask information (created during augmentation) in annotations.

split_dataset.py: Adapted file from https://github.com/pedropro/TACO. Contains function to split the data set.

util.py: functions to extract data from the data set in coco format.



# SVM Classifier
Runs a stochastic gradient descent SVM classifier on the TACO dataset and outputs a confusion matrix.

## To run
```
python svm.py --batch 0 -hog --subset_size -1 -- hog
```

The `--batch` argument specifies which annotation file to use.

Setting `--subset_size` to -1 runs the SVM with the entire dataset. Set `--subset_size` to a positive value to train with a subset of the data.

The `--hog` parameter runs the SVM with the HOG feature descriptor. 

To run with the Haralick feature descriptor, include the `--texture` argument. 

To run with the VGG19 feature descriptor, run with `--vgg`.




# YOLO
YOLO is a family of object detection models. This part of the code trains a YOLOv4-Tiny model on the customized TACO dataset using `darknet` framework. The trained model can be used to do object detection task on 
images or video. 

## To run
To train the model, run from `darknet` directory with:
```
./darknet detector train trash_data/trash.data trash_data/cfg/yolov4-tiny-trash.cfg <path_to_weight_file> 
```
Before running, make sure to update accordingly:
1. All images should have corresponding `.txt` file that includes class and bbox for each instance. 
2. Include a `train.txt` of paths to all training images, a `val.txt` of paths to all testing images, and a `trash.names` of all class labels.

To generate the above files, use `trash_data/get_yolo_files.py`.

To do transfer learning, use `yolov4-tiny.conv.29` as the weight file. (Get from `darknet` original repo.)


After training is complete, run from `darknet` directory for detection:
```
./darknet detector test trash_data/trash.data trash_data/cfg/yolov4-tiny-trash.cfg <path_to_weight_file> <path_to_image_file>
```
To check results of the trained model:
1. Generate mAP and IoU plot using `trash_data/plot_map_yolo.py`.
2. Generate confusion matrix using `trash_data/cm.py`.
3. `trash_data/plot_metrics.py` also provides functions to plot other metrics like PR, F1 scores.
4. To visually compare the ground truth vs. prediction result, use `trash_data/visualize.py` to draw bounding boxes.

Note: The YOLO model is developed based on `darknet`, please refer to https://github.com/AlexeyAB/darknet for more detailed explanation.


# Mask_RCNN
Mask R-CNN is an object detection model. This part of the code trains a Mask R-CNN model with TACO dataset. The trained model can be used to detect objects given an input image.

## To run
To train the model, run from `Mask_RCNN` directory with:
```
python trash/trash.py train --dataset=<path_to_data_dir> --model=<path_to_weight_file>
```
where the data directory should contain annotations files for both training and testing.

After training is complete, run from `Mask_RCNN/trash` directory for detection:
```
python detection.py <path_to_image_file> <path_to_weight_file>
```
To check results of the trained model:
1. Get mAP and plot with `trash/map.py` and `trash/plot_map_rcnn.py`.
2. Generate confusion matrix with `trash/cm.py`.

Note: The RCNN model is developed based on `Mask_RCNN`, please refer to https://github.com/matterport/Mask_RCNN for more detailed explanation.


