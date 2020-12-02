#!/usr/bin/env python3

import json
import os, sys
import fnmatch
import numpy as np
import datetime


from pycocotools.coco import COCO
from PIL import Image

#inclue  folder
sys.path.append(os.path.join(sys.path[0],'src'))
import pycococreatortools 



def append_images_to_annotations(original_ann_filename,ROOT_DIR, IMAGE_DIR,ANNOTATION_DIR,output_filename):

    """
    args:
        ROOT_DIR: file where the anonation and the image folders are
        original_ann_filename: json file where we want to append images
        IMAGE_DIR: directory with the images we want to add
        ANNOTATION_DIR: directory with masks of the elements in the images (.png files)
    """
    print("creating anotations..")
    original_ann_filename=os.path.join(ROOT_DIR, original_ann_filename)
    IMAGE_DIR = os.path.join(ROOT_DIR, IMAGE_DIR)
    ANNOTATION_DIR = os.path.join(ROOT_DIR, ANNOTATION_DIR)


    #start with original json file
    with open(original_ann_filename) as json_file:
        coco_output = json.load(json_file)

    image_id = len(coco_output['images']) #start with the last id 
    segmentation_id = len(coco_output['annotations']) #start with the last id of annotations
    CATEGORIES=coco_output['categories']

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = pycococreatortools.filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_filename_path=os.path.basename(IMAGE_DIR)+"/"+os.path.basename(image_filename)                 


            image_info = pycococreatortools.create_image_info(
                image_id, image_filename_path, image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = pycococreatortools.filter_for_annotations(root, files, image_filename)

                if len(annotation_files)==0:
                    print("no ann files "+image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    
                    #print(annotation_filename)
                    #skip BG
                    if "BG" not in annotation_filename:
                        s=os.path.basename(annotation_filename)
                        category_in_file=(s.split("_"))[2].split("_")[0]#extract category
                        class_id = [x["id"] for x in CATEGORIES if x["name"]==category_in_file][0]

                        category_info = {"id": class_id, "is_crowd": 'crowd' in image_filename}
                        binary_mask = np.asarray(Image.open(annotation_filename)
                            .convert('P')).astype(np.uint8)


                        annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=2)

                        if annotation_info is not None:
                            
                            coco_output["annotations"].append(annotation_info)
                        else:
                            print("No_anotation "+ annotation_filename)

                        segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open('{}/{}'.format(ROOT_DIR,output_filename), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print("Finished creating anotations.")


def new_annotations_from_files(data_folder,NEW_IMAGES_DIR,output_filename):
    """
    Create a Json file from images in a folder
    The images should contain the name of the category in between "_". for example '0_Glass bottle_0.jpg'
    """
    folder=data_folder+'/'+NEW_IMAGES_DIR
    #load list of images
    images = os.listdir(folder)

    #get all possible categories
    categories_list=[]
    for s in images:
        category=s[s.find("_")+len("_"):s.rfind("_")]
        categories_list.append(category)
    #delete repeated ones
    categories_list=list(set(categories_list))

    #Prepare to create anotations
    INFO = {
        "description": "New DataSet",
        "url": "https://github.com/",
        "version": "0.1.0",
        "year": 2020,
        "contributor": "gbravoi",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]
    CATEGORIES=[]
    cat_index=0
    for cat in categories_list:
        CATEGORIES.append({"supercategory":cat, "id": cat_index, "name": cat,})
        cat_index+=1

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }
    new_image_id=0
    new_annotation_id=0

    for image_file_name in images:
        pil_image = Image.open(folder+image_file_name)
        new_image_filename=NEW_IMAGES_DIR+image_file_name

        #ignore classes don't used in the other experiment
        if True:#"Cigarette" not in image_file_name and "Battery" not in image_file_name and "Compost" not in image_file_name:
            s=image_file_name
            class_name=(s.split("_"))[1].split("_")[0]#extract category

            image_info = {
                    "id": new_image_id,
                    "file_name": new_image_filename,
                    "width": pil_image.size[0],
                    "height": pil_image.size[1],
                    "date_captured": '',
                    "license": '',
                    "coco_url": '',
                    "flickr_url": ''
            }
            coco_output["images"].append(image_info)

            #noew lest extract the category informatio and create atonation (without segmentation)
            #get category id
            class_id=categories_list.index(class_name)
            annotation_info = {
                "id": new_annotation_id,
                "image_id": new_image_id,
                "category_id": class_id,
                "iscrowd": 0,
                "area": [],
                "bbox": [],
                "segmentation": [],
                "width": pil_image.size[1],
                "height": pil_image.size[0],} 
            coco_output["annotations"].append(annotation_info)

            new_image_id+=1
            new_annotation_id+=1
    #save
    
    with open('{}/{}'.format(data_folder,output_filename), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)