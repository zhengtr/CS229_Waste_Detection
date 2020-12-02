import os, sys
import numpy as np
import csv
import json

#augmentation images
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imageio

from pycocotools.coco import COCO

#custom libraries
sys.path.append(os.path.join(sys.path[0],'src'))
import util
import Images_to_coco


def augment_data_set(annotation_file,output_filename,data_path,augment_times=None):
    """
    Function will augment images and the create a new json file
    """
    #create augmented images and masks
    augment_images(annotation_file,data_path,augment_times)
    #create anotation file
    data_path=data_path+'/'
    IMAGE_DIR = "batch_aug"
    ANNOTATION_DIR = "batch_aug_ann"
    Images_to_coco.append_images_to_annotations(annotation_file,data_path, IMAGE_DIR,ANNOTATION_DIR,output_filename)



#image augmentation function
def augment_images(annotation_file,data_path,augment_times):
    """
    This function will create augmented images and the corresponding masks as .jpg files
    """
    print('This may take some time, as we are working with the original image size...')

    annotation_file=data_path+annotation_file
    #load coco anotation
    coco = COCO(annotation_file)

    #get images ids
    images_id_array=coco.getImgIds()
    #images_id_array=[2,3] #for testing


    for image_id in images_id_array:

        # Load image and mask with categories
        image_ori = util.load_image(coco,image_id,data_path)
        masks = util.load_image_masks(coco,image_id,image_ori.shape) 


        #folder to save images and mask
        mask_dir=data_path+"batch_aug_ann/"
        img_dir=data_path+"batch_aug/"

        #number of augmentations can depend on the category

        if augment_times is not None:
            #get categories on image
            cats_on_image=get_cat_on_image(coco,image_id)
            #get maximum number to augment based on the categories in the image
            nr_augmentations=0
            for category in cats_on_image:
                if category in augment_times.keys(): #if we have the number to augment
                    times=augment_times[category]
                    if times>nr_augmentations:
                        nr_augmentations=times
        else:
            nr_augmentations = 1

        image = image_ori

        # Define our augmentation pipeline.
        seq = iaa.Sequential([
                iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="AWGN"),
                #iaa.Affine(rotate=(-15, 15)),  # rotate by -45 to 45 degrees
                iaa.GaussianBlur(sigma=(0.0, 3.0), name="Blur"),
                iaa.Dropout([0.0, 0.05], name='Dropout'), # drop 0-5% of all pixels
                #iaa.Fliplr(0.5),
                iaa.Add((-20, 20),name="Add"),
                iaa.Multiply((0.8, 1.2), name="Multiply"),
                iaa.Affine(scale=(0.8, 1.1)),
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees
            ], random_order=True)



        
        print('image_id:'+str(image_id)+"num augmentations: "+str(nr_augmentations))
        # Augment images and masks
        for i in range(nr_augmentations):
            image_aug=None
            segmentations_aug=[]
            seq_det = seq.to_deterministic()
            image_aug_name=img_dir+str(image_id)+"_"+str(i)+".jpg"
            #check all masks
            for j in range(len(masks)):
                segmap=masks[j]["segmentation"]
                image_aug, segmap_aug = seq_det.augment(image=image, segmentation_maps=segmap)
                segmentations_aug.append(segmap_aug)
                category=masks[j]["category"]
                mask_name=str(mask_dir)+str(image_id)+"_"+str(i)+"_"+category+"_"+str(j)+".png"
                imageio.imwrite(mask_name, segmap_aug.draw()[0]) #save mask as image

            #save image to folder
            imageio.imwrite(image_aug_name, image_aug)

            #plot image
            #ia.imshow(np.hstack([ x.draw_on_image(image_aug)[0] for x in segmentations_aug
            #]))
    print("I finished procesign images!")


def replace_dataset_classes( original_json, file_new_classes,ouput_json):
    """ 
    Replaces classes of dataset based on a csv file
    arg:
        original_json: path+file name of original json file
        file_new_classes: path+filename csv file with math of categories
        ouput_json: path+filename of the output json file
    """
    #open csv
    with open(file_new_classes) as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]:row[1] for row in reader}

    #load coco dataset
    with open(original_json) as json_file:
        coco_output = json.load(json_file)

    class_new_names = list(set(class_map.values()))
    class_new_names.sort()
    class_originals = coco_output['categories'].copy()
    new_categories = []
    class_ids_map = {}  # map from old id to new id

    # Replace categories
    for id_new, class_new_name in enumerate(class_new_names):

        category = {
            'supercategory': class_new_name,
            'id': id_new,  # Background has id=0
            'name': class_new_name,
        }
        new_categories.append(category)
        # Map class names
        for class_original in class_originals:
            if class_map[class_original['name']] == class_new_name:
                class_ids_map[class_original['id']] = id_new

    # Update annotations category id tag
    for ann in coco_output['annotations']:
        ann['category_id'] = class_ids_map[ann['category_id']]

    
    #replace categories
    coco_output['categories']=new_categories
    

    #save file
    with open(ouput_json, 'w') as output_json_file:
            json.dump(coco_output, output_json_file)



def crop_mask_from_images(annotations,NEW_IMAGES_DIR,width,height):
    """
    crop images with a box in the segmented area
    Save it as an image with the name of the object in it.
    args:
        annotations: path+  json file with information of the images and their segmentations
        NEW_IMAGES_DIR: path where the new images will be saved
        width, height output image dimensions
    """


    #load coco anotation
    coco = COCO(annotations)
    ROOT_DIR = os.path.dirname(annotations)+"/"

    images_list=coco.getImgIds()
    #images_list=[272]

    #foor loop for all images and all their mask
    for image_id in images_list:
        print("imag id:"+str(image_id))
        #load image and mask
        image=util.load_image_PIL(coco,image_id,ROOT_DIR)

        # Compute Bounding boxes of all masks
        bbox_and_cat = util.extract_bboxes(coco,image_id)

        #go trough all masks
        #select image to crop
        for index in  range(len(bbox_and_cat)):
            bbox=bbox_and_cat[index]["boxes"]
            #firs bbbox
            x1=(int)(bbox[0])
            y1=(int)(bbox[1])
            x2=(int)(bbox[2]+bbox[0])
            y2=(int)(bbox[3]+bbox[1])
            # #do a square image
            # side=max(abs(y2-y1),abs(x2-x1))
            # new_y1=y1-side//2
            # new_y2=y2+side//2
            # new_x1=x1-side//2
            # new_x2=x2+side//2
            # #check we are inside the picture
            # if new_y1<0:
            #     new_y1=y1
            #     new_y2=y1+2*side//2
            # if new_y2>image.size[0]:
            #     new_y2=y2
            #     new_y1=y2-2*side//2
            # if new_x1<0:
            #     new_x1=x1
            #     new_x2=x1+2*side//2
            # if new_x2>image.size[1]:
            #     new_x2=x2
            #     new_x1=x2-2*side//2
            # image_crop = image.crop((new_x1, new_y1, new_x2, new_y2)) 
            image_crop = image.crop((x1, y1, x2, y2)) #crop the exact bounding box
            resized = image_crop.resize((width, height))

            #now save information of the image
            class_name=bbox_and_cat[index]["category"]
            new_image_filename=NEW_IMAGES_DIR+str(image_id)+'_'+class_name+'_'+str(index)+'.jpg'
            resized.save(ROOT_DIR+"/"+new_image_filename)

def get_cat_on_image(coco,image_id):
    annotations_ids=coco.getAnnIds(imgIds=image_id)
    cat_in_image=[]
    for j in range(len(annotations_ids)):
        annotation=coco.loadAnns(annotations_ids[j])[0]
        cat_id=annotation["category_id"]
        category=coco.loadCats(cat_id)[0]["name"]
        cat_in_image.append(category)
    #sort and remove repeated
    cat_in_image = list(dict.fromkeys(cat_in_image))
    cat_in_image.sort()
    return cat_in_image

def create_merged_categories(annotations,output_json,categories=[]):
    """
    To the images with more than one object, it is designated a new category
    with the name of all objects in the image ordered alphabetically
    This images won't contain segmentation, are creates for the SVM
    arg:
        annotations: json file with annotations
        output_json: name of the output json file
    """
    global new_images_id
    global new_anotation_id
    global categories_to_keep

    def determine_merged_category(coco,image_id):
        cat_in_image=[]
        #determine categories in image
        annotations_ids=coco.getAnnIds(imgIds=image_id)
        if len(annotations_ids)==1:
            annotation=coco.loadAnns(annotations_ids[0])[0]
            cat_id=annotation["category_id"]
            category=coco.loadCats(cat_id)[0]["name"]
        else:
            #determine categories in image
            cat_in_image=get_cat_on_image(coco,image_id)

            
            #write string for the name of the new category
            category = '_'.join(str(i) for i in cat_in_image) 
        return category

    def add_anotation(new_image_id,category):
        global new_images_id
        global new_anotation_id
        global categories_to_keep
        
        annotation_info = {
            "id": new_anotation_id,
            "image_id": new_image_id,
            "category_id": categories_to_keep.index(category),
            "iscrowd": 0,
            "area": "",
            "bbox": "",
            "segmentation": "",
        } 

        coco_output["annotations"].append(annotation_info)
        new_anotation_id+=1
    
    def add_image(image_id):
        global new_images_id
        image_filename_path=coco_old["images"][image_id]["file_name"]

        image_info = {
                "id": new_images_id,
                "file_name": image_filename_path,
                "width": coco.loadImgs(ids=image_id)[0]["width"],
                "height": coco.loadImgs(ids=image_id)[0]['height'],
                "date_captured": coco_old["images"][image_id]["date_captured"],
                "license": coco_old["images"][image_id]["license"],
                "coco_url": coco_old["images"][image_id]["coco_url"],
                "flickr_url": coco_old["images"][image_id]["flickr_url"]
        }
        coco_output["images"].append(image_info)
        new_images_id+=1
        return new_images_id-1
    
    ROOT_DIR = ROOT_DIR = os.path.dirname(annotations)
    #load data in coco format
    coco = COCO(annotations)

    images_ids=coco.getImgIds()

    #Check combinations present on images
    class_names = {}
    for image_id in images_ids:#check all images
        category=determine_merged_category(coco,image_id)

        #add in dict of categories
        if category in class_names:
            class_names[category]+=1
        else:
            class_names[category]=1

    ######################################EDITABLE #######################33
    #decide wich category to keep
    if len(categories)==0:#if wasn't specified what to keep, keep all
        categories_to_keep=list(class_names.keys())
    else:
        categories_to_keep=categories

    #we could keep the ones with a minimum number of samples
    #min_sample=20
    #categories_to_keep=all_categories.loc[all_categories.quantity>min_sample]['categories'].tolist()
    #or keep all
    # categories_to_keep=all_categories['categories'].tolist()

    #open json file to mod
    with open(annotations) as json_file:
        coco_old = json.load(json_file)

    #create a new output using only the things we will keep
    coco_output = {
        "info": coco_old["info"],
        "licenses": coco_old["licenses"],
        "images": [],
        "annotations": [],
        "categories": []
    }

    #append categories
    cat_id=1
    for cat in categories_to_keep:
        category = {
            'supercategory': cat,
            'id': cat_id,  
            'name': cat,
        }
        coco_output['categories'].append(category)
        cat_id+=1

    new_images_id=1
    new_anotation_id=1




        #go though all images
    for image_id in images_ids:#check all images
        category=determine_merged_category(image_id)
        
        if category in categories_to_keep:
            new_image_id=add_image(image_id)
            add_anotation(new_image_id,category)
       
    #save file
    
    with open('{}/{}'.format(ROOT_DIR,output_json), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)


def drop_categories(original_ann_filename,remove_category_list,output_filename):

    """
    Create a new json file, but without certain categoties
    args:
        ROOT_DIR: file where the anonation and the image folders are
        original_ann_filename: json file where we want to append images
        output_filename: name of the json file with the new annotations
    """
    print("creating anotations..")
    ROOT_DIR=os.path.dirname(original_ann_filename)+'/'

    #start with original json file
    with open(original_ann_filename) as json_file:
        coco_output = json.load(json_file)

    annotations_original=coco_output['annotations'].copy()
    class_originals = coco_output['categories'].copy()
    new_categories = []
    class_ids_map = {}  # map from old id to new id

    # Reorder categories, skiping drop categories
    cat_index=0
    for category in class_originals:
        category_name=category["name"]
        #if not in drop list
        if category_name not in remove_category_list:
            class_ids_map[category["id"]] = cat_index #save id to map
            category["id"]=cat_index #update id
            cat_index+=1
            new_categories.append(category)#appedn to list of categories

    ann_index=1
    new_annotations=[]
    # Update annotations category id tag
    for ann in annotations_original:
        old_cat_id=ann['category_id'] 
        old_cat_name=class_originals[old_cat_id]["name"]
        if old_cat_name not in remove_category_list:
            ann['id']=ann_index#update annotation id
            ann_index+=1
            ann['category_id'] = class_ids_map[ann['category_id']]#map category number
            #add annotation
            new_annotations.append(ann)

        
    #replace categories
    coco_output['categories']=new_categories
    #replace nnotations
    coco_output['annotations']=new_annotations


    with open('{}/{}'.format(ROOT_DIR,output_filename), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print("Finished creating anotations.")

    #the previous file may have images without annotation, delete those
    delete_images_no_annotations(output_filename,output_filename,ROOT_DIR)



def delete_images_no_annotations(ann_filename,output_filename,data_path):
    """
    search for images with to annotations and delete them of the json file
    """
    print("checking for images without annotations")
    #load coco anotation
    ann_filename=data_path+'/'+ann_filename
    coco = COCO(ann_filename)
    #load jason file
    with open(ann_filename) as json_file:
        coco_output = json.load(json_file)

    #get images
    list_images=coco_output["images"].copy()
    annotations_original=coco_output['annotations'].copy()

    #search for images without annotation, re enumerate images
    IMAGES=[]
    images_map={}
    new_image_id=0
    for image in list_images:
        image_id=image["id"]
        #add images that have more than 0 annotation
        if len(coco.getAnnIds(imgIds=image_id)):
            images_map[image["id"]]=new_image_id#save map for images
            image["id"]=new_image_id
            new_image_id+=1
            IMAGES.append(image)

    #we need to remap images 
    ANNOTATIONS=[]
    for ann in annotations_original:
        ann["image_id"]=images_map[ann["image_id"]]
        ANNOTATIONS.append(ann)
    
    #replace in coco output
    coco_output["images"]=IMAGES
    coco_output["annotations"]=ANNOTATIONS

    #save
    with open('{}/{}'.format(data_path,output_filename), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print("Finished deliting files without anotations.")


def search_images_no_annotations(ann_file,data_path):
    """
    print image_id of images without annotations
    """
    #load test image
    ann_file=data_path+'/'+ann_file
    coco = COCO(ann_file)
    all_images=coco.getImgIds()
    for image_ids in all_images:
        if len(coco.getAnnIds(imgIds=image_ids))==0:
            print(image_ids)


def remove_ann_of_small_objects(original_ann_filename,output_json,path,threshold,log=False):
    """
    Function will create a new jason file deliting the anotation of the obejcts smaller than a threshold
    args:
        original_ann_filename: original json file
        output_json: name of the output file
        path: path with the data
        threshold: fraction (number form 0 to 1) of with/height of the object with respect the complete image
    """
    print("start removing small objects")
    original_ann_filename=path+original_ann_filename
    #start with original json file
    with open(original_ann_filename) as json_file:
        coco_output = json.load(json_file)

    #annotations
    ori_annotations_list=coco_output["annotations"].copy()


    ANNOTATIONS=[]
    ann_index=1
    #go trough all annotations
    for ann in ori_annotations_list:
        #image of the annotation
        image_id=ann["image_id"]
        image_info=coco_output["images"][image_id]
        image_height=image_info["height"]
        image_width=image_info["width"]
        #check size of the box
        bbox=ann['bbox']
        bbox_width=bbox[2]
        bbox_height=bbox[3]

        #if the box is smaller that treshold (both dimensions), do not include
        if bbox_height/image_height>threshold or bbox_width/image_width>threshold:
            ann["id"]=ann_index
            ann_index+=1
            ANNOTATIONS.append(ann)
        else:
            if log:
                category_id=ann["category_id"]
                print("annotation" + str(ann["id"]) +" from image "+str(image_id)+" category:"+ str(category_id)+" removed")

    #save new annotation file
    coco_output["annotations"]=ANNOTATIONS
    with open('{}/{}'.format(path,output_json), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)

    #delete images with no annotations
    delete_images_no_annotations(output_json,output_json,path)


