import itertools
import numpy as np
import json

from pycocotools.coco import COCO

from pycocotools import mask as maskUtils

#import skimage.io
from PIL import Image
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import skimage.io
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import imgaug as ia

data_dir = "../data"

def count_images(coco):
    """
        Helper function for plotting category distributions.
        Counts the number of images in each category.

        Args:
            category_ids: The category ids.
            coco: The COCO dataset.
    """

    category_ids = coco.getCatIds()

    num_images = 0
    count_images = []
    for id in category_ids:
        images = coco.getImgIds(catIds=id)
        count_images.append(len(images))
        num_images += len(images)

    count_images = [(count / num_images)*100 for count in count_images]
    return count_images, num_images

# def get_image_info(coco,image_id):
#     """
#         return filename, width and height based on image_id

#         Args:
#             coco: The COCO dataset.
#             image_id: image id
#     """
#     image_info={}
#     image_info["width"]=coco.imgs[image_id]['width']
#     image_info["height"]=coco.imgs[image_id]['height']
#     image_info["file_name"]=coco.imgs[image_id]['file_name']
#     return image_info


def load_image(coco,image_id,path):
    """
    return image as array 3 channels
        Args:
            coco: The COCO dataset.
            image_id: image id
            path: relative location folder data
    """
    image_path=coco.imgs[image_id]['file_name']
    #image = image = Image.open(path+image_path)
    image = skimage.io.imread(path+image_path)
    return image

def load_image_PIL(coco,image_id,path):
    """
    return image as array 3 channels
        Args:
            coco: The COCO dataset.
            image_id: image id
            path: relative location folder data
    """
    image_path=coco.imgs[image_id]['file_name']
    image = image = Image.open(path+image_path)
    #image = skimage.io.imread(path+image_path)
    return image

def load_image_masks(coco,image_id,image_shape):
    masks={}
    number_annotations_on_image=len(coco.loadAnns(coco.getAnnIds(image_id)))
    for i in range(number_annotations_on_image):
        annotation=coco.loadAnns(coco.getAnnIds(image_id))[i]
        segmentation=coco.annToMask(annotation)
        segmap=segmentation.astype(np.uint8)
        segmap= SegmentationMapsOnImage(segmap, shape=image_shape)
        category_id=coco.loadAnns(coco.getAnnIds(image_id))[i]['category_id']
        masks[i]={
                "category": coco.cats[category_id]['name'] ,
                "segmentation":segmap
            }
        
    return masks

def showimage_with_masks(coco,image_id,data_path):
    """
    Show original image and the elements with masks on it
    """
    #load images and mask
    image = load_image(coco,image_id,data_path)
    masks = load_image_masks(coco,image_id,image.shape)
    
    #add all masks to image
    for i in range(len(masks)):
        mask_im=masks[i]["segmentation"]
        image=mask_im.draw_on_image(image)[0] 
    ia.imshow(image)

def extract_bboxes(coco,image_id):
    """
    Compute bounding boxes from masks. (This information is also in the annotations)
    return bbox (y1, x1, y2, x2) and the category of the element in the box
    arg:
        coco: coco dataset
        image_id: image_id of interest.
    """
    bbox={}
    number_annotations_on_image=len(coco.loadAnns(coco.getAnnIds(image_id)))
    for i in range(number_annotations_on_image):
        #load mask 
        annotation=coco.loadAnns(coco.getAnnIds(image_id))[i]
        category_id=annotation['category_id']
        segmentation=coco.annToMask(annotation)
        RS=maskUtils.encode(segmentation)
        bbox[i]={
            "boxes":maskUtils.toBbox(RS),#np.array([x,y,w,h]),
            "category":coco.cats[category_id]['name'] 
        }

    return bbox

def plot_horizontal_bar(y, x, loc='left', relative=True):
    """
        Helper function for plotting category distributions.
        Plots a horizontal bar chart.

        Args:
            y: The y values for the bar chart.
            x: The x values for the bar chart.
    """

    width = 0.35
    if loc == 'left':
        n = -0.5
    elif loc == 'right':
        n = 0.5
 
    plt.barh(np.array(y) + n*width, x, align='center', alpha=.7, height=width)

def get_categories_names(coco):
    """
    Return a list with the categories names
    arg:
        coco: coco dataset
    """
    category_ids=coco.getCatIds()
    categories = coco.loadCats(category_ids)
    categories_names = [cat["name"] for cat in categories]
    return categories_names

def plot_annotations_per_category(annotations):
    """
    Plot annotatiosn per category
    args:
        annotations: path+file name of annotations .json file
    """
    #load dataset
    coco = COCO(annotations)

    # get catgeory information
    number_categories=len(coco.getCatIds())
    class_names = get_categories_names(coco)

    #count annotations
    cat_histogram = np.zeros(number_categories,dtype=int)
    for i in range(len(class_names)):
        ann_per_cat = coco.getAnnIds(catIds=i, iscrowd=None)
        cat_histogram[i] = len(ann_per_cat)



    # Initialize the matplotlib figure
    _, ax = plt.subplots(figsize=(5,0.5*len(class_names)))

    # Convert to DataFrame
    d ={'Categories': class_names, 'Number of annotations': cat_histogram}
    df = pd.DataFrame(d)
    df = df.sort_values('Number of annotations', 0, False)

    #drop BG
    df = df.drop(df[ df['Categories'] == 'BG' ].index, axis=0)

    #print dataframe
    print(df)
    # Plot the histogram
    sns.set_color_codes("pastel")
    sns.set(style="whitegrid")
    plot_1 = sns.barplot(x="Number of annotations", y="Categories", data=df,
                label="Total", color="b", orient='h')

    fig = plot_1.get_figure()
    fig.savefig("taxonomy.png",bbox_inches = "tight")

def plot_category_distribution(train_coco, test_coco, filename):
    """
        Plots distributions of the categories in the two COCO datasets.

        Args:
            train_coco: The training COCO datset.
            test_coco: The testing COCO datset.
            filename: Where to save the plot.
    """

    category_ids = train_coco.getCatIds()
    categories = train_coco.loadCats(category_ids)
    category_names = [cat["name"] for cat in categories]

    train_distribution, num_train = count_images(train_coco)
    test_distribution, num_test = count_images(test_coco)

    plot_horizontal_bar(category_ids, train_distribution, loc='left')
    plot_horizontal_bar(category_ids, test_distribution, loc='right')

    plt.tight_layout()
    plt.yticks(category_ids, category_names)
    plt.xlabel("Percentage of images")
    plt.legend([
        'train ({0} photos)'.format(num_train),
        'test ({0} photos)'.format(num_test)
    ])
    plt.savefig(filename)

def load_taco_dataset(file_name, subset_size):
    """
        Loads the data from the given annotation file.
        
        Args:
            file_name: The annotation file to use.
            subset_size: If -1, use entire dataset. 
                Otherwise, only load subset_size images.

        Returns: The design matrix X, the labels y, and the coco dataset
    """

    subset = subset_size != -1

    annotation_file = '{0}/{1}'.format(data_dir, file_name)
    coco = COCO(annotation_file)
    category_ids = coco.getCatIds()
    categories = coco.loadCats(category_ids)

    image_ids = []
    for id in category_ids: 
        # for some reason coco.getImgIds doesn't work with an array of category ids
        image_ids.extend(coco.getImgIds(catIds=id))

    # load images and create numpy array
    num_examples = subset_size if subset else len(image_ids)
    data_x = []; data_x_images = []; data_y = []
    loop_i = 0
    while loop_i < num_examples:
        image_i = np.random.randint(0,len(image_ids)) if subset else loop_i
        image = coco.loadImgs(image_ids[image_i])[0]

        annotationIds = coco.getAnnIds(imgIds=image["id"], catIds=category_ids, iscrowd=None)
        annotations = coco.loadAnns(annotationIds)
        if not annotations: # this image doesn't have any annotations
            continue

        file_path = "{0}/{1}".format(data_dir, image["file_name"].lower().replace("jpg", "npy"))
        image_file_path = "{0}/{1}".format(data_dir, image["file_name"])
        data_x_images.append(image_file_path)
        data_x.append(np.load(file_path, allow_pickle=True))
        data_y.append(annotations[0]["category_id"])

        loop_i += 1

    print("Num examples: {0}".format(len(data_x_images)))
    print("Num labels: {0}".format(len(data_y)))
    
    return data_x_images, data_y, coco
