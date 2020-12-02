# put under samples/trash

#################################################################
#  PREPARE
#################################################################

import os
import sys
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)

from mrcnn import utils
import mrcnn.model as modellib
from trash import trash

# suppress tf warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# PATH
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
TRASH_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_trash_0120.h5')
TRASH_DIR = os.path.join(ROOT_DIR, 'datasets/trash')


def get_ax(rows=1, cols=1, size=10):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


#################################################################
#  CONFIG, DATASET & MODEL
#################################################################


# Create config for inference (BATCH_SIZE = 1)
class InferenceConfig(trash.TrashConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create test dataset
dataset = trash.TrashDataset()
dataset.load_trash(TRASH_DIR, 'test')
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Use if on CPU
# Create model
DEVICE = '/cpu:0'
TEST_MODE = 'inference'
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode=TEST_MODE, model_dir=MODEL_DIR, config=config)

print('Loading weights...', TRASH_MODEL_PATH)
model.load_weights(TRASH_MODEL_PATH, by_name=True)


#################################################################
#  DETECT & MATCH GT w/ PRED
#################################################################

def detect_matches(image_id):
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print(f'image ID: {info["source"]}.{info["id"]} ({image_id}) {dataset.image_reference(image_id)}')

    result = model.detect([image], verbose=1)[0]
    gt_match, pred_match, overlaps = \
        utils.compute_matches(gt_bbox, gt_class_id, gt_mask, result['rois'],
                              result['class_ids'], result['scores'], result['masks'])
    return gt_class_id, gt_match, result['class_ids']


#################################################################
#  BUILD CONFUSION MATRIX
#################################################################


l = len(dataset.class_names)
cm = np.zeros((l, l))


def add_to_cm(class_gt, match_gt, class_pred):
    # print(class_gt)
    # print(class_pred)

    match_gt = [int(i) for i in match_gt]
    # add all matched pairs to cm
    matched = {k: match_gt[k] for k in range(len(match_gt)) if match_gt[k] >= 0}
    for m in matched.values():
        cls = class_pred[m]
        cm[cls][cls] += 1

    # get all unpaired matches
    class_pred = [class_pred[i] for i in range(len(class_pred)) if i not in matched.values()]
    class_gt = [class_gt[i] for i in range(len(class_gt)) if i not in matched.keys()]

    # print(class_gt, class_pred)

    while class_gt and class_pred:
        cm[class_gt.pop(0)][class_pred.pop(0)] += 1

    if class_gt:
        for i in class_gt:
            cm[i][0] += 1
    elif class_pred:
        for j in class_pred:
            cm[0][j] += 1

    return cm


#################################################################
#  PLOT
#################################################################


def plot_cm(cm):
    cm_norm = cm / cm.sum(axis=1, keepdims=1)
    print(cm_norm)
    labels = dataset.class_names
    labels[0] = 'None'

    fig, ax = plt.subplots()
    cmap = plt.cm.Blues
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)


    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(labels, Fontsize=5)
    ax.set_yticklabels(labels, Fontsize=5)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Ground Truth')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = (cm_norm.max() + cm_norm.min()) / 2.0
    for i in range(8):
        for j in range(8):
            color = cmap_max if cm_norm[i, j] < thresh else cmap_min
            text = ax.text(j, i, round(cm_norm[i, j], 2),
                           ha="center", va="center", color=color)

    fig.colorbar(im)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    for image_id in dataset.image_ids:
        class_gt, match_gt, class_pred = detect_matches(image_id)
        add_to_cm(class_gt, match_gt, class_pred)
        print(cm)

    plot_cm(cm)
