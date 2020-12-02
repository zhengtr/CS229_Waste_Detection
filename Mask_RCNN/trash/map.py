# put under samples/trash

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)

from mrcnn import utils
import mrcnn.model as modellib
from trash import trash

# suppress tf warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# PATH
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
TRASH_DIR = os.path.join(ROOT_DIR, 'datasets/trash')


class InferenceConfig(trash.TrashConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Use if on CPU
# Create model
DEVICE = '/cpu:0'
TEST_MODE = 'inference'
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode=TEST_MODE, model_dir=MODEL_DIR, config=config)


dataset = trash.TrashDataset()
dataset.load_trash(TRASH_DIR, 'test')
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# mAP@IoU=50 batch of images
def compute_batch_ap(images_ids):
    APs = []
    for image_id in images_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

        results = model.detect([image], verbose=0)

        r = results[0]
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)
    return APs

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weight_dir", required=False,
        default='trash20201115T0815',
        help="relative path to image('batch_x/xxx')"
    )

    args = parser.parse_args()
    weights_dir = os.path.join(MODEL_DIR, args.weight_dir)
    weights = [w for w in os.listdir(weights_dir) if w.endswith('weights')]

    for weight in weights:
        weight_path = os.path.join(weights_dir, weight)
        print('Loading weights...', weight)
        model.load_weights(weight_path, by_name=True)

        image_ids = dataset.image_ids
        APs = compute_batch_ap(image_ids)
        print('mAP @ IoU=50: ', np.mean(APs))


if __name__ == "__main__":
    main(sys.argv)