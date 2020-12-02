# put under samples/trash

import os
import sys
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)

from mrcnn import visualize
import mrcnn.model as modellib
from trash import trash

# suppress tf warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# PATH
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
TRASH_DIR = os.path.join(ROOT_DIR, 'datasets/trash')


def get_ax(rows=1, cols=1, size=10):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


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



def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="relative path to image('batch_x/xxx')"
    )
    parser.add_argument(
        "weight",
        help="relative path to weight file"
    )

    args = parser.parse_args()
    IMAGE_PATH = os.path.join(TRASH_DIR, args.image)
    WEIGHT_PATH = os.path.join(TRASH_DIR, args.weight)

    print('Loading weights...', WEIGHT_PATH)
    model.load_weights(WEIGHT_PATH, by_name=True)

    dataset = trash.TrashDataset()
    dataset.load_trash(TRASH_DIR, 'test')
    dataset.prepare()
    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    image_id = dataset.get_image_id(IMAGE_PATH)
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print('image ID: {}.{} ({}) {}'.format(info['source'], info['id'],
                                           image_id, dataset.image_reference(image_id)))

    results = model.detect([image], verbose=1)
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=ax, title='Predictions')
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
