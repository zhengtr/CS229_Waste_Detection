import os
import sys
import matplotlib.pyplot as plt
import util
import distutils.util
import numpy as np
import pandas as pd
import transform
import argparse

from datetime import datetime
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from sklearn.linear_model import SGDClassifier
from skimage.feature import hog
from sklearn.metrics import plot_confusion_matrix

SUBSET_SIZE = 100
RESULTS_FOLDER = "../svm_results"

def parse_command_line():
    """
        Parses the command line arguments.
    """

    parser = argparse.ArgumentParser(description='User args')   
    parser.add_argument("--batch", type=int, required=True, help="Annotation file number.")
    parser.add_argument('--subset_size', type=int, required=False, default=100, \
        help='Set to -1 if you want to train the SVM on the entire dataset.')
    parser.add_argument('--hog', action='store_true', help='Include to extract HOG features.')
    parser.add_argument('--texture', action='store_true', help='Include to extract texture features.')
    parser.add_argument('--vgg', action='store_true', help='Include to use VGG19 to extract features.')
    # Copy previous line to include additional transformations

    return parser.parse_args()

def build_confusion_matrix(classifier, test_X, test_y, label_names, filename):
    """
        Plots the confusion matrix using the given classifier and test dataset.
    """

    figure, axis = plt.subplots(figsize=(9, 9))
    plot_confusion_matrix(classifier, test_X, test_y, \
        display_labels=label_names, \
        cmap=plt.cm.Blues, normalize='true',\
        xticks_rotation=70,\
        ax = axis)
    plt.savefig(filename)

def train_and_predict(train_X, train_y, test_X, test_y, coco, prefix):
    """
        Trains an SVM classifier using the given training dataset.
        Then, makes predictions with the given test dataset.
    """

    # train
    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    sgd_clf.fit(train_X, train_y)
    print(sgd_clf.get_params(deep=True))

    # test
    print("Predicting with trained SVM...")
    y_predictions = sgd_clf.predict(test_X)
    print('Percentage correct: ', 100 * np.sum(y_predictions == test_y) / len(test_y))

    # plot confusion matrix
    unique_labels = np.unique(test_y + list(y_predictions))
    categories = coco.loadCats(unique_labels)
    category_names = [cat["name"] for cat in categories]
    build_confusion_matrix(sgd_clf, test_X, test_y, category_names,\
        "{0}/{1}_confusion_matrix.png".format(results_folder, prefix))

if __name__ == "__main__":
    args = parse_command_line()

    batch_num = args.batch
    filepath = "SVM_ann/annotations_{0}_train.json".format(batch_num)

    timestamp = datetime.now().strftime("%m_%d_%Y_%I_%M_%S_%p")
    results_folder = "{0}/annotations_{1}-{2}".format(RESULTS_FOLDER, batch_num, timestamp)
    os.mkdir(results_folder)

    print("Loading Training Dataset...")
    train_images_X, train_y, train_coco = util.load_taco_dataset(filepath, args.subset_size)

    print("\nLoading Test Dataset...")
    filepath = "SVM_ann/annotations_{0}_test.json".format(batch_num)
    test_images_X, test_y, test_coco = util.load_taco_dataset(filepath, args.subset_size)

    print("\nPlotting category distribution...")
    util.plot_category_distribution(train_coco, test_coco, \
        "{0}/category_distribution.png".format(results_folder))

    if args.hog:
        print("\nTraining SVM with HOG features...")
        train_X_prepared = transform.prepare_hog_data(train_X)
        test_X_prepared = transform.prepare_hog_data(test_X)
        transform.plot_hog_image(train_X[np.random.randint(0, len(train_X))], \
            "{0}/original.png".format(results_folder), "{0}/hog.png".format(results_folder))

        train_and_predict(train_X_prepared, train_y, test_X_prepared, test_y, test_coco, "hog")
    
    if args.texture:
        print("\nTraining SVM with texture features...")
        train_X_prepared = transform.prepare_texture_data(train_images_X)
        test_X_prepared = transform.prepare_texture_data(test_images_X)

        train_and_predict(train_X_prepared, train_y, test_X_prepared, test_y, test_coco, "texture")
   
    if args.vgg:
        print("\nTraining SVM with VGG19 features...")
        train_X_prepared = transform.prepare_vgg_features(train_images_X)
        test_X_prepared = transform.prepare_vgg_features(test_images_X)

        train_and_predict(train_X_prepared, train_y, test_X_prepared, test_y, test_coco, "vgg")

