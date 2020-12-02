import numpy as np

import cv2
import mahotas

import skimage
import skimage.color
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
        Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
 
class HogTransformer(BaseEstimator, TransformerMixin):
    """
        Expects an array of 2d arrays (1 channel images)
        Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

def prepare_hog_data(X):
    """
        Performs a HOG transform on the given design matrix X.
        based off this tutorial: https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
    """

    # transform data
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(8, 8),
        cells_per_block=(2,2),
        orientations=9,
        block_norm='L2-Hys'
    )
    scalify = StandardScaler()

    X_gray = grayify.fit_transform(X)
    X_hog = hogify.fit_transform(X_gray)
    X_prepared = scalify.fit_transform(X_hog)

    return X_prepared

def plot_hog_image(image, filename_original, filename_hog):
    """
        Plots a HOG image against its original.
    """

    hog_, hog_image = hog(
        image, 
        pixels_per_cell=(12, 12),
        cells_per_block=(2,2),
        orientations=8,
        visualize=True,
        block_norm='L2-Hys')

    plot_transformed_image(image, hog_image, "hog", filename_original, filename_hog)

def plot_transformed_image(original, transformed, transformed_title, filename_original, filename_hog):
    """
        Plots a transformed image next to its original image.
    """

    # fig, ax = plt.subplots(1,2)
    # fig.set_size_inches(8,6)
    # remove ticks and their labels
    # [a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    #     for a in ax]
    
    plt.clf()
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.imshow(original, cmap='gray')
    # plt.set_title('original')
    plt.savefig(filename_original)

    plt.clf()
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.imshow(transformed, cmap='gray')
    # plt.savefig("transformed_{0}".format(filename))
    plt.savefig(filename_hog)

def prepare_texture_data(X):
    # https://gogul.dev/software/image-classification-python
    def hu_moments(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature
    
    def fd_haralick(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        return haralick

    def color_histogram(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    features = []
    for file_path in X:
        image = cv2.imread(file_path)

        moments = hu_moments(image)
        haralick = fd_haralick(image)
        # histogram = color_histogram(image)
        features.append(np.hstack([moments, haralick]))
    
    return features

def prepare_vgg_features(X):
    # http://parneetk.github.io/blog/CNN-TransferLearning1/

    base_model = VGG19(weights="imagenet")
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    features = []
    for file_path in X:
        img = image.load_img(file_path, target_size=(224,224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        fc2_features = model.predict(img)
        features.append(fc2_features.flatten()) #reshape((fc2_features.shape[0], 4096)))

    return features


        


