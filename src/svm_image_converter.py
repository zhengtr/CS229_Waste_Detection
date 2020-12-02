import sys
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

data_dir = "../data/"

def convert_folder(folder, width, height):
    print("Converting images in {0}".format(folder))

    images = os.listdir(data_dir + folder)
    for image in images:
        if ".npy" in image:
            continue

        pil_image = Image.open(data_dir + "/" + folder + "/" + image)
        pil_image = pil_image.resize((width, height))
        np_image = np.asarray(pil_image)

        new_file_name = data_dir + "/" + folder + "/" + image.lower().replace(".jpg", ".npy")
        np.save(new_file_name, np_image, allow_pickle=True)
        pil_image.close()

if __name__ == "__main__":
    for i in range(1, 16):
        convert_folder("batch_{0}".format(i), int(sys.argv[1]), int(sys.argv[2]))
    convert_folder("batch_aug", int(sys.argv[1]), int(sys.argv[2]))
    convert_folder("batch_crop_images", int(sys.argv[1]), int(sys.argv[2]))

