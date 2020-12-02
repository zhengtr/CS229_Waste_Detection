import cv2
import sys
import argparse
import json
from math import inf


categories = ['Bottle', 'Bottle cap', 'Can', 'Cup', 'Other', 'Plastic bag + wrapper', 'Straw']


def show_and_save(title, image, path):
    cv2.imwrite(path, image)

def load_annotations(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def drawBoundingBox(imgcv, boxes, labels):
    for i in range(len(boxes)):
        print(i)
        box = [int(a) for a in boxes[i]]
        l, t = box[0], box[1]
        r, b = box[0] + box[2], box[1] + box[3]
        label = categories[labels[i]]
        cv2.rectangle(imgcv, (l, t), (r, b), (0, 255, 0), 8)
        cv2.putText(imgcv, label, (l, t-10), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

    return imgcv


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "img_file",
        help="path to image file"
    )
    args = parser.parse_args()

    in_path = 'data/images/' + args.img_file
    out_path = 'out.jpg'

    image = cv2.imread(in_path, cv2.IMREAD_COLOR)

    data = load_annotations('annotations_0_test.json')
    images = sorted(data['images'], key=lambda img: img['id'])
    annotations = sorted(data['annotations'], key=lambda img: img['image_id'])

    id = inf
    for img in images:
        if img['file_name'] == args.img_file:
            id = img['id']
            break

    boxes = []
    labels = []

    for a in annotations:
        if a['image_id'] == id:
            boxes.append(a['bbox'])
            labels.append(a['category_id'])

    img_out = drawBoundingBox(image, boxes, labels)

    show_and_save(args.img_file, img_out, out_path)


if __name__ == "__main__":
    main(sys.argv)