import os
import sys
import json
import argparse


def load_annotations(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def get_class_file(data):
    categories = sorted(data['categories'], key=lambda x: x['id'])
    classes = [i['supercategory'] for i in categories]
    with open('trash.names', 'w') as f:
        for c in classes:
            f.write(c+'\n')


def get_paths_file(paths, subset):
    with open(f'{subset}.txt', 'w') as f:
        for path in paths:
            f.write(f'trash_data/images/{path}\n')


def get_yolo_txt(data):
    images = sorted(data['images'], key=lambda img: img['id'])
    annotations = sorted(data['annotations'], key=lambda img: img['image_id'])

    j = 0
    paths_for_file = []

    for img in images:
        fp = img['file_name']
        id = img['id']
        img_w = img['width']
        img_h = img['height']

        objs = []
        while annotations[j]['image_id'] == id:
            objs.append(annotations[j])
            j += 1
            if j == len(annotations):
                break

        if not objs:
            print(f'Image {fp} has no annotations!')
            continue

        paths_for_file.append(fp)

        txt_fp = os.path.join('data/images', f'{fp[:-3]}txt')

        with open(txt_fp, 'w') as f:
            for obj in objs:
                box = obj['bbox']
                box_center_x = (box[0] + box[2] / 2) / img_w
                box_center_y = (box[1] + box[3] / 2) / img_h
                box_w = box[2] / img_w
                box_h = box[3] / img_h
                box_id = obj['category_id']
                f.write(f'{box_id} {box_center_x} {box_center_y} {box_w} {box_h}\n')

    return paths_for_file

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subset",
        help="train/test/val"
    )
    args = parser.parse_args()

    file = f'annotations_0_{args.subset}.json'
    data = load_annotations(file)
    paths_for_files = get_yolo_txt(data)
    get_paths_file(paths_for_files, args.subset)
    get_class_file(data)



if __name__ == "__main__":
    main(sys.argv)
