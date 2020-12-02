import os
import numpy as np
import matplotlib.pyplot as plt


GT_PATH = 'trash_data/images'
PRED_PATH = 'predictions/images'
IMAGE_FILE_PATH = 'predictions/rela_path_test.txt'


def get_class_and_box(path):
    class_ids = []
    boxes = []
    with open(path, 'r') as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
        for line in lines:
            line = line.split(' ')
            class_ids.append(int(line[0]) + 1)
            [cx, cy, w, h] = [float(i) for i in line[1:]]
            # box [topleft x, y, bottomright x, y]
            box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            box = [0.0 if b < 0 else b for b in box]
            boxes.append(box)
    return class_ids, boxes

def compute_box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def compute_iou_instance(box1, box2, box1_area, box2_area):
    x1 = max(box1[0], box2[0])
    x2 = min(box1[2], box2[2])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = box1_area + box2_area - inter

    return inter / union

def load_gt_pred(imageinfo_path):
    gt = os.path.join(GT_PATH, imageinfo_path)
    class_gt, box_gt = get_class_and_box(gt)

    pred = os.path.join(PRED_PATH, imageinfo_path)
    class_pred, box_pred = get_class_and_box(pred)

    return class_gt, box_gt, class_pred, box_pred

def compute_match(class_gt, box_gt, class_pred, box_pred):
    area_gt = [compute_box_area(box) for box in box_gt]
    area_pred = [compute_box_area(box) for box in box_pred]

    match_gt = [-1] * len(class_gt)
    match_pred = [-1] * len(class_pred)

    for i in range(len(class_gt)):
        box = box_gt[i]
        cls = class_gt[i]
        area_box = area_gt[i]
        cur_max = 0
        for j in range(len(class_pred)):
            if match_pred[j] > -1:
                continue
            iou = compute_iou_instance(box, box_pred[j], area_box, area_pred[j])
            if iou >= 0.5 and iou > cur_max and class_pred[j] == cls:
                cur_max = iou
                match_gt[i] = j
                match_pred[j] = i

    return class_gt, match_gt, class_pred, match_pred


def add_to_cm(cm, class_gt, match_gt, class_pred):

    match_gt = [int(i) for i in match_gt]
    # add all matched pairs to cm
    matched = {k: match_gt[k] for k in range(len(match_gt)) if match_gt[k] >= 0}
    for m in matched.values():
        cls = class_pred[m]
        cm[cls][cls] += 1

    # get all unpaired matches
    class_pred = [class_pred[i] for i in range(len(class_pred)) if i not in matched.values()]
    class_gt = [class_gt[i] for i in range(len(class_gt)) if i not in matched.keys()

    while class_gt and class_pred:
        cm[class_gt.pop(0)][class_pred.pop(0)] += 1

    if class_gt:
        for i in class_gt:
            cm[i][0] += 1
    elif class_pred:
        for j in class_pred:
            cm[0][j] += 1

    return cm


def plot_cm(cm, labels):
    cm_norm = cm / cm.sum(axis=1, keepdims=1)
    print(cm_norm)

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


def main():
    cm = np.zeros((8, 8))
    with open(IMAGE_FILE_PATH, 'r') as f:
        lines = [line.rstrip("\n")[:-4] + '.txt' for line in f.readlines()]
        for line in lines:
            class_gt, box_gt, class_pred, box_pred = load_gt_pred(line)
            class_gt, match_gt, class_pred, match_pred = compute_match(class_gt, box_gt, class_pred, box_pred)
            add_to_cm(cm, class_gt, match_gt, class_pred)

    with open('trash_data/trash.names', 'r') as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
        labels = ['None'] + lines

    plot_cm(cm, labels)



if __name__ == "__main__":
    main()