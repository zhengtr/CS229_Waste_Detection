import matplotlib.pyplot as plt
import os
import pandas as pd

map_dir = 'mAP'
categories = ['Battery', 'Cigarette', 'Compost', 'Glass', 'Litter', 'Metal',
              'Non-recyclable plastic', 'Paper', 'Plastic']


def metrics_df_allfiles():
    fps = [os.path.join(map_dir, f) for f in os.listdir(map_dir) if f != '.DS_Store']
    ap_df = pd.DataFrame(columns=categories)
    pr_df = pd.DataFrame(columns=['precision', 'recall'])
    f1_df = pd.DataFrame(columns=['f1'])
    iou_df = pd.DataFrame(columns=['IoU'])
    for fp in fps:
        iter = int(fp.split()[1])
        print(iter)
        ap_classes, pr, f1, iou = get_metrics(fp)
        ap_df.loc[iter] = ap_classes
        pr_df.loc[iter] = pr
        f1_df.loc[iter] = f1
        iou_df.loc[iter] = iou
    for df in [ap_df, pr_df, f1_df, iou_df]:
        df.sort_index(axis=0, inplace=True)


    return ap_df, pr_df, f1_df, iou_df


def get_metrics(file):
    with open(file, 'r') as f:
        lines = [line.rstrip("\n") for line in f.readlines()]

    lines_ap = lines[8:17]
    ap_classes = [float(line.split(',')[2][6:10]) / 100 for line in lines_ap]

    line_pr = lines[18]
    pr = [float(l[-4:]) for l in line_pr.split(',')[1:3]]

    line_f1 = lines[18]
    f1 = float(line_f1.split(',')[3][11:16])

    line_iou = lines[19]
    iou = float(line_iou.split(',')[-1][14:20]) /100

    return ap_classes, pr, f1, iou


def plot_metric(df, metric):
    plt.figure(figsize=(10, 6))
    cols = df.columns
    for col in cols:
        plt.plot(df.index, df[col], 'o-', label=col)
        plt.legend(bbox_to_anchor=(0.8, 0.7, 0.3, 0.2))
        plt.title(f'{metric} over iterations', size=15)
        plt.ylabel(metric)
        plt.xlabel('iters')
    plt.show()

def plot_pr(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['recall'], df['precision'], '-')
    plt.title('Precision vs. Recall', size=15)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()


def main():
    ap_df, pr_df, f1_df, iou_df = metrics_df_allfiles()
    # ap_df.drop(columns=['Battery'], inplace=True)
    # print(ap_df)
    plot_metric(ap_df, 'mAP')
    plot_metric(f1_df, 'f1')
    plot_metric(iou_df, 'IoU')
    plot_pr(pr_df)

main()