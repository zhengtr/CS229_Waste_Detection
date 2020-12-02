import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "log_file",
        help="path to log file"
    )

    args = parser.parse_args()

    f = open(args.log_file)

    lines = [line.rstrip("\n") for line in f.readlines()]

    numbers = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}

    map_iters = np.arange(1000, 13960, 270)
    map = []
    iou = []

    iters = []
    loss = []

    for line in lines:
        args = line.strip().split(' ')

        if args[0][-1:] == ':' and args[0][0] in numbers:
            iters.append(int(args[0][:-1]))
            loss.append(float(args[2]))

        # get map
        if args[0] == 'mean_average_precision':
            map.append(float(args[-1]))

        # get iou
        if args[0] == 'for' and args[-1] == '%':
            iou.append(float(args[-2]) / 100)

    fig, ax1 = plt.subplots()
    ax1.plot(map_iters, map, color='coral', label='mAP')
    ax1.plot(map_iters, iou, color='lightblue', label='IoU')
    ax1.set_xlabel('iters')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main(sys.argv)

