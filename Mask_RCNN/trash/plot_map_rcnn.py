import argparse
import sys
import matplotlib.pyplot as plt


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "log_file",
        help="path to log file"
    )

    args = parser.parse_args()

    f = open(args.log_file)

    lines = [line.rstrip("\n") for line in f.readlines() if line.startswith('mAP')]
    map = [float(line.split(' ')[-1]) for line in lines]

    fig, ax1 = plt.subplots()

    ax1.plot(list(range(1, 121)), map, color='coral', label='mAP')
    ax1.set_xlabel('epochs')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main(sys.argv)

