import matplotlib
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse


plt.style.use('seaborn-whitegrid')
blue = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
green = (0.3333333333333333, 0.6588235294117647, 0.40784313725490196)
red = (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)


def plot(dst_path,
         csv_path,
         metric):
    df = pd.read_csv(csv_path)
    df = df[['label', metric]]

    inliers = df[df.label == 'inlier'][metric]
    outliers = df[df.label == 'outlier'][metric]

    x = range(len(inliers))
    plt.scatter(x, inliers,
                label='inlier',
                color=blue,
                s=1)

    x = range(len(inliers), len(inliers)+len(outliers))
    plt.scatter(x,
                outliers,
                label='outlier',
                color=red,
                s=1)
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center',
               borderaxespad=0., ncol=2, fontsize='x-large')
    plt.xlabel('index', fontsize='x-large')
    plt.ylabel(metric, fontsize='x-large')
    plt.savefig(dst_path, dpi=300)
    plt.clf()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str)
    parser.add_argument('metric', type=str)
    args = parser.parse_args()

    dst_path = os.path.join(os.path.dirname(args.csv_path),
                            "%s_scatter.png" % args.metric)
    plot(dst_path, args.csv_path, args.metric)


if __name__ == "__main__":
    main()
