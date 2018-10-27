import matplotlib
matplotlib.use('agg')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse


plt.style.use('seaborn-whitegrid')
blue = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
green = (0.3333333333333333, 0.6588235294117647, 0.40784313725490196)
red = (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)


def plot(dst_path, csv_path, bins=30, norm_hist=True):
    df = pd.read_csv(csv_path)
    df = df[['label', 'normality']]

    inlier_normality = df[df.label == 'inlier']['normality']
    outlier_normality = df[df.label == 'outlier']['normality']

    sns.distplot(inlier_normality,
                 label='inlier',
                 kde=False,
                 color=blue,
                 norm_hist=norm_hist,
                 bins=np.linspace(0, float(df['normality'].max()), bins),
                 hist_kws={'rwidth': 1.0,
                           'alpha': 0.8,
                           'stacked': False})

    sns.distplot(outlier_normality,
                 label='outlier',
                 kde=False,
                 color=red,
                 norm_hist=norm_hist,
                 bins=np.linspace(0, float(df['normality'].max()), bins),
                 hist_kws={'rwidth': 1.0,
                           'alpha': 0.8,
                           'stacked': False})

    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center',
               borderaxespad=0., ncol=2, fontsize='x-large')
    plt.xlabel('normality', fontsize='x-large')
    plt.savefig(dst_path, dpi=300)
    plt.clf()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str)
    parser.add_argument('--non_norm_hist', '-nnr', dest='norm_hist', action='store_false')
    parser.set_defaults(norm_hist=True)

    args = parser.parse_args()

    dst_path = os.path.join(os.path.dirname(args.csv_path),
                            "normality_histogram.png")
    plot(dst_path, args.csv_path, norm_hist=args.norm_hist)


if __name__ == "__main__":
    main()
