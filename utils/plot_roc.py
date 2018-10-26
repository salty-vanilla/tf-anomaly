import os
import sys
import matplotlib
sys.path.append(os.getcwd())
matplotlib.use('agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import auc
import seaborn as sns


def compute_fpr_tpr(path,
                    metric,
                    interval=100):
    df = pd.read_csv(path)
    df = df[['label', metric]]

    fpr_list = []
    tpr_list = []

    max_ = max(df[metric])
    step = max_ / interval
    for thr in np.arange(0.0, max_ + 1., step):
        if metric == 'normality':
            tp = len(df[(df.label == 'outlier') & (df[metric] < thr)])
            fp = len(df[(df.label == 'inlier') & (df[metric] < thr)])
        else:
            tp = len(df[(df.label == 'outlier') & (df[metric] > thr)])
            fp = len(df[(df.label == 'inlier') & (df[metric] > thr)])

        fpr = fp / len(df[df.label == 'inlier'])
        tpr = tp / len(df[df.label == 'outlier'])

        fpr_list.append(fpr)
        tpr_list.append(tpr)
    return np.array(fpr_list), np.array(tpr_list)


def draw_roc_curves(dst_path,
                    fprs,
                    tprs,
                    aucs,
                    names,
                    linestyles,
                    display_auc):
    names = ['' for _ in range(len(fprs))] if names is None else names
    plt.grid()
    for (fpr, tpr, auc_, name, ls) in zip(fprs, tprs, aucs, names, linestyles):
        plt.xlim(0., 1.0)
        plt.ylim(0., 1.1)
        plt.xlabel("False Positive Rate", fontsize='xx-large')
        plt.ylabel("True Positive Rate", fontsize='xx-large')
        print('{} : {}'.format(name, auc_))
        if display_auc:
            name += "_{:.4f}".format(auc_)
        plt.plot(fpr, tpr, label=name, lw=3, linestyle=ls)

    leg = plt.legend(loc='lower right',
                     fontsize='large',
                     frameon=True)
    plt.tick_params(labelsize='x-large')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.0)
    plt.savefig(dst_path, dpi=300,  bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_paths', nargs='+', type=str)
    parser.add_argument('metric', type=str)
    parser.add_argument('--labels', nargs='+', type=str, default=None)
    parser.add_argument('--interval', type=int, default=1000)
    parser.add_argument('--dst_path', type=str, default='./roc.png')
    parser.add_argument('--display_auc', '-auc', default=False, action='store_true')
    parser.add_argument('--linestyles', '-ls', nargs='+', default=None)
    args = parser.parse_args()

    assert len(args.csv_paths) == len(args.labels)

    os.makedirs(os.path.dirname(args.dst_path), exist_ok=True)

    fprs = []
    tprs = []
    aucs = []

    if args.linestyles is None:
        args.linestyles = ['solid']*len(args.csv_paths)
    for csv_path in args.csv_paths:
        fpr, tpr = compute_fpr_tpr(csv_path,
                                   args.metric,
                                   args.interval)
        auc_ = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc_)

    draw_roc_curves(args.dst_path,
                    fprs, tprs, aucs,
                    args.labels, args.linestyles,
                    args.display_auc)


if __name__ == '__main__':
    main()
