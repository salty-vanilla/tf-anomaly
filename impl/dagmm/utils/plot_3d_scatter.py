import matplotlib
import argparse
import os
import numpy as np
import pandas as pd



blue = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
green = (0.3333333333333333, 0.6588235294117647, 0.40784313725490196)
red = (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)


def plot(dst_path, x, y, z, label, 
         x_label='z_0',
         y_label='z_1',
         z_label='mse'):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(np.ravel(x[label == 'inlier']),
                 np.ravel(y[label == 'inlier']),
                 np.ravel(z[label == 'inlier']),
                 color=blue)
    ax.scatter3D(np.ravel(x[label == 'outlier']),
                 np.ravel(y[label == 'outlier']),
                 np.ravel(z[label == 'outlier']),
                 color=red)
    ax.set_xlabel(x_label, fontsize='large')
    ax.set_ylabel(y_label, fontsize='large')
    ax.set_zlabel(z_label, fontsize='large')
    ax.set_zlim(0.)
    plt.savefig(dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str)
    parser.add_argument('-x', type=str, dest='x', default='z_0')
    parser.add_argument('-y', type=str, dest='y', default='z_1')
    parser.add_argument('-z', type=str, dest='z', default='mse')
    parser.add_argument('--visualize', action='store_true', default=False)

    args = parser.parse_args()
    if not args.visualize:
        matplotlib.use('agg')

    dst_path = os.path.join(os.path.dirname(args.csv_path),
                            "3d_scatter.png")

    df = pd.read_csv(args.csv_path)
    x = df[args.x]
    y = df[args.y]
    z = df[args.z]
    label = df['label']

    plot(dst_path, x, y, z, label)

    if args.visualize:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
