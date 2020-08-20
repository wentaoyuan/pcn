'''
MIT License

Copyright (c) 2018 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import argparse
import importlib
import models
import os
import numpy as np
import tensorflow as tf

from io_util import read_pcd, save_pcd
from visu_util import show_pcd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_pcd(ax, pcd):
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir='y', c=pcd[:, 0], s=0.5, cmap='Reds', vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default='demo_data/car.pcd', help='path to the input point cloud')
    parser.add_argument('-o', '--output_path', type=str, help='path to the output directory')
    parser.add_argument('-m', '--model_type', type=str, default='pcn_cd', help='model type')
    parser.add_argument('-c', '--checkpoint', type=str, default='data/trained_models/pcn_cd', help='path to the checkpoint')
    parser.add_argument('-n', '--num_gt_points', type=int, default=16384, help='number of ground truth points')
    args = parser.parse_args()

    return args


def create_plots(partial, complete):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121, projection='3d')
    plot_pcd(ax, partial)
    ax.set_title('Input')
    ax = fig.add_subplot(122, projection='3d')
    plot_pcd(ax, complete)
    ax.set_title('Output')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)


def main():
    args = parse_args()

    inputs = tf.placeholder(tf.float32, (1, None, 3))
    gt = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
    npts = tf.placeholder(tf.int32, (1,))
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs, npts, gt, tf.constant(1.0))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    partial = read_pcd(args.input_path)
    complete = sess.run(model.outputs, feed_dict={inputs: [partial], npts: [partial.shape[0]]})[0]
    create_plots(partial, complete)

    if args.output_path is None:
        show_pcd(complete)
        plt.show()
    else:
        os.makedirs(args.output_path, exist_ok=True)
        filename = os.path.splitext(os.path.basename(args.input_path))[0]

        output_file = os.path.join(args.output_path, filename + '.pcd')
        save_pcd(output_file, complete)

        output_file = os.path.join(args.output_path, filename + '.png')
        plt.savefig(output_file)


if __name__ == '__main__':
    main()
