# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import argparse
import importlib
import models
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from open3d import *


def plot_pcd(ax, pcd):
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir='y', c=pcd[:, 0], s=0.5, cmap='Reds', vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='demo_data/car.pcd')
    parser.add_argument('--model_type', default='pcn_cd')
    parser.add_argument('--checkpoint', default='data/trained_models/pcn_cd')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    args = parser.parse_args()

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

    partial = read_point_cloud(args.input_path)
    partial = np.array(partial.points)
    complete = sess.run(model.outputs, feed_dict={inputs: [partial], npts: [partial.shape[0]]})[0]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121, projection='3d')
    plot_pcd(ax, partial)
    ax.set_title('Input')
    ax = fig.add_subplot(122, projection='3d')
    plot_pcd(ax, complete)
    ax.set_title('Output')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    plt.show()
