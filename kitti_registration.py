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
import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from open3d import *


def bbox2rt(bbox):
    center = (bbox.min(0) + bbox.max(0)) / 2
    bbox -= center
    yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0],
                         [0, 0, 1]])
    return rotation, center


def register(source, target, args):
    residual = TransformationEstimationPointToPoint()
    criteria = ICPConvergenceCriteria(max_iteration=args.max_iter)
    # Align the centroids of the point clouds
    source_points = np.array(source.points)
    target_points = np.array(target.points)
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    source = PointCloud()
    source.points = Vector3dVector(source_points - source_center)
    target = PointCloud()
    target.points = Vector3dVector(target_points - target_center)
    result = registration_icp(source, target, args.max_dist, np.eye(4), residual, criteria)
    source_trans = copy.deepcopy(source)
    source_trans.transform(result.transformation)
    R = result.transformation[:3, :3]
    t = result.transformation[:3, 3] + target_center - np.dot(source_center, R.T)
    return R, t, np.array(source_trans.points), np.array(target.points)


def rotation_error(R1, R2):
    cos = (np.trace(np.dot(R1, R2.T)) - 1) / 2
    cos = np.maximum(np.minimum(cos, 1), -1)
    return 180 * np.arccos(cos) / np.pi


def translation_error(t1, t2):
    return np.sqrt(np.sum((t1 - t2) ** 2))


def plot_pcd_pair(ax, pcd1, pcd2, title, cmaps, size, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1, 2)):
    ax.scatter(pcd1[:, 0], pcd1[:, 1], pcd1[:, 2], c=pcd1[:, 0], s=size, cmap=cmaps[0], vmin=-5, vmax=1.5)
    ax.scatter(pcd2[:, 0], pcd2[:, 1], pcd2[:, 2], c=pcd2[:, 0], s=size, cmap=cmaps[1], vmin=-5, vmax=1.5)
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)


def track(args):
    os.makedirs(os.path.join(args.results_dir, 'plots'), exist_ok=True)
    csv_file = open(os.path.join(args.results_dir, 'error.csv'), 'w')
    writer = csv.writer(csv_file)
    writer.writerow(['id', 'r_err_part', 't_err_part', 'r_err_comp', 't_err_comp'])

    n = 0
    total_r_err_part = 0
    total_t_err_part = 0
    total_r_err_comp = 0
    total_t_err_comp = 0
    for filename in os.listdir(args.tracklet_dir):
        tracklet_id = filename.split('.')[0]
        with open(os.path.join(args.tracklet_dir, filename)) as file:
            car_ids = file.read().splitlines()

        prev_frame = int(car_ids[0].split('_')[1])
        prev_R, prev_t = bbox2rt(np.loadtxt(os.path.join(args.bbox_dir, '%s.txt' % car_ids[0])))
        prev_partial = read_point_cloud(os.path.join(args.partial_dir, '%s.pcd' % car_ids[0]))
        prev_complete = read_point_cloud(os.path.join(args.complete_dir, '%s.pcd' % car_ids[0]))
        for i in range(args.interval, len(car_ids), args.interval):
            n += 1
            frame = int(car_ids[i].split('_')[1])
            instance_id = '%s_frame_%d_to_%d' % (tracklet_id, prev_frame, frame)

            R, t = bbox2rt(np.loadtxt(os.path.join(args.bbox_dir, '%s.txt' % car_ids[i])))
            R_gt = np.dot(R, prev_R.T)
            t_gt = t - np.dot(prev_t, R_gt.T)

            partial = read_point_cloud(os.path.join(args.partial_dir, '%s.pcd' % car_ids[i]))
            R_part, t_part, partial_trans, partial_target = register(prev_partial, partial, args)
            r_err_part = rotation_error(R_part, R_gt)
            t_err_part = translation_error(t_part, t_gt)
            total_r_err_part += r_err_part
            total_t_err_part += t_err_part

            complete = read_point_cloud(os.path.join(args.complete_dir, '%s.pcd' % car_ids[i]))
            R_comp, t_comp, complete_trans, complete_target = register(prev_complete, complete, args)
            r_err_comp = rotation_error(R_comp, R_gt)
            t_err_comp = translation_error(t_comp, t_gt)
            total_r_err_comp += r_err_comp
            total_t_err_comp += t_err_comp

            writer.writerow([instance_id, r_err_part, t_err_part, r_err_comp, t_err_comp])

            if n % args.plot_freq == 0:
                fig = plt.figure(figsize=(8, 4))
                ax = fig.add_subplot(121, projection='3d')
                plot_pcd_pair(ax, partial_trans, partial_target,
                              'Rotation error %.4f\nTranslation error %.4f' % (r_err_part, t_err_part),
                              ['Reds', 'Blues'], size=5)
                ax = fig.add_subplot(122, projection='3d')
                plot_pcd_pair(ax, complete_trans, complete_target,
                              'Rotation error %.4f\nTranslation error %.4f' % (r_err_comp, t_err_comp),
                              ['Reds', 'Blues'], size=0.5)
                plt.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0)
                fig.savefig(os.path.join(args.results_dir, 'plots', '%s.png' % instance_id))
                plt.close(fig)
    print('Using original pcd: average roration error %.4f  average translation error %.4f' %
          (total_r_err_part / n, total_t_err_part / n))
    print('Using completed pcd: average roration error %.4f  average translation error %.4f' %
          (total_r_err_comp / n, total_t_err_comp / n))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial_dir', default='data/kitti/cars')
    parser.add_argument('--complete_dir', default='data/results/kitti/pcn_emd/completions')
    parser.add_argument('--bbox_dir', default='data/kitti/bboxes')
    parser.add_argument('--tracklet_dir', default='data/kitti/tracklets')
    parser.add_argument('--results_dir', default='data/results/kitti_registration')
    parser.add_argument('--interval', type=int, default=1, help='number of frames to skip')
    parser.add_argument('--max_iter', type=int, default=100, help='max iteration for ICP')
    parser.add_argument('--max_dist', type=float, default=0.05, help='matching threshold for ICP')
    parser.add_argument('--plot_freq', type=int, default=100)
    args = parser.parse_args()

    track(args)
