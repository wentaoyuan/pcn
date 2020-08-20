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
import numpy as np
import os
import parse_tracklet_xml as xmlParser
import pykitti
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from open3d import *


def load_tracklets_for_frames(n_frames, xml_path):
    """
    Loads dataset labels also referred to as tracklets, saving them individually for each frame.

    Parameters
    ----------
    n_frames    : Number of frames in the dataset.
    xml_path    : Path to the tracklets XML.

    Returns
    -------
    Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
    contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
    types as strings.
    """
    tracklets = xmlParser.parseXML(xml_path)

    frame_tracklets = {}
    frame_tracklets_types = {}
    frame_tracklets_ids = {}
    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklets_types[i] = []
        frame_tracklets_ids[i] = []

    # loop over tracklets
    for i, tracklet in enumerate(tracklets):
        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])
        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # determine if object is in the image; otherwise continue
            if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            frame_tracklets[absoluteFrameNumber].append(cornerPosInVelo)
            frame_tracklets_types[absoluteFrameNumber].append(tracklet.objectType)
            frame_tracklets_ids[absoluteFrameNumber].append(i)

    return frame_tracklets, frame_tracklets_types, frame_tracklets_ids


def plot_bbox(ax, bbox):
    # plot vertices
    ax.scatter(bbox[:, 0], bbox[:, 1], bbox[:, 2], c='k')

    # list of sides' polygons of figure
    verts = [[bbox[0], bbox[1], bbox[2], bbox[3]],
             [bbox[4], bbox[5], bbox[6], bbox[7]],
             [bbox[0], bbox[1], bbox[5], bbox[4]],
             [bbox[2], bbox[3], bbox[7], bbox[6]],
             [bbox[1], bbox[2], bbox[6], bbox[5]],
             [bbox[4], bbox[7], bbox[3], bbox[0]],
             [bbox[2], bbox[3], bbox[7], bbox[6]]]

    # plot sides
    bbox = Poly3DCollection(verts, linewidths=1, edgecolors='r', alpha=.1)
    bbox.set_facecolor('cyan')
    ax.add_collection3d(bbox)


def within_bbox(point, bbox):
    """Determine whether the given point is inside the given bounding box."""
    x = bbox[3, :] - bbox[0, :]
    y = bbox[0, :] - bbox[1, :]
    z = bbox[4, :] - bbox[0, :]
    return ((np.dot(x, bbox[0, :]) <= np.dot(x, point)) and (np.dot(x, point) <= np.dot(x, bbox[3, :])) and
            (np.dot(y, bbox[1, :]) <= np.dot(y, point)) and (np.dot(y, point) <= np.dot(y, bbox[0, :])) and
            (np.dot(z, bbox[0, :]) <= np.dot(z, point)) and (np.dot(z, point) <= np.dot(z, bbox[4, :])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('date')
    parser.add_argument('drive')
    parser.add_argument('output_dir')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    dataset = pykitti.raw(args.data_dir, args.date, args.drive)
    frames = list(dataset.velo)
    print('Number of frames', len(frames))

    tracklet_path = os.path.join(args.data_dir, args.date, '%s_drive_%s_sync' % (args.date, args.drive),
                                 'tracklet_labels.xml')
    tracklet_rects, tracklet_types, tracklet_ids = load_tracklets_for_frames(len(frames), tracklet_path)

    car_dir = os.path.join(args.output_dir, 'cars')
    bbox_dir = os.path.join(args.output_dir, 'bboxes')
    tracklet_dir = os.path.join(args.output_dir, 'tracklets')
    os.makedirs(car_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)
    os.makedirs(tracklet_dir, exist_ok=True)

    num_points = []
    for i, points in enumerate(frames):
        points = np.array(points)[:, :3]
        for j in range(len(tracklet_types[i])):
            if tracklet_types[i][j] == 'Car':
                car_id = 'frame_%d_car_%d' % (i, j)

                bbox = np.array(tracklet_rects[i][j]).T
                car_points = np.array([point for point in points if within_bbox(point, bbox)])

                if args.plot:
                    ax = plt.subplot(projection='3d')
                    plot_bbox(ax, bbox)
                    ax.scatter(car_points[:, 0], car_points[:, 1], car_points[:, 2], c='r')
                    plt.axis('equal')
                    plt.axis('off')
                    plt.show()

                if car_points.shape[0] > 0:
                    num_points.append(car_points.shape[0])

                    pcd = PointCloud()
                    pcd.points = Vector3dVector(car_points)
                    write_point_cloud(os.path.join(car_dir, '%s.pcd' % car_id), pcd)

                    np.savetxt(os.path.join(bbox_dir, '%s.txt' % car_id), bbox)

                    with open(os.path.join(tracklet_dir, 'tracklet_%d.txt' % tracklet_ids[i][j]), 'a+') as file:
                        file.write(car_id + '\n')

    print('Average number of points', np.mean(num_points), 'std', np.std(num_points))
