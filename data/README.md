This directory is used to store data, trained models and results files downloaded from this [Google Drive folder](https://drive.google.com/open?id=1Af9igOStb6O9YHwjYHOwR0qW4uP3zLA6), which is organized as follows:
```
data
    |-- kitti
        |-- bboxes
        |-- cars
        |-- tracklets
    |-- shapenet
        |-- test
            |-- partial
            |-- complete
        |-- test_novel
            |-- partial
            |-- complete
        |-- test.list
        |-- test_novel.list
        |-- train.list
        |-- train.lmdb
        |-- valid.list
        |-- valid.lmdb
    |-- shapenet-car
        |-- train.list
        |-- train.lmdb
        |-- valid.list
        |-- valid.lmdb
    |-- trained_models
    |-- results
        |-- kitti
        |-- shapenet_test
        |-- shapenet_test_novel
```
`kitti` contains processed data from the `2011_09_26_drive_0009` LiDAR sequence in the [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) dataset. `cars` contains raw point clouds of labeled cars in the sequence. `bboxes` contains the corresponding bounding boxes. `tracklets` contains the IDs of point clouds that belong to the same car instance.

`shapenet` contains training and testing data created from synthetic models in [ShapeNetCore.v1](https://shapenet.org). There are four lists of model IDs: `train.list` contains 28974 models used for training; `valid.list` contains 800 models used for training; `test.list` contains 1200 models used for testing; and `test_novel.list` contains additional 1200 test models from shape categories not seen during training. Training and validation data are processed into `lmdb` format for more efficient data loading. Testing data are stored as point clouds in `pcd` format and put into two folders, where `partial` contains partial inputs and `complete` contains complete ground truths.

`shapenet_car` contains training and validation data for the car category only.

`trained_models` contains trained model weights that can be loaded with `tf.Saver`.

`results` contains outputs (completed point clouds) of the trained models on the KITTI sequence and the ShapeNet test set.