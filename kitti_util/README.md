This directory contains code to extract car point clouds from raw LiDAR sequences in KITTI.
#### Usage
1. Install [pykitti](https://github.com/utiasSTARS/pykitti).
2. Download a raw sequence from [KITTI](http://cvlibs.net/datasets/kitti/raw_data.php). Specifically, we need the `[synced+rectified data]` and `[tracklets]`.
3. Put the data into a directory organized as follows.
    ```
    data
        |--[date]
            |--[date]_drive_[drive_id]_sync
                |--tracklet_labels.xml
                |--...
    ```
4. Run `python3 process_kitti_raw.py [data directory] [date] [drive_id] [output directory]`.