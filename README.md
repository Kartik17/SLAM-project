# Identification  and  Removal  of  Dynamic  Objects  in  Visual  SLAM

## Introduction
We  present  a  module  that  detects and tracks the dynamic objects in a scene using Deep Learning and  Kalman  Filter  respectively.  The  module  is  added  to  the existing SLAM system - ORBSLAM2 and results in improvingtheir robustness and accuracy in highly dynamic environments.We combine the instance segmentation and Object Tracking to generate masks for the dynamic objects in the scene, so as to not pass  ORB  features  of  the  masked  area  to  the  SLAM  pipeline.To  determine  which  objects  are  actually  moving,  we  segments potential  classes  of  dynamic  objects  such  as  person,  car,  chair etc. and then track the Bounding box in each each frame using Multi  Object  tracking  -  SORT.  Using  the  Depth  image  and Camera  Pose  generated  from  the  Slam  system  we  determine the(X, Y, Z)world of  a  object  and  pass  it  to  a  Kalman  Filter to  track  it  state.  Finally  based  on  the  velocity  of  the  object we  determine  whether  the  object  is  dynamic  or  not.  We  have evaluated our method on sequences of TUM RGBD and KITTI dataset using ORB-SLAM 2 [1]. Both the datasets are publicly available. Finally the results show that our approach improves the  accuracy  and  robustness  of  ORB-SLAM  2,  especially  inhighly  dynamic  environments.

## Install ORBSLAM-2
Clone the repository:

Replace the folders 'src' 'include' 'Examples' 'lib' that are inside the ORB_SLAM2 with the folders that's in the main repository.

We provide a script build.sh to build the Thirdparty libraries and ORB-SLAM2. Please make sure you have installed all required dependencies.
Execute:
```
cd ORB_SLAM2
chmod +x build.sh
./build.sh
```

(In case of issue please refer to the README.md of ORB_SLAM2 main repository)


## Install Environment (for Detectron2 dependencies)
conda env create -f environment.yml

## Dataset preparation
TUM Dataset
Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.
And arrange it as follows
```
seq/
  -rgbd_dataset_freiburg3_walking_rpy/
    -depth/
    -rgb/
    -accelerometer.txt
    -depth.txt
    -groundtruth.txt
    -rgb.txt
```

KITTI Dataset

Download the dataset (grayscale images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php
```
Nr.     Sequence name     Start   End
---------------------------------------
00: 2011_10_03_drive_0027 000000 004540
01: 2011_10_03_drive_0042 000000 001100
02: 2011_10_03_drive_0034 000000 004660
03: 2011_09_26_drive_0067 000000 000800
04: 2011_09_30_drive_0016 000000 000270
05: 2011_09_30_drive_0018 000000 002760
06: 2011_09_30_drive_0020 000000 001100
07: 2011_09_30_drive_0027 000000 001100
08: 2011_09_30_drive_0028 001100 005170
09: 2011_09_30_drive_0033 000000 001590
10: 2011_09_30_drive_0034 000000 001200
```
Arrange the data as:
```
seq/
  -2011_09_30_drive_0027_sync/
    -image_02/
    -oxts/
    -proj_depth/
    -velodyne_points/
    -07.txt
    -timestamps.txt
```

## How to Run
### Generate Mask
To generate mask for dataset - ('tum', 'kitti'). In the config.yaml file change DATASET = 'kitti' or DATASET = 'tum'
```
python dynamic_detector_main.py
```

The will be generated in ./masks folder.


### TUM Dataset
1. Associate RGB images and depth images using the python script associate.py. We already provide associations for some of the sequences in Examples/RGB-D/associations/. You can generate your own associations file executing:
```
python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
```
2. Now we need to refine the masks as per the dataset and send them in proper folder by executing:
```
python associate_masks_tum.py PATH_TO_SEQUENCE/rgb/ PATH_TO_MASKS_FOLDER/
```
3.Execute the following command. Change TUMX.yaml to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change PATH_TO_SEQUENCE_FOLDERto the uncompressed sequence folder. Change ASSOCIATIONS_FILE to the path to the corresponding associations file.
```
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
```

### KITTI Dataset
1. Now we need to refine the masks as per the dataset and send them in proper folder by executing:
```
python associate_masks_kitti.py PATH_TO_SEQUENCE/image_0/ PATH_TO_MASKS_FOLDER/
``` 
2. Execute the following command. Change KITTIX.yamlto KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml for sequence 0 to 2, 3, and 4 to 12 respectively. Change PATH_TO_DATASET_FOLDER to the uncompressed dataset folder. Change SEQUENCE_NUMBER to 00, 01, 02,.., 11.
```
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
```



## Evaluation
### TUM Dataset
1. Once the sequence finshes running it saves "KeyFrameTrajectory.txt" file and we have the groundtruth of the keyframe in PATH_TO_SEQUENCE/groundtruth.txt. Evaluate ate (absolute trajectory error) the trajectory generated by our algorithm by executing:
```
python evaluate_ate_tum.py PATH_TO_SEQUENCE/groundtruth.txt PATH_TO_ORB_OUTPUT/KeyFrameTrajectory.txt
```
2.Evaluate rpe (relative pose error) the trajectory generated by our algorithm by executing:
```
python evaluate_rpe_tum.py PATH_TO_SEQUENCE/groundtruth.txt PATH_TO_ORB_OUTPUT/KeyFrameTrajectory.txt
```
### KITTI Dataset
1. Once the sequence finshes running it saves "CameraTrajectory.txt" file and we have the groundtruth of the keyframe in PATH_TO_SEQUENCE/dataset/poses/XX.txt.  In KITTI dataset we need to convert both groundtruth and output text file in quaternions before evaluation.By executing the below it generates a quaternion file in the same folder.
For ground truth:
```
python kitti_convert_to_quaternions.py PATH_TO_SEQUENCE/dataset/sequences/SEQUENCE_NUMBER/times.txt PATH_TO_SEQUENCE/dataset/poses/XX.txt
```
For output file
```
python kitti_convert_to_quaternions.py PATH_TO_SEQUENCE/dataset/sequences/SEQUENCE_NUMBER/times.txt PATH_TO_OUTPUT_FILE/CameraTrajectory.txt
```
2.Evaluate ate (absolute trajectory error) the trajectory generated by our algorithm by executing:
```
python evaluate_ate_tum.py PATH_TO_SEQUENCE/groundtruth_quaternions.txt PATH_TO_ORB_OUTPUT/KeyFrameTrajectory.txt
```
3.Evaluate rpe (relative pose error) the trajectory generated by our algorithm by executing:
```
python evaluate_rpe_tum.py PATH_TO_SEQUENCE/groundtruth.txt PATH_TO_ORB_OUTPUT/KeyFrameTrajectory.txt
```


## Results

#### RMSE Absolute Trajectory Error (ATE) on TUM-RGBD Dataset
| Sequences      | ORB-SLAM2 | ORB-SLAM2 Masked |
|----------------|---------------|-------------|
| walking_static | 0.4030m       | 0.0081m     |
| walking_xyz    | 0.1780m       | **0.0247m** |

## Videos


