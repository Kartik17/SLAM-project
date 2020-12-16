# Identification  and  Removal  of  Dynamic  Objects  in  Visual  SLAM

## Introduction
We  present  a  module  that  detects and tracks the dynamic objects in a scene using Deep Learning and  Kalman  Filter  respectively.  The  module  is  added  to  the existing SLAM system - ORBSLAM2 and results in improvingtheir robustness and accuracy in highly dynamic environments.We combine the instance segmentation and Object Tracking to generate masks for the dynamic objects in the scene, so as to not pass  ORB  features  of  the  masked  area  to  the  SLAM  pipeline.To  determine  which  objects  are  actually  moving,  we  segments potential  classes  of  dynamic  objects  such  as  person,  car,  chair etc. and then track the Bounding box in each each frame using Multi  Object  tracking  -  SORT.  Using  the  Depth  image  and Camera  Pose  generated  from  the  Slam  system  we  determine the(X, Y, Z)world of  a  object  and  pass  it  to  a  Kalman  Filter to  track  it  state.  Finally  based  on  the  velocity  of  the  object we  determine  whether  the  object  is  dynamic  or  not.  We  have evaluated our method on sequences of TUM RGBD and KITTI dataset using ORB-SLAM 2 [1]. Both the datasets are publicly available. Finally the results show that our approach improves the  accuracy  and  robustness  of  ORB-SLAM  2,  especially  inhighly  dynamic  environments.

## Install ORBSLAM-2
To Install and Run ORB SLAM please refer to the README.md of ORB SLAM

## Install Environment
conda env create -f environment.yml

## Dataset preparation
```
Seq/
  -rgbd_dataset_freiburg3_walking_rpy/
    -depth/
    -rgb/
    -accelerometer.txt
    -depth.txt
    -groundtruth.txt
    -rgb.txt
```
