# Identification  and  Removal  of  Dynamic  Objects  in  Visual  SLAM

## Introduction
We  present  a  module  that  detectsand tracks the dynamic objects in a scene using Deep Learningand  Kalman  Filter  respectively.  The  module  is  added  to  theexisting SLAM system - ORBSLAM2 and results in improvingtheir robustness and accuracy in highly dynamic environments.We combine the instance segmentation and Object Tracking togenerate masks for the dynamic objects in the scene, so as to notpass  ORB  features  of  the  masked  area  to  the  SLAM  pipeline.To  determine  which  objects  are  actually  moving,  we  segmentspotential  classes  of  dynamic  objects  such  as  person,  car,  chairetc. and then track the Bounding box in each each frame usingMulti  Object  tracking  -  SORT.  Using  the  Depth  image  andCamera  Pose  generated  from  the  Slam  system  we  determinethe(X, Y, Z)worldof  a  object  and  pass  it  to  a  Kalman  Filterto  track  it  state.  Finally  based  on  the  velocity  of  the  objectwe  determine  whether  the  object  is  dynamic  or  not.  We  haveevaluated our method on sequences of TUM RGBD and KITTIdataset using ORB-SLAM 2 [1]. Both the datasets are publiclyavailable. Finally the results show that our approach improvesthe  accuracy  and  robustness  of  ORB-SLAM  2,  especially  inhighly  dynamic  environments.

## Install ORBSLAM-2
To Install and Run ORB SLAM please refer to the README.md of ORB SLAM

## Install Environment
conda env create -f environment.yml
