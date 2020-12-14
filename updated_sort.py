import os
import numpy as np
import time
from filterpy.kalman import KalmanFilter
from kalman_utils import *
from kalman3d import Kalman3DTracker
np.random.seed(0)

class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    # State -(x,y,s,r,ud,vd,sd) 
    self.kf = KalmanFilter(dim_x=8, dim_z=4)
   
    self.dt = 0.1
    self.kf.F = np.array([[1,0,0,0,self.dt,0,0,0],
                          [0,1,0,0,0,self.dt,0,0],
                          [0,0,1,0,0,0,self.dt,0],
                          [0,0,0,1,0,0,0,self.dt],
                          [0,0,0,0,1,0,0,0],
                          [0,0,0,0,0,1,0,0],
                          [0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,1]])
    
    self.kf.H = np.array([[1,0,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0,0],
                          [0,0,1,0,0,0,0,0],
                          [0,0,0,1,0,0,0,0]])

    # R - (5,5), P,Q,F- (9,9), H-(5,9)
    
    # Measurement Noise
    self.kf.R = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [0,0,5,0],
                          [0,0,0,5]])
    #Process Noise
    self.kf.P = np.array([[10,0,0,0,0,0,0,0],
                          [0,10,0,0,0,0,0,0],
                          [0,0,10,0,0,0,0,0],
                          [0,0,0,10,0,0,0,0],
                          [0,0,0,0,5000,0,0,0],
                          [0,0,0,0,0,5000,0,0],
                          [0,0,0,0,0,0,5000,0],
                          [0,0,0,0,0,0,0,5000]])

    self.kf.Q = np.array([[1,0,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0,0],
                          [0,0,0.01,0,0,0,0,0],
                          [0,0,0,0.01,0,0,0,0],
                          [0,0,0,0,0.01,0,0,0],
                          [0,0,0,0,0,0.01,0,0],
                          [0,0,0,0,0,0,0.01,0],
                          [0,0,0,0,0,0,0,0.01]
                          ])

  
    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.kf.x[4:] = 0.0
    
    self.kf3d = Kalman3DTracker(bbox)

    self.time_since_update = 0
    self.id = KalmanBoxTracker.count

    self.cls_id = bbox[9]
    KalmanBoxTracker.count += 1
    self.history = []

    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    
    self.masks = [bbox[8]]
    self.past_x = []
    self.past_len = 2
    self.mask_len = 2

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    
    self.kf.update(convert_bbox_to_z(bbox))
    self.kf3d.update(bbox)

    self.cls_id = bbox[9]

    if(len(self.masks) < self.mask_len):
      self.masks.append(bbox[8])
    else:
      temp = [self.masks[1],bbox[8]]
      self.masks = temp

  def maskIOU(self):


    if(len(self.masks) > 1):
      intersect_seg = np.sum(np.bitwise_and(self.masks[-1],self.masks[-2]))
      union_seg = np.sum(np.bitwise_or(self.masks[-1] , self.masks[-2]))
      iou_seg = intersect_seg/ union_seg
    else:
      iou_seg = 0.0

    return iou_seg

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()
    self.age += 1
    
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    
    self.history.append(convert_x_to_bbox(self.kf.x))

    return np.concatenate((self.history[-1][0], self.kf3d.predict()))

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf3d.get_state(), getState(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8),dtype=int)
  
  iou_matrix = iou_batch(detections[:,:4], trackers[:,:4])
  
  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 8)), cls_ids = None, delta = None, masks = None):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,X,Y,Z,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    
    
    self.frame_count += 1
    N = len(self.trackers)
    trks = np.zeros((N, 8))
    
    to_del = []
    

    # get predicted locations from existing trackers.
    for t, trk in enumerate(trks):
      if delta != None:
        self.trackers[t].dt = delta
      pos = self.trackers[t].predict()
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4],pos[5],pos[6], 0]
      
      if np.any(np.isnan(pos)):
        to_del.append(t)
    
    #trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

    for t in to_del:
      self.trackers.pop(t)
    
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)

    i = len(self.trackers)
    ret = []
    for trk in reversed(self.trackers):
        # State -(u,v,s,r,z,ud,vd,sd,zd)
        state_3d, bbox_state  = trk.get_state()
        
        if (trk.time_since_update < self.max_age) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          output = np.concatenate((np.concatenate((bbox_state[:4],state_3d)),[trk.maskIOU(), trk.cls_id, trk.id+1, trk.masks[-1]])).reshape(1,-1)
          ret.append(output) 
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,13))





