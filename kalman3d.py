from kalman_utils import *
from filterpy.kalman import KalmanFilter

class Kalman3DTracker(object):
  def __init__(self,bbox):
    # State -(X,Y,Z,Xd,Yd,Zd) 
    self.dt = 0.1
    self.kfbox_3d = KalmanFilter(dim_x=6, dim_z=3)
    self.kfbox_3d.F = np.array([[1,0,0,self.dt,0,0],
                                [0,1,0,0,self.dt,0],
                                [0,0,1,0,0,self.dt],
                                [0,0,0,1,0,0],
                                [0,0,0,0,1,0],
                                [0,0,0,0,0,1]])
    
    self.kfbox_3d.H = np.array([[1,0,0,0,0,0],
                                [0,1,0,0,0,0],
                                [0,0,1,0,0,0]])
    # Measurement Noise
    self.kfbox_3d.R = np.array([[0.8,0,0],
                                [0,0.8,0],
                                [0,0,0.8]])
    #Process Noise
    self.kfbox_3d.P = np.array([[10,0,0,0,0,0],
                                [0,10,0,0,0,0],
                                [0,0,10,0,0,0],
                                [0,0,0,10000,0,0],
                                [0,0,0,0,10000,0],
                                [0,0,0,0,0,10000]])

    self.kfbox_3d.Q = np.array([[0.01,  0.0,  0.0,  0.0,  0.0,  0.0],
                                [0.0,  0.01,  0.0,  0.0, 0.0,   0.0],
                                [0.0,  0.0,  0.01,  0.0,   0.0, 0.0],
                                [0.0,  0.0,  0.0,  0.01,  0.0,  0.0],
                                [0.0,  0.0,  0.0,  0.0,  0.01,  0.0],
                                [0.0,  0.0,  0.0,  0.0,  0.0, 0.01]])    
    
    self.kfbox_3d.x[:3] = convert_bbox_to_xyz(bbox)
    self.kfbox_3d.x[3:] = 0.0
    self.history_3d = []

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.kfbox_3d.update(convert_bbox_to_xyz(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kfbox_3d.predict()
    self.history_3d.append(self.kfbox_3d.x.reshape(-1))

    return self.history_3d[-1]

  def get_state(self):
    """
    Returns the current 3D box estimate.
    """
    return self.kfbox_3d.x.reshape(-1)

