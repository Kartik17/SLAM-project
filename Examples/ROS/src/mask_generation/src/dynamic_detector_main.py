#! /usr/bin/python
import roslib
import os
import time
import cv2
from PIL import Image
from associate import read_file_list
from detectron2_detection import Detectron2
from util import draw_bboxes
from updated_sort import *
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml
from dataset_utils import *

# ros lib
roslib.load_manifest('mask_generation')
import sys
import rospy
import cv2
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image as Image_ros
from cv_bridge import CvBridge, CvBridgeError

class Detector(object):
    def __init__(self):
        self.use_cuda = True
        self.display  = True
        self.config = yaml.load(open('config.yaml', 'r'))
        self.dataset = self.config['DATASET']['NAME']
        self.detectron2 = Detectron2()
        # ros
        self.image_pub = rospy.Publisher("/segmentation_mask",Image_ros,queue_size=10,latch=True)
        self.pose_sub = rospy.Subscriber("/camera_pose", Image_ros,self.pose_callback)
        # self.pose_true = False
        self.pose = np.eye(4)
        self.bridge = CvBridge()

        if(self.dataset == 'kitti'):

            self.sequence_list = os.listdir(self.config['DATASET']['KITTI']['DATA_PATH'])
            self.velo2cam = np.array(self.config['DATASET']['KITTI']['TRANSFORMS']['Velo2cam'])
            
            self.kitti_timestamps = open(self.config['DATASET']['KITTI']['TIMESTAMPS']).readlines()
            self.kitti_odom = open(self.config['DATASET']['KITTI']['ODOM_PATH']).readlines()

            self.matches = sorted(self.sequence_list)
            self.max_vel = self.config['DATASET']['KITTI']['MAX_VEL']


            self.fx = self.config['DATASET']['KITTI']['CAMERA']['focal_length_x']
            self.fy = self.config['DATASET']['KITTI']['CAMERA']['focal_length_y']
            self.cx = self.config['DATASET']['KITTI']['CAMERA']['optical_center_x']
            self.cy = self.config['DATASET']['KITTI']['CAMERA']['optical_center_y']

        elif(self.dataset == 'tum'):
            #self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)
            self.first_list  = read_file_list(self.config['DATASET']['TUM']['RGB_PATH'] + '.txt')
            self.second_list = read_file_list(self.config['DATASET']['TUM']['DEPTH_PATH'] + '.txt')
            self.third_list  = read_file_list(self.config['DATASET']['TUM']['ODOM_PATH'] + '.txt')
            
            self.matches = associate(self.first_list, self.second_list, self.third_list, 0.0,0.02)

            self.max_vel = self.config['DATASET']['TUM']['MAX_VEL']
            self.fx = self.config['DATASET']['TUM']['CAMERA']['focal_length_x']
            self.fy = self.config['DATASET']['TUM']['CAMERA']['focal_length_y']
            self.cx = self.config['DATASET']['TUM']['CAMERA']['optical_center_x']
            self.cy = self.config['DATASET']['TUM']['CAMERA']['optical_center_y']
        
        self.class_names = self.config['CLASSES']['ALL']
        self.rigid = self.config['CLASSES']['RIGID']
        self.not_rigid = self.config['CLASSES']['NON_RIGID']
    
    def pose_callback(self,pose_): 
        self.pose = np.asarray(self.bridge.cv2_to_imgmsg(pose_, "8UC1"))

    def findDepth(self,outputs, cls_ids, masks, T, depth_im):
        upd_dets = []
        off_y = 0; off_x = 0;
        
        for idx,box in enumerate(outputs):
            
            if(np.sum(masks[idx]) > self.config['MASK_POINTS']):
                obj_mask = depth_im[masks[idx]]
                Z = np.sum(obj_mask)/np.count_nonzero(obj_mask)

                mask_x, mask_y = np.where(masks[idx])
                
                mask_mean_x, mask_mean_y = np.ceil(np.median(mask_x)), np.ceil(np.median(mask_y)) 


                X = Z*(mask_mean_y - self.cx)/(self.fx)
                Y = Z*(mask_mean_x - self.cy)/(self.fy)

                r_cam = np.array([X,Y,Z,1.0]).T
                R_cam_to_world = T
                R_world = np.matmul(R_cam_to_world, r_cam).reshape(-1)

                if(Z < self.config['MAX_DEPTH']):
                    upd_dets.append([box[0],box[1],box[2],box[3],R_world[0],R_world[1],R_world[2],T, masks[idx], cls_ids[idx], box[4]])
                else:
                    continue

            else:
                continue

            
        outputs = np.array(upd_dets)
        
        return outputs
    

    def findStaticDyanmic(self, box3d_state, cls_ids, maskIOUs):
        label = []
        for i,cls_id in enumerate(cls_ids):
            vel = np.round(np.linalg.norm(box3d_state[i,3:]),2) 
            print("{}:{}".format(self.class_names[cls_id], vel))
            

            if(vel >= self.max_vel):
                if(self.class_names[cls_id] in self.rigid):
                    if(maskIOUs[i] > self.config['MAX_IOU']):
                        label.append(1)     
                    else:
                        label.append(0)
                else:
                    label.append(0)       
            else:
                label.append(1)
                
        return np.array(label)
                    

    # def detect(self):
    def detect_callback(self,rgb_im,depth_im): # all image files
        dt = 0.1
        frame_no = 0.0
        time_prev = 0.0 #self.kitti_timestamps[0].split('\n')[0].split(' ')[1]
        t_prev = np.array([0.,0.,0.])
        mot_tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.5)
        
        for idx,sequence_list in enumerate(self.matches):
        
            start = time.time()
            frame_no += 1.0
            
            # if(self.dataset == 'tum'):
            #     rgb_name , depth_name, odom_name = sequence_list
            #     rgb_name = str(rgb_name)

            #     if len(rgb_name) < 17:
            #         rgb_name += '0'*(17 - len(rgb_name))
            #     img_path =   os.path.join(  self.config['DATASET']['TUM']['RGB_PATH'], rgb_name + '.png')
            #     depth_path = os.path.join(self.config['DATASET']['TUM']['DEPTH_PATH'], str(depth_name) + '.png')
                
            #     oxt = self.third_list[odom_name]
            #     t = np.array(oxt[:3]).astype('float32')
            #     q = np.array(oxt[3:]).astype('float32')

            #     if (os.path.isfile(depth_path)):
            #         depth_im  = Image.open(depth_path)
            #         depth_im = np.asarray(depth_im)/self.config['DATASET']['TUM']['DEPTH_FACTOR']
                    
            #     else:
            #         time_prev = odom_name
            #         continue
            #     r = R.from_quat(q).as_dcm()
            #     t_prev = t
            #     time_prev = odom_name

            #     T = np.array([[r[0,0], r[0,1], r[0,2], t[0]],
            #                   [r[1,0], r[1,1], r[1,2], t[1]],
            #                   [r[2,0], r[2,1], r[2,2], t[2]],
            #                   [0.0,0.0,0.0,1.0]])

            # elif(self.dataset == 'kitti'):
            #     im_name = sequence_list
            #     img_path = os.path.join('./seq/2011_09_30_drive_0018_sync/image_02/data/', im_name)
            #     oxt_path = os.path.join('./seq/2011_09_26/2011_09_26_drive_0009/oxts/data/', im_name.split('.')[0] +  '.txt')
            #     depth_path = os.path.join('./seq/2011_09_30_drive_0018_sync/proj_depth/groundtruth/image_02/', im_name)
 
            #     r = np.array(self.kitti_odom[idx-1].split('\n')[0].split(' ')).reshape(3,4).astype('float32')
            #     T = np.vstack((r,[0.,0.,0.,1.]))


            #     if (os.path.isfile(depth_path)):
            #         depth_im = depth_read(depth_path)
            #     else:
            #         continue

            # if(time_prev != 0):
            #     dt = odom_name - time_prev


            depth_im = np.asarray(self.bridge.cv2_to_imgmsg(depth_im, "8UC1"))
            print("depth array :\n",depth_im)
            # im = np.asarray(Image.open(img_path))
            im = np.asarray(self.bridge.cv2_to_imgmsg(rgb_im, "rgb8"))
            T = self.pose
            
            bbox_xcycwh, cls_conf, cls_ids, dets, masks = self.detectron2.detect(im)
            dets = self.findDepth(dets, cls_ids, masks, T,  depth_im)


            if(len(dets) == 0):
                outputs = mot_tracker.update(delta = dt)
            else:
                outputs = mot_tracker.update(dets, delta = dt)

            if len(outputs) > 0:                
                bbox_xyxy   = outputs[:, :4]
                box3d_state = outputs[:,4:10]

                maskIOUs   = outputs[:,10]
                cls_ids    = outputs[:,11].astype('int')
                identities = outputs[:,12]
                masks_obj  = outputs[:,13]
                
                is_static = self.findStaticDyanmic(box3d_state, cls_ids, maskIOUs)
            
                im = draw_bboxes(im, bbox_xyxy, identities, box3d_state[:,3:], is_static)
                save_path = './masks/'
                
                mask = np.zeros_like(depth_im)
                for i in range(len(is_static)):
                    
                    if(is_static[i] == 0):
                        mask = mask + masks_obj[i]
                mask = mask.astype('bool').astype('int')*255
                if(self.config['SAVE_MASK']):
                    cv2.imwrite(save_path + rgb_name + '_mask.png', mask)
                if(self.config['PUBLISH_ROS_TOPIC']):
                    try:
                        img = mask
                        img = np.stack((img,) * 3,-1)
                        img = img.astype(np.uint8)
                        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        thresh = cv2.threshold(grayed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                        mask = thresh
                        self.image_pub.publish(self.bridge.cv2_to_imgmsg(mask, "8UC1"))
                    except CvBridgeError as e:
                        print(e)

            end = time.time()
            dt = np.round(end - start,2)
            print("time: {}: {}s, fps: {}".format(frame_no, end - start, 1 / (end - start)))

            self.display = True
            if self.display:
                cv2.imshow("test", im)
                cv2.waitKey(10)
            '''
            if self.save_path:
                self.output.write(im)
            '''

if __name__ == "__main__":
    
    det = Detector()
    rospy.init_node('detector', anonymous=True)
    rgb_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image_ros)
    depth_sub = message_filters.Subscriber("camera/depth_registered/image_raw", Image_ros)
    # pose_sub = message_filters.Subscriber("/camera_pose", Image)
    ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub], 10)
    ts.registerCallback(det.detect_callback)
    # det.detect()