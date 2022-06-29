    #!/usr/bin/env python

from ast import While
import rospy
import sys
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# import tensorflow as tf
# from tensorflow import keras
# import imgaug.augmenters as iaa

import torch
import openpifpaf

from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped

import getpass

USERNAME = getpass.getuser()

def kp_to_msg(keypoint, time) -> PointStamped:
    msg = PointStamped()
    msg.header.stamp = time
    msg.point.x = keypoint[0]
    msg.point.y = keypoint[1]
    msg.point.z = keypoint[2]
    return msg

def get_pose_msg(translation, rotation, time) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = time
    msg.header.frame_id = "cam"
    rotation = rotation.flatten()
    #print(rotation)
    rotation_obj = R.from_euler('xyz', rotation)       # VERIFY THAT PNP RETURNS ROTATION AS XYZ AND NOT ZXY
    rotation_quat = rotation_obj.as_quat()
    #print(rotation_quat)
    msg.pose.position.x = translation[0]
    msg.pose.position.y = translation[1]
    msg.pose.position.z = translation[2]
    msg.pose.orientation.x = rotation_quat[0]
    msg.pose.orientation.y = rotation_quat[1]
    msg.pose.orientation.z = rotation_quat[2]
    msg.pose.orientation.w = rotation_quat[3]
    return msg

class PolePoseNode:
    def __init__(self, in_topic, out_topic) -> None:
        network = "/home/" + USERNAME + "/BT_Vision/outputs/" + "mobilenetv2-220622-202035-pole_detect.pkl.epoch900"
        # self.model = keras.models.load_model(network)
        self.camera = cv2.VideoCapture("/dev/v4l/by-id/usb-e-con_systems_See3CAM_CU55_1020CE08-video-index0") #Attention: check port nr on udoo!
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture("/dev/v4l/by-id/usb-e-con_systems_See3CAM_CU55_0C10CE08-video-index0")


        self.predictor = openpifpaf.Predictor(checkpoint=network)
        cv2.namedWindow('keypoints')

        if not self.camera.isOpened():
            print("[PolePoseNode] Cannot open camera!")

        self.kp_pub_0 = rospy.Publisher('PolePoseNode/KeyPoints/1', PointStamped, queue_size=1)
        self.kp_pub_1 = rospy.Publisher('PolePoseNode/KeyPoints/2', PointStamped, queue_size=1)
        self.kp_pub_2 = rospy.Publisher('PolePoseNode/KeyPoints/3', PointStamped, queue_size=1)
        self.kp_pub_3 = rospy.Publisher('PolePoseNode/KeyPoints/4', PointStamped, queue_size=1)
        self.kp_pub_4 = rospy.Publisher('PolePoseNode/KeyPoints/5', PointStamped, queue_size=1)
        self.kp_pub_5 = rospy.Publisher('PolePoseNode/KeyPoints/6', PointStamped, queue_size=1)
        self.kp_pub_6 = rospy.Publisher('PolePoseNode/KeyPoints/7', PointStamped, queue_size=1)

        self.pose_pub = rospy.Publisher('PolePoseNode/EstimatedPose', PoseStamped, queue_size=1)

        self.bridge = CvBridge()

        self.points_3d = np.array([
                                (0.0, 0.0,      1.185), # tip
                                (0.0, -0.0625,  1.0), # top_l
                                (0.0, 0.0625,   1.0), # top_r
                                (0.0, -0.0625,  0.505), # mid_l
                                (0.0, 0.0625,   0.505), # mid_r
                                (0.0, -0.0625,  0.0), # bottom_l
                                (0.0, 0.0625,   0.0) # bottom_r
                                ])
        self.camera_matrix = np.array([(350.39029238592053, 0.0, 315.61588345580935),
                                     (0.0, 350.1576972997113, 248.7907274896496),
                                     (0.0, 0.0, 1.0)])
        self.dist_coeffs = np.array([-0.05588267213463404, 0.06434203436364982, -0.09068050773123988, 0.04042938732850454])

    def __call__(self) -> None:

        ret, frame = self.camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return
        img = np.asarray(frame)

        keypoints = []
 
        predictions, gt_anns, image_meta = self.predictor.numpy_image(img)
        if len(predictions) == 0:
            print("No Keypoints found!")

        else:
            #print("found something: ")
            #print(predictions[0].data)
            keypoints = predictions[0].data

            success, rotation_vec, translation_vec = self.estimate_pose(keypoints)

            if success:
                self.plot_pnp_comp(frame=frame, keypoints=keypoints, rotation_vec=rotation_vec, translation_vec=translation_vec)

            self.plot_keypoints(frame=frame, keypoints=keypoints)
            self.publish_kp(keypoints)
        self.display_img(frame)


    def estimate_pose(self, keypoints) -> None:

        success = False
        nonzero_indices = []
        nonzero_skeleton = []
        nonzero_keypoints = []

        for i, kp in enumerate(keypoints):
            if kp[2] > 0.05:
                nonzero_indices.append(i)
                nonzero_skeleton.append(self.points_3d[i])
                nonzero_keypoints.append(kp[0:2])

        nonzero_indices = np.array(nonzero_indices)
        nonzero_skeleton = np.array(nonzero_skeleton)
        nonzero_keypoints = np.array(nonzero_keypoints)
        #print(nonzero_skeleton)
        #print(nonzero_keypoints)

        if np.shape(nonzero_keypoints)[0] > 3:
            try:
                success, rotation_vec, translation_vec = cv2.solvePnP(nonzero_skeleton, nonzero_keypoints, self.camera_matrix, self.dist_coeffs)
                pose_msg = get_pose_msg(translation_vec, rotation_vec, rospy.Time(0))
                self.pose_pub.publish(pose_msg)
                success = True
            except Exception as e:
                rotation_vec = np.empty([1,3])
                translation_vec = np.empty([1,3])
                print("error: ", e)

            return success, rotation_vec, translation_vec
        else:
            
            rotation_vec = np.empty([1,3])
            translation_vec = np.empty([1,3])
            return success, rotation_vec, translation_vec


    def publish_kp(self, keypoints) -> None:
        for i, kp in enumerate(keypoints):
            kp_msg = kp_to_msg(kp, rospy.Time(0))
            getattr(self, f'kp_pub_{i}').publish(kp_msg)

    def plot_keypoints(self, frame, keypoints) -> None:
        for p in keypoints:
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0,0,255) )
            cv2.putText(img=frame, text=str((round(p[2],4))), org=(int(p[0]), int(p[1])+4), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0, 255, 0),thickness=1)


    def plot_pnp_comp(self, frame, keypoints, rotation_vec, translation_vec) -> None:
        if not (rotation_vec.size == 0):
            top_point2d, jacobian = cv2.projectPoints(np.array([(0.0,0.0,1.185)]), rotation_vec, translation_vec, self.camera_matrix, self.dist_coeffs)
            bottom_point2d, jacobian = cv2.projectPoints(np.array([(0.0,0.0,0.0)]), rotation_vec, translation_vec, self.camera_matrix, self.dist_coeffs)

            point1 = ( int(top_point2d[0][0][0]), int(top_point2d[0][0][1]) )
            point2 = ( int(bottom_point2d[0][0][0]), int(bottom_point2d[0][0][1]) )

            cv2.line(frame, point1, point2, (255,255,255), 2)

    def display_img(self,frame) -> None:
        # Display the resulting frame
        cv2.imshow('keypoints', frame)
        cv2.waitKey(1)




def main():
    rospy.init_node("PolePoseNode", anonymous=True)

    in_topic = 'image'
    out_topic = 'keypoints'

    PoleDetector = PolePoseNode(in_topic, out_topic)

    #PoleDetector.display_keypoints()
    
    while not rospy.is_shutdown():
        PoleDetector()

    cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
    