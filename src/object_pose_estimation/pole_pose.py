#!/usr/bin/env python

from ast import While
import rospy
import sys
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

# import tensorflow as tf
# from tensorflow import keras
# import imgaug.augmenters as iaa

import torch
import re
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
        #network = "/home/" + USERNAME + "/BT_Vision/outputs/" + "mobilenetv2-220705-125352-pole_detect.pkl.epoch198"
        network = "/home/" + USERNAME + "/BT_Vision/outputs/" + "mobilenetv2-220707-073441-pole_detect.pkl.epoch185"

        self.camera = cv2.VideoCapture("/dev/v4l/by-id/usb-e-con_systems_See3CAM_CU55_1020CE08-video-index0")
        # self.camera = cv2.VideoCapture("/home/nico/Videos/Test_Video.avi") # for debug
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture("/dev/v4l/by-id/usb-e-con_systems_See3CAM_CU55_0C10CE08-video-index0")


        self.predictor = openpifpaf.Predictor(checkpoint=network)

        if not self.camera.isOpened():
            print("[PolePoseNode] Cannot open camera!")

        self.save_images = False
        self.kp_pub_0 = rospy.Publisher('PolePoseNode/KeyPoints/1', PointStamped, queue_size=1)
        self.kp_pub_1 = rospy.Publisher('PolePoseNode/KeyPoints/2', PointStamped, queue_size=1)
        self.kp_pub_2 = rospy.Publisher('PolePoseNode/KeyPoints/3', PointStamped, queue_size=1)
        self.kp_pub_3 = rospy.Publisher('PolePoseNode/KeyPoints/4', PointStamped, queue_size=1)
        self.kp_pub_4 = rospy.Publisher('PolePoseNode/KeyPoints/5', PointStamped, queue_size=1)
        self.kp_pub_5 = rospy.Publisher('PolePoseNode/KeyPoints/6', PointStamped, queue_size=1)
        self.kp_pub_6 = rospy.Publisher('PolePoseNode/KeyPoints/7', PointStamped, queue_size=1)

        self.pose_pub = rospy.Publisher('PolePoseNode/EstimatedPose', PoseStamped, queue_size=1)
        self.image_pub = rospy.Publisher('PolePoseNode/image', Image, queue_size=1)

        self.imwrite_dir = "/home/" + USERNAME + "/BT_Vision/images/" 

        if len(os.listdir(self.imwrite_dir)) > 0:
            max_num = 0
            for file_name in os.listdir(self.imwrite_dir):

                img_num = int(file_name.split('.')[0].split('_')[1])
                print(img_num)
                if img_num > max_num:
                    max_num = img_num
            self.image_counter = max_num + 1
            print("image counter init with ", self.image_counter)
        else:   
            self.image_counter = 0

        # self.points_3d = np.array([
        #                         (0.0, 0.0,      1.185), # tip
        #                         (0.0, -0.0625,  1.0), # top_l
        #                         (0.0, 0.0625,   1.0), # top_r
        #                         (0.0, -0.0625,  0.505), # mid_l
        #                         (0.0, 0.0625,   0.505), # mid_r
        #                         (0.0, -0.0625,  0.0), # bottom_l
        #                         (0.0, 0.0625,   0.02) # bottom_r
        #                         ])

        self.points_3d = np.array([
                                (0.0, 0.0,      1.185), # tip
                                (0.0, -0.0645,  1.005), # top_l
                                (0.0, 0.0645,   1.005), # top_r
                                (0.0, -0.0645,  0.505), # mid_l
                                (0.0, 0.0645,   0.505), # mid_r
                                (0.0112, -0.0635,  0.0), # bottom_l
                                (0.0112, 0.0635,   0.02) # bottom_r
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

            success, rotation_vec, translation_vec = self.estimate_pose(keypoints = keypoints, frame = frame)

            if success:
                self.plot_pnp_comp(frame=frame, keypoints=keypoints, rotation_vec=rotation_vec, translation_vec=translation_vec)

            self.plot_keypoints(frame=frame, keypoints=keypoints)
            self.publish_kp(keypoints)
        #self.display_img(frame)
        self.publish_img(frame, rospy.Time.now())


    def estimate_pose(self, keypoints, frame):

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
        if  (0 < np.shape(nonzero_keypoints)[0] < 7) and self.save_images:
            filename = self.imwrite_dir + "img_" + str(self.image_counter) + ".png"
            print("saving image to: ", filename)
            cv2.imwrite(filename, frame)
            self.image_counter += 1
        if np.shape(nonzero_keypoints)[0] > 6:
            try:
                success, rotation_vec, translation_vec = cv2.solvePnP(nonzero_skeleton, nonzero_keypoints, self.camera_matrix, self.dist_coeffs, flags = cv2.SOLVEPNP_ITERATIVE)
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

            angles = [i for i in range(1, 360+1)]
            heights = [i for i in range(1, 500+1)]

            for height in heights:
                point_z_axis , _ = cv2.projectPoints(np.array([(0.0,0.0,height * 1.185 / 500)]), rotation_vec, translation_vec, self.camera_matrix, self.dist_coeffs)
                cv2.circle(frame, (int(point_z_axis[0][0][0]), int(point_z_axis[0][0][1])), 2, (255,255,255))



            for angle in angles:
                circle_point_2d_top, _ = cv2.projectPoints(np.array([(np.cos(np.radians(angle))*0.0625,np.sin(np.radians(angle))*0.0625,1.0)]), rotation_vec, translation_vec, self.camera_matrix, self.dist_coeffs)
                circle_point_2d_bottom, _ = cv2.projectPoints(np.array([(np.cos(np.radians(angle))*0.0625,np.sin(np.radians(angle))*0.0625,0.0)]), rotation_vec, translation_vec, self.camera_matrix, self.dist_coeffs)
                cv2.circle(frame, (int(circle_point_2d_top[0][0][0]), int(circle_point_2d_top[0][0][1])), 2, (255,255,255))
                cv2.circle(frame, (int(circle_point_2d_bottom[0][0][0]), int(circle_point_2d_bottom[0][0][1])), 2, (255,255,255))

            for keypoint_3d in list(self.points_3d):
                keypoint_proj , _ = cv2.projectPoints(np.array(keypoint_3d), rotation_vec, translation_vec, self.camera_matrix, self.dist_coeffs)
                cv2.circle(frame, (int(keypoint_proj[0][0][0]), int(keypoint_proj[0][0][1])), 3, (0,0,0))
                cv2.circle(frame, (int(keypoint_proj[0][0][0]), int(keypoint_proj[0][0][1])), 2, (0,0,0))
                cv2.circle(frame, (int(keypoint_proj[0][0][0]), int(keypoint_proj[0][0][1])), 1, (0,0,0))


    def display_img(self,frame) -> None:
        # Display the resulting frame
        cv2.imshow('keypoints', frame)
        cv2.waitKey(1)

    def publish_img(self, frame, time) -> None:
        bridge = CvBridge()
        try:
            img_msg = bridge.cv2_to_imgmsg(frame)
            img_msg.header.stamp = time
            self.image_pub.publish(img_msg)
        except Exception as e:
            print("Converting Image failed, error:", e)




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
    