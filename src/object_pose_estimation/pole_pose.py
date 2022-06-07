#!/usr/bin/env python

from ast import While
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import tensorflow as tf
from tensorflow import keras
import imgaug.augmenters as iaa

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
    # msg.point.z = keypoint[2] 2D!!
    return msg

def get_pose_msg(translation, rotation, time) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = time
    rotation = rotation.flatten()
    print(rotation)
    rotation_obj = R.from_euler('xyz', rotation)
    rotation_quat = rotation_obj.as_quat()
    print(rotation_quat)
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
        network = "/home/" + USERNAME + "/BT_Vision/keras_implementation/network"
        self.model = keras.models.load_model(network)
        self.camera = cv2.VideoCapture("/dev/v4l/by-id/usb-e-con_systems_See3CAM_CU55_0C10CE08-video-index0") #Attention: check port nr on udoo!
        if not self.camera.isOpened():
            print("[PolePoseNode] Cannot open camera!")

        self.kp_pub_0 = rospy.Publisher('PolePoseNode/KeyPoints/1', PointStamped, queue_size=1)
        self.kp_pub_1 = rospy.Publisher('PolePoseNode/KeyPoints/2', PointStamped, queue_size=1)
        self.kp_pub_2 = rospy.Publisher('PolePoseNode/KeyPoints/3', PointStamped, queue_size=1)
        self.kp_pub_3 = rospy.Publisher('PolePoseNode/KeyPoints/4', PointStamped, queue_size=1)
        self.kp_pub_4 = rospy.Publisher('PolePoseNode/KeyPoints/5', PointStamped, queue_size=1)
        self.kp_pub_5 = rospy.Publisher('PolePoseNode/KeyPoints/6', PointStamped, queue_size=1)
        self.kp_pub_6 = rospy.Publisher('PolePoseNode/KeyPoints/7', PointStamped, queue_size=1)

        self.keypoints = []
        self.pose_pub = rospy.Publisher('PolePoseNode/EstimatedPose', PoseStamped, queue_size=1)

        self.bridge = CvBridge()

        self.points_3d = np.array([
                                (0.0, 0.0, 0.0),
                                (-0.075, 0.0, -0.2),
                                (0.075, 0.0, -0.2),
                                (-0.075, 0.0, -0.7),
                                (0.075, 0.0, -0.7),
                                (-0.075, 0.0, -1.3),
                                (-0.075, 0.0, -1.3),
                                ])
        self.camera_matrix = np.array([(347.5293999809815 * 244/640, 0.0, 314.7548267525618 * 244/640),
                                     (0.0, 347.45033648440716 * 244/480, 247.32551331252066 * 244/480),
                                     (0.0, 0.0, 1.0)])
        self.dist_coeffs = np.array([-0.06442475368146962, 0.10266027381230053, -0.16303799346444728, 0.08403964035356283])

    def __call__(self) -> None:

        ret, frame = self.camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224), interpolation = cv2.INTER_LINEAR)
        img = np.expand_dims(img, axis=0)
        predictions = self.model.predict(img).reshape(-1, 7, 2) * 224
        self.keypoints = []
        if len(predictions) == 0:
            print("Did not find any Keypoints!")
        else:
            print(predictions[0])
            self.keypoints = predictions[0]

            self.publish_kp(self.keypoints)
            self.estimate_pose(self.keypoints)

    def estimate_pose(self, keypoints):
        success, rotation_vec, translation_vec = cv2.solvePnP(self.points_3d, keypoints, self.camera_matrix, self.dist_coeffs)
        pose_msg = get_pose_msg(translation_vec, rotation_vec, rospy.Time(0))
        self.pose_pub.publish(pose_msg)


    def publish_kp(self, keypoints) -> None:
        i = 0
        for kp in self.keypoints:
            kp_msg = kp_to_msg(kp, rospy.Time(0))
            getattr(self, f'kp_pub_{i}').publish(kp_msg)
            i += 1
            if i == 7:
                break

    def display_keypoints(self) -> None:
        ret = True
        while ret:
            # Capture frame-by-frame
            ret, frame = self.camera.read()

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            frame_res = cv2.resize(frame, (224,224), interpolation = cv2.INTER_LINEAR)
            # try:
            #     frame_res = np.ascontiguousarray(frame_res, dtype=np.uint8)
            # except Exception as e: 
            #     print("exception:", e)
            frame_data = np.expand_dims(frame_res, axis=0) # ADDED; CONTINUE HERE AND ADD IN __call__
            predictions = self.model.predict(frame_data).reshape(-1, 7, 2) * 224

            for p in predictions[0]:
                cv2.circle( frame_res, (int(p[0]), int(p[1])), 4, (0,0,255) )
            # Display the resulting frame

            cv2.imshow('keypoints', frame_res)

            if cv2.waitKey(1) == ord('q'):
                break


def main():
    rospy.init_node("PolePoseNode", anonymous=True)

    in_topic = 'image'
    out_topic = 'keypoints'

    PoleDetector = PolePoseNode(in_topic, out_topic)

    #PoleDetector.display_keypoints()
    
    while not rospy.is_shutdown():
        PoleDetector()


if __name__ == "__main__":
	main()
    