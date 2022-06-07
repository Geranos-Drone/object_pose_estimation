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

from geometry_msgs.msg import PointStamped


def _to_msg(keypoint, time) -> PointStamped:
    msg = PointStamped()
    msg.header.stamp = time
    msg.point.x = keypoint[0]
    msg.point.y = keypoint[1]
    msg.point.z = keypoint[2]
    return msg

class PolePoseNode:
    def __init__(self, in_topic, out_topic) -> None:
        self.model = keras.models.load_model('/home/nico/ComputerVisionBA/Code/BT_Vision/keras_implementation/network')
        self.camera = cv2.VideoCapture("/dev/v4l/by-id/usb-e-con_systems_See3CAM_CU55_0C10CE08-video-index0") #Attention: check port nr on udoo!
        if not self.camera.isOpened():
            print("[PolePoseNode] Cannot open camera!")
        self.keypoints = []
        self.kp_pub_0 = rospy.Publisher('PifPaf/KeyPoints/1', PointStamped, queue_size=1)
        self.kp_pub_1 = rospy.Publisher('PifPaf/KeyPoints/2', PointStamped, queue_size=1)
        self.kp_pub_2 = rospy.Publisher('PifPaf/KeyPoints/3', PointStamped, queue_size=1)
        self.kp_pub_3 = rospy.Publisher('PifPaf/KeyPoints/4', PointStamped, queue_size=1)
        self.kp_pub_4 = rospy.Publisher('PifPaf/KeyPoints/5', PointStamped, queue_size=1)
        self.kp_pub_5 = rospy.Publisher('PifPaf/KeyPoints/6', PointStamped, queue_size=1)
        self.kp_pub_6 = rospy.Publisher('PifPaf/KeyPoints/7', PointStamped, queue_size=1)

        self.bridge = CvBridge()

    def __call__(self) -> None:

        ret, frame = self.camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224), interpolation = cv2.INTER_LINEAR)
        img = np.expand_dims(img, axis=0)
        predictions = self.model.predict(img).reshape(-1, 7, 2) * 224
        #predictions, gt_anns, image_meta = self.predictor.numpy_image(img) #dtype predicitions: list
        if len(predictions) == 0:
            print("Did not find any Keypoints!")
        else:
            for pred in predictions:
                #print(pred)
                self.keypoints.append(pred)
            self.publish(keypoints)

    def publish(self, keypoints) -> None:
        i = 0
        for kp in self.keypoints:
            kp_msg = _to_msg(kp, rospy.Time(0))
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
                print(p)
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

    PoleDetector.display_keypoints()
    
    while not rospy.is_shutdown():
        PoleDetector()


if __name__ == "__main__":
	main()
    