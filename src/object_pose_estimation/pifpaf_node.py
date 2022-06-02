#!/usr/bin/env python

from ast import While
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from openpifpaf.predictor import Predictor
from geometry_msgs.msg import PointStamped


def _to_msg(keypoint, time) -> PointStamped:
    msg = PointStamped()
    msg.header.stamp = time
    msg.point.x = keypoint[0]
    msg.point.y = keypoint[1]
    msg.point.z = keypoint[2]
    return msg

class PifPafNode:
    def __init__(self, in_topic, out_topic) -> None:
        self.predictor = Predictor(checkpoint="resnet50")
        self.camera = cv2.VideoCapture(2) #Attention: check port nr on udoo!
        if not self.camera.isOpened():
            print("[pifpaf_node] Cannot open camera!")
        
        self.kp_pub_0 = rospy.Publisher('PifPaf/KeyPoints/0', PointStamped, queue_size=1)
        self.kp_pub_1 = rospy.Publisher('PifPaf/KeyPoints/1', PointStamped, queue_size=1)
        self.kp_pub_2 = rospy.Publisher('PifPaf/KeyPoints/2', PointStamped, queue_size=1)
        self.kp_pub_3 = rospy.Publisher('PifPaf/KeyPoints/3', PointStamped, queue_size=1)
        self.kp_pub_4 = rospy.Publisher('PifPaf/KeyPoints/4', PointStamped, queue_size=1)
        self.kp_pub_5 = rospy.Publisher('PifPaf/KeyPoints/5', PointStamped, queue_size=1)
        self.kp_pub_6 = rospy.Publisher('PifPaf/KeyPoints/6', PointStamped, queue_size=1)
        self.kp_pub_7 = rospy.Publisher('PifPaf/KeyPoints/7', PointStamped, queue_size=1)

        self.bridge = CvBridge()

    def __call__(self) -> None:
        ret, frame = self.camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        predictions, gt_anns, image_meta = self.predictor.numpy_image(img) #dtype predicitions: list
        keypoints = []
        if len(predictions) == 0:
            print("Did not find any Keypoints!")
        else:
            for pred in predictions[0].data:
                print(pred)
                keypoints.append(pred)
            self.publish(keypoints)

    def publish(self, keypoints) -> None:
        i = 0
        for kp in keypoints:
            kp_msg = _to_msg(kp, rospy.Time(0))
            getattr(self, f'kp_pub_{i}').publish(kp_msg)
            i += 1
            if i == 8:
                break

    def display_video(self) -> None:
        ret = True
        while ret:
            # Capture frame-by-frame
            ret, frame = self.camera.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Display the resulting frame
            cv2.imshow('gray', gray)
            if cv2.waitKey(1) == ord('q'):
                break
        



def main():
    rospy.init_node("pifpaf_node", anonymous=True)

    in_topic = 'image'
    out_topic = 'keypoints'

    PoleDetector = PifPafNode(in_topic, out_topic)

    PoleDetector.display_video()
    
    while not rospy.is_shutdown():
        PoleDetector()


if __name__ == "__main__":
	main()
    