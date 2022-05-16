#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from openpifpaf.predictor import Predictor
from geometry_msgs.msg import PointStamped


def _to_msg(keypoint, time):
    msg = PointStamped()
    msg.header.stamp = time
    msg.point.x = keypoint[0]
    msg.point.y = keypoint[1]
    msg.point.z = keypoint[2]
    return msg

class PifPafNode:
    def __init__(self, in_topic, out_topic):
        self.predictor = Predictor()
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

    def callback(self, data):
        rgb_img = self.bridge.imgmsg_to_cv2(data, "rgb8")
        pred, _, meta = self.predictor.numpy_image(rgb_img)

        keypoints = []
        for p in pred:
            kpoint = p.json_data()
            keypoints.append(kpoint)

        self.publish(keypoints)

    def publish(self, keypoints):
        i = 0
        for kp in keypoints:
            kp_msg = _to_msg(kp)
            getattr(self, f'kp_pub_{i}').publish(msg)
            i += 1

    def display_video(self):
        # Capture frame-by-frame
        ret, frame = self.camera.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return False
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('frame', gray)
        return True



def main():
    rospy.init_node("pifpaf_node", anonymous=True)

    in_topic = 'image'
    out_topic = 'keypoints'

    # recog = PifPafNode(in_topic, out_topic)
    
    # while not rospy.is_shutdown():
    #     if not recog.display_video():
    #         break

    camera = cv2.VideoCapture(2)

    if not camera.isOpened():
            print("[pifpaf_node] Cannot open camera!")

    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
    