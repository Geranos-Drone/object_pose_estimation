#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

from std_msgs.msg import String
from sensor_msgs.msg import Image

class VideoNode():
    def __init__(self):
        self.camera = cv2.VideoCapture(2)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("image_topic_2",Image, queue_size=1)

    def __call__(self):
        ret, frame = self.camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return
        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.image_pub.publish(img_msg)
        except Exception as e:
            print("Converting Image failed", e)

def main():
  ROSVideo = VideoNode()
  rospy.init_node('ros_video_node', anonymous=True)
  while not rospy.is_shutdown():
      ROSVideo()
      rospy.sleep(0.01)

if __name__ == '__main__':
    main()
