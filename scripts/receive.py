#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2, os
from cv_bridge import CvBridge, CvBridgeError
from rospkg import RosPack

package = RosPack()
package_path = package.get_path('yolov5_pytorch_ros')

def callback(data):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    path = os.path.join(package_path, "scripts", "detected_image.png")
    cv2.imwrite(path, cv_image)

def receive():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('receive', anonymous=True)

    rospy.Subscriber('detections_image_topic', Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    receive()
