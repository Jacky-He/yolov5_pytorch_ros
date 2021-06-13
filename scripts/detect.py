#!/usr/bin/env python

import argparse
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImage, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

## ROS IMPORTS ##
import rospy
import std_msgs.msg
from rospkg import RosPack
from std_msgs.msg import UInt8
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, Point32
from yolov5_pytorch_ros.msg import BoundingBox, BoundingBoxes
from cv_bridge import CvBridge, CvBridgeError

package = RosPack()
package_path = package.get_path('yolov5_pytorch_ros')

class DetectorManager():

    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic")
        self.weights = rospy.get_param("~weights")
        self.weights = os.path.join(package_path, "scripts", "weights", self.weights)
        self.imgsz = int(rospy.get_param("~imgsz"))
        self.conf_thres = float(rospy.get_param("~conf-thres"))
        self.iou_thres = float(rospy.get_param("~iou-thres"))
        self.max_det = int(rospy.get_param("~max-det"))
        self.device = rospy.get_param("~device")
        self.publish_image = rospy.get_param("~publish_image")
        self.detected_objects_topic = rospy.get_param("~detected_objects_topic")
        self.detections_image_topic = rospy.get_param("~detections_image_topic")
        self.line_thickness = int(rospy.get_param("~bbox_line_thickness"))

        # Initialize
        self.device = select_device(self.device)

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names

        # Load CvBridge
        self.bridge = CvBridge()
        
        # Define subscribers
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_detect, queue_size = 1, buff_size = 2**24)

        # Define publishers
        self.pub_ = rospy.Publisher(self.detected_objects_topic, BoundingBoxes, queue_size=10)
        self.pub_viz_ = rospy.Publisher(self.detections_image_topic, Image, queue_size=10)
        rospy.loginfo("Launched node for object detection")

        # Spin
        rospy.spin()
        return

    def image_detect(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Initialize detection results
        detection_results = BoundingBoxes()
        detection_results.header = data.header
        detection_results.image_header = data.header
        
        # DataLoader
        dataset = LoadImage(self.cv_image, img_size=self.imgsz, stride=self.stride)

        # Run Inference
        torch.no_grad()
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        for img, im0s in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)

            # Process detections
            for _, det in enumerate(pred):  # detections per image
                im0 = im0s.copy()
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = f'{self.names[c]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=self.line_thickness)

                        # Populate message 
                        detection_msg = BoundingBox()
                        detection_msg.xmin = int(xyxy[0].item())
                        detection_msg.ymin = int(xyxy[1].item())
                        detection_msg.xmax = int(xyxy[2].item())
                        detection_msg.ymax = int(xyxy[3].item())
                        detection_msg.probability = conf.item() 
                        detection_msg.Class = self.names[c]
                        detection_results.bounding_boxes.append(detection_msg)
                self.pub_.publish(detection_results)
                image_msg = self.bridge.cv2_to_imgmsg(im0, "bgr8")
                self.pub_viz_.publish(image_msg)
                
if __name__ == '__main__':
    rospy.init_node("detector_manager_node")
    dm = DetectorManager()
