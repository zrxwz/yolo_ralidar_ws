#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv_bridge
import numpy as np
import roslib.packages
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult
import cv2
import logging

# 禁用 YOLO 的日志输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# KCF tracker state variables
isTracking = False
bbox = None  # KCF 跟踪框
yolo_bbox = None  # YOLO 原始检测框对应的当前跟踪目标
kcf_tracker = cv2.legacy.TrackerKCF_create()  # KCF tracker for OpenCV 4.x


class TrackerNode:
    def __init__(self):
        # 获取 ROS 参数
        yolo_model = rospy.get_param("~yolo_model", "yolov8n.pt")
        self.input_topic = rospy.get_param("~input_topic", "image_raw")
        self.result_topic = rospy.get_param("~result_topic", "yolo_result")
        self.result_image_topic = rospy.get_param("~result_image_topic", "yolo_image")
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.max_det = rospy.get_param("~max_det", 300)
        
        # 获取并处理类别参数
        self.classes = rospy.get_param("~classes", "0")  # 默认为 '0'
        if isinstance(self.classes, int):
            self.classes = [self.classes]
        elif isinstance(self.classes, str):
            self.classes = [int(c) for c in self.classes.split(',')]

        self.device = rospy.get_param("~device", None)
        self.result_conf = rospy.get_param("~result_conf", True)
        self.result_line_width = rospy.get_param("~result_line_width", None)
        self.result_font_size = rospy.get_param("~result_font_size", None)
        self.result_font = rospy.get_param("~result_font", "Arial.ttf")
        self.result_labels = rospy.get_param("~result_labels", True)
        self.result_boxes = rospy.get_param("~result_boxes", True)

        # 获取 YOLO 模型路径
        path = roslib.packages.get_pkg_dir("ultralytics_ros")
        self.model = YOLO(f"{path}/models/{yolo_model}", verbose=False)  # 设置模型不输出详细信息
        self.model.fuse()

        # 初始化 ROS 订阅和发布
        self.sub = rospy.Subscriber(self.input_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.results_pub = rospy.Publisher(self.result_topic, YoloResult, queue_size=1)
        self.result_image_pub = rospy.Publisher(self.result_image_topic, Image, queue_size=1)

        self.bridge = cv_bridge.CvBridge()

    def adjust_tracking_bbox(self, frame):
        """
        使用 YOLO 检测框初始化跟踪框，并记录相关的检测信息
        """
        global bbox, isTracking, yolo_bbox

        # 获取 YOLO 的检测结果
        results = self.model(frame)

        best_bbox = None
        max_conf = 0
        class_name = ""
        detected = False

        # 查找目标类别和置信度最高的检测框
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # YOLO 检测框坐标
            conf = result.conf[0].item()
            cls = result.cls[0].item()
            detected_class_name = self.model.names[int(cls)]

            if self.classes is None or cls in self.classes:
                if conf > max_conf:
                    max_conf = conf
                    best_bbox = (x1, y1, x2 - x1, y2 - y1)  # YOLO 检测框
                    yolo_bbox = result.xyxy[0].tolist()  # 保存原始 YOLO 检测框
                    class_name = detected_class_name
                    detected = True

        if detected and best_bbox and max_conf > self.conf_thres:
            bbox = best_bbox  # 初始化跟踪框
            isTracking = True
            kcf_tracker.init(frame, tuple(bbox))  # 初始化 KCF 跟踪器
            rospy.loginfo(f"New tracking initialized: {bbox}")
        else:
            isTracking = False
            yolo_bbox = None
            rospy.logwarn("No valid detection for tracking initialization.")

    def create_detections_array(self):
        """
        生成当前跟踪目标对应的 YOLO 检测框消息
        """
        detections_msg = Detection2DArray()

        if yolo_bbox is not None:
            x1, y1, x2, y2 = map(int, yolo_bbox)
            w, h = x2 - x1, y2 - y1
            detection_msg = Detection2D()
            detection_msg.bbox.center.x = x1 + w / 2
            detection_msg.bbox.center.y = y1 + h / 2
            detection_msg.bbox.size_x = w
            detection_msg.bbox.size_y = h

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(self.classes[0])
            hypothesis.score = 1.0
            detection_msg.results.append(hypothesis)
            detections_msg.detections.append(detection_msg)

        return detections_msg

    def create_result_image(self, frame):
        """
        将 OpenCV 图像转换为 ROS 格式的 sensor_msgs/Image
        """
        return self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

    def image_callback(self, msg):
        """
        处理图像消息，绘制跟踪框并发布 YOLO 检测框信息
        """
        global bbox, isTracking

        # 将ROS图像消息转换为OpenCV图像
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        try:
            # 如果未跟踪，或者跟踪失败，则重新检测目标
            if not isTracking:
                self.adjust_tracking_bbox(cv_image)
            else:
                ok, bbox = kcf_tracker.update(cv_image)
                if ok:
                    x, y, w, h = tuple(map(int, bbox))
                    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制跟踪框
                    cv2.putText(
                        cv_image, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                    )
                else:
                    rospy.logwarn("Tracking failed, reinitializing tracking.")
                    isTracking = False
                    self.adjust_tracking_bbox(cv_image)  # 重新检测并初始化跟踪

            # 发布 YOLO 原始检测框信息
            yolo_result_msg = YoloResult()
            yolo_result_image_msg = Image()
            yolo_result_msg.header = msg.header
            yolo_result_image_msg.header = msg.header
            yolo_result_msg.detections = self.create_detections_array()

            # 将绘制了跟踪框的图像转换并发布
            yolo_result_image_msg = self.create_result_image(cv_image)

            self.results_pub.publish(yolo_result_msg)
            self.result_image_pub.publish(yolo_result_image_msg)

        except Exception as e:
            rospy.logwarn(f"Exception in image_callback: {e}")


if __name__ == "__main__":
    rospy.init_node("tracker_node")
    node = TrackerNode()
    rospy.spin()
