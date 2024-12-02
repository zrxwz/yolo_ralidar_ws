#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv_bridge  # 用于ROS与OpenCV之间的图像转换
import numpy as np  # 用于数值计算，特别是处理数组和矩阵
import roslib.packages  # ROS包的管理模块
import rospy  # ROS的Python客户端库
from sensor_msgs.msg import Image  # 导入ROS图像消息类型
from ultralytics import YOLO  # 导入Ultralytics YOLO模型库
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose  # 导入YOLO检测结果消息类型
from ultralytics_ros.msg import YoloResult  # 导入自定义消息类型YoloResult

# 定义跟踪节点的类
class TrackerNode:
    def __init__(self):
        # 从ROS参数服务器获取参数，设置YOLO模型和其他配置
        yolo_model = rospy.get_param("~yolo_model", "yolov8n.pt")  # YOLO模型文件名
        self.input_topic = rospy.get_param("~input_topic", "image_raw")  # 输入图像话题
        self.result_topic = rospy.get_param("~result_topic", "yolo_result")  # YOLO结果发布话题
        self.result_image_topic = rospy.get_param("~result_image_topic", "yolo_image")  # YOLO结果图像发布话题
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)  # 检测置信度阈值
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)  # IOU阈值
        self.max_det = rospy.get_param("~max_det", 300)  # 最大检测数
        self.classes = rospy.get_param("~classes", None)  # 要检测的类别
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml")  # 跟踪器配置文件
        self.device = rospy.get_param("~device", None)  # 设备选择，默认为None
        self.result_conf = rospy.get_param("~result_conf", True)  # 是否显示置信度
        self.result_line_width = rospy.get_param("~result_line_width", None)  # 结果框线宽
        self.result_font_size = rospy.get_param("~result_font_size", None)  # 字体大小
        self.result_font = rospy.get_param("~result_font", "Arial.ttf")  # 字体类型
        self.result_labels = rospy.get_param("~result_labels", True)  # 是否显示标签
        self.result_boxes = rospy.get_param("~result_boxes", True)  # 是否显示框

        # 获取Ultralytics ROS包路径
        path = roslib.packages.get_pkg_dir("ultralytics_ros")
        # 加载YOLO模型
        self.model = YOLO(f"{path}/models/{yolo_model}")
        self.model.fuse()  # 对模型进行融合，优化模型

        # 订阅输入图像话题
        self.sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,  # 图像回调函数
            queue_size=1,
            buff_size=2**24,  # 缓冲区大小
        )

        # 发布检测结果和结果图像
        self.results_pub = rospy.Publisher(self.result_topic, YoloResult, queue_size=1)
        self.result_image_pub = rospy.Publisher(self.result_image_topic, Image, queue_size=1)
        
        # 创建OpenCV到ROS消息的转换桥接器
        self.bridge = cv_bridge.CvBridge()
        
        # 判断模型是否为分割模型（文件名以"-seg.pt"结尾）
        self.use_segmentation = yolo_model.endswith("-seg.pt")

    def image_callback(self, msg):
        """
        图像回调函数，接收来自图像话题的消息。
        """
        # 将ROS图像消息转换为OpenCV图像
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # 使用YOLO模型进行目标检测和跟踪
        results = self.model.track(
            source=cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,  # 使用指定的跟踪器
            device=self.device,  # 使用指定设备
            verbose=False,
            retina_masks=True,  # 是否使用RetinaNet模型的掩模
        )

        # 如果检测结果不为空，进行结果发布
        if results is not None:
            yolo_result_msg = YoloResult()  # 创建YOLO结果消息
            yolo_result_image_msg = Image()  # 创建图像消息
            yolo_result_msg.header = msg.header
            yolo_result_image_msg.header = msg.header
            # 创建检测结果数组，并填充
            yolo_result_msg.detections = self.create_detections_array(results)
            # 创建带框图像
            yolo_result_image_msg = self.create_result_image(results)
            
            # 如果是分割模型，添加分割掩模
            if self.use_segmentation:
                yolo_result_msg.masks = self.create_segmentation_masks(results)

            # 发布检测结果
            self.results_pub.publish(yolo_result_msg)
            # 发布带框图像
            self.result_image_pub.publish(yolo_result_image_msg)

    def create_detections_array(self, results):
        """
        创建YOLO检测结果数组。
        """
        detections_msg = Detection2DArray()  # 创建检测结果数组
        bounding_box = results[0].boxes.xywh  # 获取检测框（xywh）
        classes = results[0].boxes.cls  # 获取检测框的类别
        confidence_score = results[0].boxes.conf  # 获取检测框的置信度
        
        # 遍历每个检测框
        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            detection = Detection2D()  # 创建Detection2D消息
            detection.bbox.center.x = float(bbox[0])
            detection.bbox.center.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])
            hypothesis = ObjectHypothesisWithPose()  # 创建假设消息
            hypothesis.id = int(cls)  # 类别ID
            hypothesis.score = float(conf)  # 置信度
            detection.results.append(hypothesis)  # 添加假设到检测结果
            detections_msg.detections.append(detection)  # 添加检测结果到数组

        return detections_msg

    def create_result_image(self, results):
        """
        根据检测结果绘制图像并返回ROS图像消息。
        """
        # 绘制检测结果框
        plotted_image = results[0].plot(
            conf=self.result_conf,
            line_width=self.result_line_width,
            font_size=self.result_font_size,
            font=self.result_font,
            labels=self.result_labels,
            boxes=self.result_boxes,
        )
        # 转换为ROS图像消息
        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
        return result_image_msg

    def create_segmentation_masks(self, results):
        """
        创建分割掩模消息。
        """
        masks_msg = []  # 创建掩模消息列表
        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for mask_tensor in result.masks:
                    mask_numpy = (
                        np.squeeze(mask_tensor.data.to("cpu").detach().numpy()).astype(
                            np.uint8
                        )
                        * 255
                    )
                    # 将掩模图像转换为ROS消息
                    mask_image_msg = self.bridge.cv2_to_imgmsg(mask_numpy, encoding="mono8")
                    masks_msg.append(mask_image_msg)
        return masks_msg


# ROS节点的入口函数
if __name__ == "__main__":
    rospy.init_node("tracker_node")  # 初始化ROS节点
    node = TrackerNode()  # 创建TrackerNode实例
    rospy.spin()  # 保持节点持续运行
