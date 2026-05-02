"""
models/yolo_detector.py
YOLOv8 检测器封装类（OOP，方便跟踪模块调用）
"""
from ultralytics import YOLO
import cv2


class YOLODetector:
    def __init__(self, weights='runs/detect/train/weights/best.pt', conf=0.5, device=0):
        """
        初始化检测器

        Args:
            weights: 训练好的权重路径（默认用 Day3 的 best.pt）
            conf: 置信度阈值
            device: 0=GPU, 'cpu'=CPU
        """
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device
        self.class_names = {
            0: 'car',
            1: 'person',
            2: 'traffic light',
            3: 'traffic sign'
        }

    def detect(self, frame):
        """
        单帧检测

        Args:
            frame: BGR 图片 (numpy array)

        Returns:
            results: ultralytics Results 对象
        """
        results = self.model(frame, conf=self.conf, device=self.device, verbose=False)
        return results[0] if results else None

    def get_detections(self, result):
        """
        将 ultralytics 结果转换为 DeepSORT 需要的格式

        Returns:
            list: [[x1, y1, x2, y2, conf, cls_id], ...]
        """
        if result is None or result.boxes is None:
            return []

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf)
            cls_id = int(box.cls)
            detections.append([x1, y1, x2, y2, conf, cls_id])

        return detections

    def draw_boxes(self, frame, result, color_map=None):
        """
        绘制检测框（不带 ID，纯检测可视化）
        """
        if color_map is None:
            color_map = {
                0: (0, 255, 0),  # car: 绿
                1: (255, 0, 0),  # person: 蓝
                2: (0, 0, 255),  # traffic light: 红
                3: (255, 255, 0),  # traffic sign: 青
            }

        for det in self.get_detections(result):
            x1, y1, x2, y2, conf, cls_id = map(int, det)
            color = color_map.get(cls_id, (255, 255, 255))
            label = f"{self.class_names.get(cls_id, 'unknown')} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame