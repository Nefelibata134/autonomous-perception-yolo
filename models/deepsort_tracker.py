"""
models/deepsort_tracker.py
DeepSORT 多目标跟踪封装
核心：卡尔曼滤波（运动预测）+ 外观特征（ReID）+ 匈牙利算法（匹配）
"""
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2


class DeepsortTracker:
    def __init__(self, max_age=30, n_init=3, max_cosine_distance=0.3):
        """
        初始化 DeepSORT 跟踪器

        Args:
            max_age: 目标丢失后保留的最大帧数（防遮挡）
            n_init: 连续检测多少帧才确认为正式跟踪目标（防误检）
            max_cosine_distance: 外观特征相似度阈值（越小越严格）
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=100,  # 外观特征预算（内存控制）
            override_track_class=None
        )

        # 类别颜色映射
        self.color_map = {
            0: (0, 255, 0),  # car
            1: (255, 0, 0),  # person
            2: (0, 0, 255),  # traffic light
            3: (255, 255, 0),  # traffic sign
        }

    def update(self, detections, frame):
        """
        更新跟踪器（每帧调用一次）

        Args:
            detections: [[x1, y1, x2, y2, conf, cls_id], ...]（来自 YOLODetector）
            frame: 当前帧图片（用于提取外观特征）

        Returns:
            tracks: 跟踪结果列表，每个 track 包含 id, bbox, class 等
        """
        # 转换为 DeepSORT 需要的格式：[([x, y, w, h], confidence, class_name), ...]
        deepsort_dets = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            w = x2 - x1
            h = y2 - y1
            class_name = str(cls_id)  # DeepSORT 用字符串标识类别
            deepsort_dets.append(([x1, y1, w, h], conf, class_name))

        # 更新跟踪（内部执行：特征提取 → 卡尔曼预测 → 匈牙利匹配 → 更新状态）
        tracks = self.tracker.update_tracks(deepsort_dets, frame=frame)
        return tracks

    def draw_tracks(self, frame, tracks, draw_class=True):
        """
        绘制跟踪框 + 跟踪 ID

        Args:
            tracks: DeepSORT 跟踪结果
            draw_class: 是否显示类别名
        """
        for track in tracks:
            # 只绘制"已确认"的跟踪目标（过滤掉临时目标）
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # 获取 left, top, right, bottom
            x1, y1, x2, y2 = map(int, ltrb)

            # 获取类别（从原始检测继承）
            cls_id = int(track.get_det_class()) if track.get_det_class() else 0
            color = self.color_map.get(cls_id, (255, 255, 255))

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 标签：ID + 类别
            label = f"ID:{track_id}"
            if draw_class:
                class_name = {0: 'car', 1: 'person', 2: 'traffic light', 3: 'traffic sign'}.get(cls_id, '')
                label += f" {class_name}"

            # 画标签背景（防遮挡）
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return frame

    def get_track_info(self, tracks):
        """
        提取跟踪信息（用于后续分析，如车速估计、轨迹预测）
        """
        info = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            info.append({
                'id': track.track_id,
                'bbox': [int(v) for v in ltrb],
                'class': int(track.get_det_class()) if track.get_det_class() else 0,
                'hits': track.hits,  # 累计匹配成功次数
                'age': track.age,  # 存活帧数
            })
        return info