"""
models/bev_transform.py
BEV 鸟瞰图转换：基于 IPM（逆透视变换）
简化版：假设相机高度 1.2m，俯角 20°，生成单应性矩阵
"""
import cv2
import numpy as np


class BEVTransform:
    def __init__(self, src_size=(640, 480), bev_size=(400, 600)):
        """
        初始化 BEV 转换器

        Args:
            src_size: 输入前视图分辨率 (w, h)
            bev_size: 输出 BEV 图分辨率 (w, h)
        """
        self.src_w, self.src_h = src_size
        self.bev_w, self.bev_h = bev_size

        # 计算单应性矩阵 H（前视图 → BEV）
        self.H = self._compute_homography()

        # 颜色映射（跟踪 ID 颜色，和 DeepSORT 一致）
        self.color_map = {
            0: (0, 255, 0),  # car
            1: (255, 0, 0),  # person
            2: (0, 0, 255),  # traffic light
            3: (255, 255, 0),  # traffic sign
        }

    def _compute_homography(self):
        """
        计算单应性矩阵（简化版，基于典型车载相机参数）

        真实项目中，这个 H 应该通过相机标定（chessboard）获得。
        这里用经验值模拟：假设相机高度 1.2m，俯角 25°，安装在前挡风玻璃。
        """
        # 前视图中的 4 个地面参考点（梯形区域）
        # 顺序：左上、右上、右下、左下（逆时针）
        src_pts = np.float32([
            [self.src_w * 0.3, self.src_h * 0.6],  # 左上：远处左车道线
            [self.src_w * 0.7, self.src_h * 0.6],  # 右上：远处右车道线
            [self.src_w * 0.9, self.src_h * 0.95],  # 右下：近处右
            [self.src_w * 0.1, self.src_h * 0.95],  # 左下：近处左
        ])

        # BEV 中的对应 4 个点（矩形，鸟瞰）
        dst_pts = np.float32([
            [self.bev_w * 0.2, 0],  # 左上：远处
            [self.bev_w * 0.8, 0],  # 右上
            [self.bev_w * 0.8, self.bev_h],  # 右下：近处
            [self.bev_w * 0.2, self.bev_h],  # 左下
        ])

        # 计算透视变换矩阵
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return H

    def transform(self, frame):
        """
        将前视图转换为 BEV 鸟瞰图

        Args:
            frame: 前视图 BGR 图片

        Returns:
            bev: BEV 鸟瞰图
        """
        bev = cv2.warpPerspective(frame, self.H, (self.bev_w, self.bev_h))
        return bev

    def transform_point(self, x, y):
        """
        将前视图中的一个点 (x, y) 映射到 BEV 坐标

        Args:
            x, y: 前视图像素坐标

        Returns:
            (bev_x, bev_y): BEV 像素坐标，或 None（如果映射失败）
        """
        # 齐次坐标变换
        pt = np.array([[x, y]], dtype=np.float32)
        pt = np.array([pt])
        dst = cv2.perspectiveTransform(pt, self.H)

        bev_x, bev_y = int(dst[0][0][0]), int(dst[0][0][1])

        # 边界检查
        if 0 <= bev_x < self.bev_w and 0 <= bev_y < self.bev_h:
            return (bev_x, bev_y)
        return None

    def draw_tracks_bev(self, bev_canvas, tracks_info):
        """
        在 BEV 画布上绘制跟踪目标的位置

        Args:
            bev_canvas: BEV 背景图（可以是黑色背景或 warpPerspective 后的图）
            tracks_info: 跟踪信息列表，每个元素包含 'id', 'bbox', 'class'
                         bbox 格式 [x1, y1, x2, y2]（前视图坐标）
        """
        for info in tracks_info:
            track_id = info['id']
            x1, y1, x2, y2 = info['bbox']
            cls_id = info['class']

            # 取边界框底部中心点（代表车辆/行人在地面的位置）
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)  # 底部

            # 映射到 BEV
            bev_pt = self.transform_point(foot_x, foot_y)
            if bev_pt is None:
                continue

            bev_x, bev_y = bev_pt
            color = self.color_map.get(cls_id, (255, 255, 255))

            # 在 BEV 上画圆点代表目标位置
            cv2.circle(bev_canvas, (bev_x, bev_y), 8, color, -1)

            # 画 ID
            label = f"ID:{track_id}"
            cv2.putText(bev_canvas, label, (bev_x + 10, bev_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return bev_canvas

    def draw_bev_grid(self, bev_canvas, grid_spacing=50):
        """
        在 BEV 图上绘制网格（模拟车道线/距离刻度）
        """
        h, w = bev_canvas.shape[:2]

        # 纵向网格（车道线方向）
        for x in range(0, w, grid_spacing):
            cv2.line(bev_canvas, (x, 0), (x, h), (50, 50, 50), 1)

        # 横向网格（距离刻度，越往下越近）
        for y in range(0, h, grid_spacing):
            cv2.line(bev_canvas, (0, y), (w, y), (50, 50, 50), 1)
            # 标注距离（简化：假设每格 2 米）
            dist = int((h - y) / h * 40)  # 最远约 40 米
            cv2.putText(bev_canvas, f"{dist}m", (5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        return bev_canvas