"""
inference_bev.py
前视图 + 检测 + 跟踪 + BEV 鸟瞰图（四合一可视化）
输出：左右并排（左=前视图跟踪，右=BEV 俯视）
"""
import cv2
import argparse
import numpy as np
from pathlib import Path
from models.yolo_detector import YOLODetector
from models.deepsort_tracker import DeepsortTracker
from models.bev_transform import BEVTransform


def inference_bev(source, output='output_bev.mp4'):
    """
    完整 pipeline：检测 → 跟踪 → BEV 转换 → 可视化
    """
    # 初始化三个模块
    print("🚀 初始化检测器...")
    detector = YOLODetector(weights='runs/detect/train/weights/best.pt', conf=0.5)

    print("🚀 初始化跟踪器...")
    tracker = DeepsortTracker(max_age=30, n_init=3)

    print("🚀 初始化 BEV 转换器...")
    bev_transform = BEVTransform(src_size=(1280, 720), bev_size=(400, 600))

    # 打开视频
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ 无法打开: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出尺寸：左(前视图) + 右(BEV)，并排
    out_w = w + 400  # 原图宽 + BEV 宽
    out_h = max(h, 600)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (out_w, out_h))

    frame_count = 0

    print(f"▶️ 开始处理: {source} ({w}x{h})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 检测
        result = detector.detect(frame)
        detections = detector.get_detections(result)

        # 2. 跟踪
        tracks = tracker.update(detections, frame)
        tracks_info = tracker.get_track_info(tracks)

        # 3. 前视图绘制（带跟踪框）
        vis_front = tracker.draw_tracks(frame.copy(), tracks)

        # 4. BEV 转换
        bev = bev_transform.transform(frame)

        # 5. BEV 绘制（黑色背景 + 网格 + 目标点）
        bev_canvas = np.zeros((600, 400, 3), dtype=np.uint8)
        bev_canvas = bev_transform.draw_bev_grid(bev_canvas)
        bev_canvas = bev_transform.draw_tracks_bev(bev_canvas, tracks_info)

        # 6. 合并：左=前视图，右=BEV
        # 调整前视图高度匹配输出
        front_resized = cv2.resize(vis_front, (w, out_h))
        bev_resized = cv2.resize(bev_canvas, (400, out_h))

        combined = np.hstack([front_resized, bev_resized])

        # 添加标题
        cv2.putText(combined, "Front View + Tracking", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "BEV (Bird's Eye View)", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(combined)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   已处理 {frame_count} 帧，跟踪目标: {len(tracks_info)}")

    cap.release()
    out.release()

    print(f"\n✅ BEV 推理完成！")
    print(f"   输出: {output}")
    print(f"   总帧数: {frame_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assets/real_street.mp4')
    parser.add_argument('--output', type=str, default='output_bev.mp4')
    args = parser.parse_args()

    inference_bev(args.source, args.output)

    # 复制到 assets
    if Path(args.output).exists():
        import shutil
        shutil.copy(args.output, 'assets/demo_bev.mp4')
        print("✅ 已复制到 assets/demo_bev.mp4")