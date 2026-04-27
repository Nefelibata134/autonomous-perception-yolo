"""
inference.py
单张图/视频推理脚本
"""
import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path


def detect_image(model_path, source, conf=0.5, save=True):
    """
    单张图推理

    Args:
        model_path: YOLO 模型路径（yolov8n.pt 等）
        source: 图片路径
        conf: 置信度阈值（低于此值的框不显示）
        save: 是否保存结果
    """
    # 加载模型（自动下载预训练权重）
    model = YOLO(model_path)

    # 推理
    results = model(source, conf=conf, verbose=True)

    # 解析结果
    result = results[0]  # 单张图只有1个结果

    # 打印检测到的物体
    boxes = result.boxes  # 检测框
    print(f"\n检测到 {len(boxes)} 个物体：")

    for i, box in enumerate(boxes):
        cls_id = int(box.cls)  # 类别ID
        conf_score = float(box.conf)  # 置信度
        name = result.names[cls_id]  # 类别名

        # 坐标
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"  {i + 1}. {name} ({conf_score:.2f}) 位置: ({x1},{y1})-({x2},{y2})")

    # 保存可视化结果
    if save:
        output_path = Path(source).stem + "_detected.jpg"
        result.save(filename=output_path)
        print(f"\n结果已保存: {output_path}")

    return result


def detect_video(model_path, source, conf=0.5, save=True):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("无法打开视频")
        return

    out = None  # ← 加这一行！防止 save=False 时未定义
    if save:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_detected.mp4', fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        cv2.imshow('YOLOv8 Detection', annotated)

        if save and out is not None:  # ← 加判断
            out.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out is not None:  # ← 加判断
        out.release()
    cv2.destroyAllWindows()
    print(f"视频处理完成，共 {frame_count} 帧")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assets/test.jpg', help='输入图片或视频')
    parser.add_argument('--weights', type=str, default='yolov8s.pt', help='模型权重')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--save', action='store_true', help='保存结果')
    args = parser.parse_args()

    # 自动判断图片还是视频
    ext = Path(args.source).suffix.lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        detect_image(args.weights, args.source, args.conf, args.save)
    else:
        detect_video(args.weights, args.source, args.conf, args.save)