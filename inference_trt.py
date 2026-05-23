"""
inference_trt.py
使用 TensorRT Engine 进行实时推理（部署版）
比 PyTorch 快 3-5 倍，适合车载边缘设备
"""
import cv2
import argparse
from ultralytics import YOLO
from pathlib import Path


def inference_trt(source, output='output_trt.mp4'):
    """
    TensorRT 实时推理（FP16 加速）
    """
    # 本地 RTX 4070 导出的 TensorRT engine（FP16，23.8 MB）
    # 如需其他 GPU 的 engine，先运行: python export_deploy.py
    engine_path = 'runs/detect/train/weights/best_local.engine'

    if not Path(engine_path).exists():
        print(f"❌ Engine 不存在: {engine_path}")
        print("请先运行: python export_deploy.py")
        return

    print(f"🚀 加载 TensorRT Engine: {engine_path}")
    model = YOLO(engine_path)

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (w, h))

    frame_count = 0
    total_time = 0

    print(f"▶️ 开始 TensorRT 推理: {source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # TensorRT 推理（FP16，速度极快）
        import time
        t0 = time.time()
        results = model(frame, verbose=False)
        t1 = time.time()

        total_time += (t1 - t0)
        frame_count += 1

        # 绘制结果
        annotated = results[0].plot()
        out.write(annotated)

        if frame_count % 30 == 0:
            avg_fps = frame_count / total_time
            print(f"   已处理 {frame_count} 帧 | 实时 FPS: {avg_fps:.1f}")

    cap.release()
    out.release()

    final_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\n✅ TensorRT 推理完成！")
    print(f"   输出: {output}")
    print(f"   平均 FPS: {final_fps:.1f}")
    print(f"   总帧数: {frame_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assets/real_street.mp4')
    parser.add_argument('--output', type=str, default='output_trt.mp4')
    args = parser.parse_args()

    inference_trt(args.source, args.output)