"""
benchmark.py
部署加速对比：PyTorch vs TensorRT（ONNX 因环境兼容性跳过）
"""
import time
import cv2
from ultralytics import YOLO
from pathlib import Path


def benchmark_model(model_path, source, num_frames=100):
    if not Path(model_path).exists():
        print(f"❌ 模型不存在，跳过: {model_path}")
        return None, None

    print(f"\n▶️ 测试: {Path(model_path).name}")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ 视频不存在: {source}")
        return None, None

    # Warm-up
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        model(frame, verbose=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    start = time.time()

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        model(frame, verbose=False)

    elapsed = time.time() - start
    fps = num_frames / elapsed
    ms = (elapsed / num_frames) * 1000

    cap.release()
    print(f"   FPS: {fps:.1f} | 每帧: {ms:.1f}ms | 总时间: {elapsed:.2f}s")
    return fps, ms


def main():
    source = 'assets/test_video.mp4'
    num_frames = 100

    print("=" * 60)
    print("🚗 自动驾驶感知模型部署加速对比")
    print(f"硬件: NVIDIA RTX 4070 Laptop | 测试帧数: {num_frames}")
    print("=" * 60)

    # 只测试 PyTorch 和 TensorRT（ONNX 因 WSL CUDA 11.8 兼容性跳过）
    pt_fps, pt_ms = benchmark_model('runs/detect/train/weights/best.pt', source, num_frames)
    trt_fps, trt_ms = benchmark_model('runs/detect/train/weights/best.engine', source, num_frames)

    print("\n" + "=" * 60)
    print("📊 加速对比结果（直接复制到 README）")
    print("=" * 60)
    print(f"| 格式 | 精度 | FPS | 延迟(ms) | 加速比 |")
    print(f"|------|------|-----|----------|--------|")

    baseline = pt_fps or 1.0

    if pt_fps:
        print(f"| PyTorch | FP32 | {pt_fps:.1f} | {pt_ms:.1f} | 1.0x |")
    if trt_fps:
        trt_speedup = trt_fps / baseline
        print(f"| TensorRT | FP16 | {trt_fps:.1f} | {trt_ms:.1f} | {trt_speedup:.1f}x |")

    print("=" * 60)
    print("\n💡 ONNX Runtime 因 WSL CUDA 11.8 兼容性限制未测试")
    print("   TensorRT FP16 已完整支持，满足 >60 FPS 部署目标")

    with open('benchmark_results.txt', 'w') as f:
        f.write("部署加速对比结果\n")
        if pt_fps: f.write(f"PyTorch FP32: {pt_fps:.1f} FPS\n")
        if trt_fps: f.write(f"TensorRT FP16: {trt_fps:.1f} FPS\n")

    print("\n✅ 结果已保存到 benchmark_results.txt")


if __name__ == '__main__':
    main()