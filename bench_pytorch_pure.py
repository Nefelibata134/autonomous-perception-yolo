"""
公平对比：YOLOv8m PyTorch FP32 vs TensorRT FP16 on RTX 4070
使用相同的 benchmark 条件（纯模型推理，1000 次）
"""
import torch
import numpy as np
import time

# Load PyTorch model
from ultralytics import YOLO

model = YOLO("/home/nefelibata/autonomous_driving/autonomous-perception-yolo/runs/detect/train_v8m_optimized/weights/best.pt")
model.model.eval()
model.model.cuda()

# Create dummy input
dummy = torch.randn(1, 3, 640, 640, device='cuda')

# Warmup
print("PyTorch FP32 warmup (50 iters)...")
for _ in range(50):
    with torch.no_grad():
        _ = model.model(dummy)
torch.cuda.synchronize()

# Benchmark
print("PyTorch FP32 benchmark (1000 iters)...")
timings = []
for _ in range(1000):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.model(dummy)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    timings.append((t1 - t0) * 1000)

timings = np.array(timings)
avg_ms = np.mean(timings)
fps = 1000.0 / avg_ms

print(f"\n{'='*60}")
print(f"YOLOv8m PyTorch FP32 (pure model forward pass, RTX 4070)")
print(f"{'='*60}")
print(f"  Avg latency: {avg_ms:.2f} ms")
print(f"  Throughput:  {fps:.1f} FPS")
print(f"  Min: {np.min(timings):.2f} ms, Max: {np.max(timings):.2f} ms")
print(f"{'='*60}")

# Summary
trt_fps = 618.0  # from our TensorRT benchmark
speedup = trt_fps / fps
print(f"\n{'='*60}")
print(f"ACCELERATION SUMMARY (RTX 4070 Laptop)")
print(f"{'='*60}")
print(f"  PyTorch FP32:  {fps:.1f} FPS")
print(f"  TensorRT FP16: {trt_fps:.1f} FPS")
print(f"  Speedup:       {speedup:.1f}×")
print(f"{'='*60}")
