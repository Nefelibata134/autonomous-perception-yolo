"""
YOLOv8m 完整流程 benchmark：PyTorch FP32 vs TensorRT FP16
Ultralytics pipeline (preprocessing + inference + NMS + postprocessing)
"""
from ultralytics import YOLO
import torch
import numpy as np
import time
import cv2

MODEL_PT = "/home/nefelibata/autonomous_driving/autonomous-perception-yolo/runs/detect/train_v8m_optimized/weights/best.pt"
MODEL_ENGINE = "/home/nefelibata/autonomous_driving/autonomous-perception-yolo/runs/detect/train/weights/best_local.engine"

# Create dummy 640x640 image
dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

def bench(model_path, label, iters=500):
    model = YOLO(model_path)
    # Warmup
    for _ in range(30):
        _ = model(dummy_img, verbose=False)
    
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model(dummy_img, verbose=False)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)
    
    timings = np.array(timings)
    avg_ms = np.mean(timings)
    fps = 1000.0 / avg_ms
    
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"  Avg latency:  {avg_ms:.2f} ms")
    print(f"  Throughput:   {fps:.1f} FPS")
    print(f"  Std:          {np.std(timings):.2f} ms")
    print(f"  Min/Max:      {np.min(timings):.2f} / {np.max(timings):.2f} ms")
    return fps

print("=" * 60)
print("YOLOv8m Full Pipeline Benchmark (preprocess + infer + NMS)")
print("GPU: NVIDIA GeForce RTX 4070 Laptop (8GB)")
print("=" * 60)

# PyTorch
fps_pt = bench(MODEL_PT, "PyTorch FP32", iters=500)

# TensorRT  
fps_trt = bench(MODEL_ENGINE, "TensorRT FP16", iters=500)

speedup = fps_trt / fps_pt
print(f"\n{'='*60}")
print(f"ACCELERATION: TensorRT FP16 vs PyTorch FP32")
print(f"{'='*60}")
print(f"  PyTorch FP32:  {fps_pt:.1f} FPS")
print(f"  TensorRT FP16: {fps_trt:.1f} FPS")
print(f"  Speedup:       {speedup:.1f}×")
print(f"{'='*60}")
