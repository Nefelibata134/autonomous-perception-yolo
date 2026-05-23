"""
Measure VRAM usage for YOLOv8m + DeepSORT (ReID)
"""
import torch
import time
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

MODEL_PT = "/home/nefelibata/autonomous_driving/autonomous-perception-yolo/runs/detect/train_v8m_optimized/weights/best.pt"
MODEL_ENGINE = "/home/nefelibata/autonomous_driving/autonomous-perception-yolo/runs/detect/train/weights/best_local.engine"

# Create a realistic dummy frame (640x640 BGR)
dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

def get_vram():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024**3), torch.cuda.max_memory_allocated() / (1024**3)

# Reset peak tracking
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
time.sleep(0.5)

baseline, _ = get_vram()
print(f"Baseline VRAM: {baseline:.3f} GB")

# Load YOLOv8m PyTorch
print("\n1. Loading YOLOv8m (PyTorch FP32)...")
model = YOLO(MODEL_PT)
vram_yolo, vram_yolo_peak = get_vram()
print(f"   VRAM after YOLOv8m load: {vram_yolo:.3f} GB (peak: {vram_yolo_peak:.3f} GB)")

# Single inference warmup
_ = model(dummy_img, verbose=False)
torch.cuda.synchronize()
vram_infer_pt, vram_infer_pt_peak = get_vram()
print(f"   VRAM during PyTorch inference: {vram_infer_pt:.3f} GB (peak: {vram_infer_pt_peak:.3f} GB)")

# Initialize DeepSORT (loads ReID model)
print("\n2. Loading DeepSORT (MobileNetV2 ReID)...")
deepsort = DeepSort(max_age=30, n_init=3)
# Warmup: format = ([x, y, w, h], confidence, class_name)
dets = [([100, 100, 100, 100], 0.9, '0')]
_ = deepsort.update_tracks(dets, frame=dummy_img)
torch.cuda.synchronize()
vram_ds, vram_ds_peak = get_vram()
print(f"   VRAM after DeepSORT init: {vram_ds:.3f} GB (peak: {vram_ds_peak:.3f} GB)")

# Run full pipeline (detection + tracking) for a few frames
print("\n3. Running YOLOv8m + DeepSORT pipeline (10 frames)...")
torch.cuda.reset_peak_memory_stats()
for i in range(10):
    results = model(dummy_img, verbose=False)
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        dets = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if cls_id in [0, 1, 2, 3]:
                dets.append((x1, y1, x2, y2, conf, cls_id))
        if dets:
            # Convert to DeepSORT format
            ds_dets = [([x1, y1, x2-x1, y2-y1], conf, str(cls_id)) for x1, y1, x2, y2, conf, cls_id in dets]
            _ = deepsort.update_tracks(ds_dets, frame=dummy_img)
torch.cuda.synchronize()
vram_full, vram_full_peak = get_vram()

print(f"   VRAM during full pipeline: {vram_full:.3f} GB (peak: {vram_full_peak:.3f} GB)")

# Load TensorRT engine
print("\n4. Loading YOLOv8m TensorRT FP16...")
model_trt = YOLO(MODEL_ENGINE)
_ = model_trt(dummy_img, verbose=False)
torch.cuda.synchronize()
vram_trt, vram_trt_peak = get_vram()
print(f"   VRAM after TensorRT load: {vram_trt:.3f} GB (peak: {vram_trt_peak:.3f} GB)")

# Run TensorRT pipeline
print("\n5. Running TensorRT YOLOv8m + DeepSORT pipeline (10 frames)...")
torch.cuda.reset_peak_memory_stats()
for i in range(10):
    results = model_trt(dummy_img, verbose=False)
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        dets = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if cls_id in [0, 1, 2, 3]:
                dets.append((x1, y1, x2, y2, conf, cls_id))
        if dets:
            ds_dets = [([x1, y1, x2-x1, y2-y1], conf, str(cls_id)) for x1, y1, x2, y2, conf, cls_id in dets]
            _ = deepsort.update_tracks(ds_dets, frame=dummy_img)
torch.cuda.synchronize()
vram_trt_full, vram_trt_full_peak = get_vram()

print(f"   VRAM during TRT pipeline: {vram_trt_full:.3f} GB (peak: {vram_trt_full_peak:.3f} GB)")

# Summary
print(f"\n{'='*60}")
print("VRAM Summary (RTX 4070 Laptop 8GB)")
print(f"{'='*60}")
print(f"  YOLOv8m (PyTorch) alone:        {vram_infer_pt:.3f} GB")
print(f"  YOLOv8m + DeepSORT (PyTorch):   {vram_full:.3f} GB")
print(f"  YOLOv8m + DeepSORT (TensorRT):  {vram_trt_full:.3f} GB")
print(f"  GPU total:                      8.00 GB")
print(f"{'='*60}")

# Print YOLOv8m model size and param count
pt_size = os.path.getsize(MODEL_PT)
engine_size = os.path.getsize(MODEL_ENGINE)
reid_engine_size = os.path.getsize("/home/nefelibata/autonomous_driving/autonomous-perception-yolo/runs/reid_mobilenetv2.engine")

print(f"\nModel Sizes:")
print(f"  YOLOv8m .pt:             {pt_size/1024/1024:.1f} MB")
print(f"  YOLOv8m .engine (FP16):  {engine_size/1024/1024:.1f} MB")
print(f"  ReID .engine (FP16):     {reid_engine_size/1024/1024:.1f} MB")
print(f"  Total engines:           {(engine_size+reid_engine_size)/1024/1024:.1f} MB")

# YOLOv8m param count
params = sum(p.numel() for p in model.model.parameters())
print(f"\n  YOLOv8m parameters:      {params/1e6:.1f}M")
