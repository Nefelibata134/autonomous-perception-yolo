"""
Get accurate VRAM using nvidia-smi during full pipeline
"""
import torch
import time
import numpy as np
import subprocess
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

MODEL_PT = "/home/nefelibata/autonomous_driving/autonomous-perception-yolo/runs/detect/train_v8m_optimized/weights/best.pt"
MODEL_ENGINE = "/home/nefelibata/autonomous_driving/autonomous-perception-yolo/runs/detect/train/weights/best_local.engine"

dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

def smi():
    out = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
    used, total = out.strip().split(", ")
    return float(used)/1024, float(total)/1024

print(f"GPU: RTX 4070 Laptop 8GB")
used, total = smi()
print(f"Idle VRAM: {used:.2f}/{total:.2f} GB")

# Load YOLOv8m PyTorch
print("\nLoading YOLOv8m (PyTorch FP32)...")
model = YOLO(MODEL_PT)
_ = model(dummy_img, verbose=False)
used, _ = smi()
print(f"VRAM after YOLOv8m load + 1 inference: {used:.2f} GB")
vram_yolo = used

# Load DeepSORT with ReID
print("\nLoading DeepSORT + ReID...")
deepsort = DeepSort(max_age=30, n_init=3)
# Convert format and warmup
dets_yolo = model(dummy_img, verbose=False)[0].boxes
if dets_yolo and len(dets_yolo) > 0:
    box = dets_yolo[0]
    ds_dets = [([float(box.xyxy[0][0]), float(box.xyxy[0][1]), 
                 float(box.xyxy[0][2]-box.xyxy[0][0]), float(box.xyxy[0][3]-box.xyxy[0][1])], 
                float(box.conf[0]), str(int(box.cls[0])))]
    _ = deepsort.update_tracks(ds_dets, frame=dummy_img)
used, _ = smi()
print(f"VRAM after YOLOv8m + DeepSORT: {used:.2f} GB")
vram_full_pt = used

# Run 50 frames to stabilize
print("\nRunning 50 frames (PyTorch pipeline)...")
for _ in range(50):
    results = model(dummy_img, verbose=False)
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        ds_dets = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if cls_id < 4:
                ds_dets.append(([x1, y1, x2-x1, y2-y1], conf, str(cls_id)))
        if ds_dets:
            _ = deepsort.update_tracks(ds_dets, frame=dummy_img)
torch.cuda.synchronize()
time.sleep(1)
used, _ = smi()
print(f"VRAM during steady state (PyTorch): {used:.2f} GB")
vram_steady_pt = used

# Now TensorRT
print("\nSwitching to TensorRT...")
del model
torch.cuda.empty_cache()
time.sleep(1)
used, _ = smi()
print(f"VRAM after cleanup: {used:.2f} GB")

model_trt = YOLO(MODEL_ENGINE)
_ = model_trt(dummy_img, verbose=False)
used, _ = smi()
print(f"VRAM after TensorRT YOLOv8m load: {used:.2f} GB")
vram_trt = used

print("\nRunning 50 frames (TensorRT pipeline)...")
for _ in range(50):
    results = model_trt(dummy_img, verbose=False)
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        ds_dets = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if cls_id < 4:
                ds_dets.append(([x1, y1, x2-x1, y2-y1], conf, str(cls_id)))
        if ds_dets:
            _ = deepsort.update_tracks(ds_dets, frame=dummy_img)
torch.cuda.synchronize()
time.sleep(1)
used, _ = smi()
print(f"VRAM during steady state (TensorRT): {used:.2f} GB")
vram_steady_trt = used

print(f"\n{'='*60}")
print("VRAM USAGE SUMMARY")
print(f"{'='*60}")
print(f"  YOLOv8m (PyTorch) alone:        {vram_yolo:.1f} GB")
print(f"  YOLOv8m + DeepSORT (PyTorch):   {vram_steady_pt:.1f} GB")
print(f"  YOLOv8m + DeepSORT (TensorRT):  {vram_steady_trt:.1f} GB")
print(f"  GPU total:                      8.00 GB")
print(f"{'='*60}")
