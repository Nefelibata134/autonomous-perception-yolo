"""
Export DeepSORT ReID model (MobileNetV2) to TensorRT and benchmark
"""
import torch
import numpy as np
import time
import os
from deep_sort_realtime.embedder.mobilenetv2_bottle import MobileNetV2_bottle
import pkg_resources

MODEL_WTS = pkg_resources.resource_filename("deep_sort_realtime", "embedder/weights/mobilenetv2_bottleneck_wts.pt")
OUT_DIR = "/home/nefelibata/autonomous_driving/autonomous-perception-yolo/runs/"
ONNX_PATH = os.path.join(OUT_DIR, "reid_mobilenetv2.onnx")
ENGINE_PATH = os.path.join(OUT_DIR, "reid_mobilenetv2.engine")

# Load model
model = MobileNetV2_bottle(input_size=224, width_mult=1.0)
model.load_state_dict(torch.load(MODEL_WTS))
model.eval()
model.cuda()

# Export to ONNX (fixed batch=1, easier for TRT)
dummy = torch.randn(1, 3, 224, 224, device='cuda')
print("Exporting to ONNX...")
torch.onnx.export(model, dummy, ONNX_PATH, input_names=['input'], output_names=['output'],
                  opset_version=17)
print(f"ONNX saved: {ONNX_PATH} ({os.path.getsize(ONNX_PATH)/1024/1024:.1f} MB)")

# Build TensorRT engine
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(ONNX_PATH, "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
config.set_flag(trt.BuilderFlag.FP16)

print("Building TensorRT engine (FP16)...")
t0 = time.time()
engine_data = builder.build_serialized_network(network, config)
engine_bytes = bytes(engine_data)
build_time = time.time() - t0

with open(ENGINE_PATH, "wb") as f:
    f.write(engine_bytes)
print(f"Engine saved: {ENGINE_PATH} ({len(engine_bytes)/1024/1024:.1f} MB, {build_time:.1f}s)")

# Benchmark PyTorch FP32
print("\nBenchmarking PyTorch FP32...")
dummy = torch.randn(1, 3, 224, 224, device='cuda')
for _ in range(30):
    _ = model(dummy)
torch.cuda.synchronize()

times_pt = []
for _ in range(500):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model(dummy)
    torch.cuda.synchronize()
    times_pt.append((time.perf_counter() - t0) * 1000)

pt_avg = np.mean(times_pt)
pt_fps = 1000 / pt_avg
print(f"  PyTorch FP32: {pt_avg:.2f} ms, {pt_fps:.0f} FPS")

# Benchmark TensorRT FP16
print("Benchmarking TensorRT FP16...")
runtime = trt.Runtime(logger)
with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)
context.set_input_shape(input_name, (1, 3, 224, 224))

d_in = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device='cuda')
d_out = torch.zeros(1, 1280, dtype=torch.float32, device='cuda')
context.set_tensor_address(input_name, d_in.data_ptr())
context.set_tensor_address(output_name, d_out.data_ptr())

for _ in range(30):
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()

times_trt = []
for _ in range(500):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    times_trt.append((time.perf_counter() - t0) * 1000)

trt_avg = np.mean(times_trt)
trt_fps = 1000 / trt_avg
speedup = pt_fps / trt_fps  # wait - speedup = trt/pt
speedup_correct = trt_fps / (1000/pt_avg)  # this is wrong too
speedup2 = (1000/trt_avg) / (1000/pt_avg)

print(f"  TensorRT FP16: {trt_avg:.2f} ms, {trt_fps:.0f} FPS")

print(f"\n{'='*60}")
print("ReID Model (MobileNetV2) Optimization Summary")
print(f"{'='*60}")
print(f"  PyTorch FP32:  {pt_avg:.2f} ms")
print(f"  TensorRT FP16: {trt_avg:.2f} ms")
print(f"  Speedup:       {pt_avg/trt_avg:.1f}×")
print(f"  Model size:    {len(engine_bytes)/1024/1024:.1f} MB")
