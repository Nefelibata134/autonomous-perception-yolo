"""
YOLOv8m TensorRT FP16 Benchmark on local RTX 4070
Builds engine from ONNX, benchmarks inference speed.
"""
import tensorrt as trt
import numpy as np
import time
import os

ONNX_PATH = "/home/nefelibata/autonomous_driving/autonomous-perception-yolo/runs/detect/train/weights/best.onnx"
ENGINE_PATH = "/home/nefelibata/autonomous_driving/autonomous-perception-yolo/runs/detect/train/weights/best_local.engine"

def build_engine(onnx_path, engine_path, fp16=True):
    """Build TensorRT engine from ONNX"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    print(f"Loading ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX Parse Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX")
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled")
        else:
            print("WARNING: FP16 not supported, falling back to FP32")
    
    print(f"Building engine... (this may take 2-5 minutes)")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine")
    
    build_time = time.time() - t0
    engine_bytes = bytes(serialized_engine)  # TRT 10 returns IHostMemory
    engine_size_mb = len(engine_bytes) / (1024 * 1024)
    
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    
    print(f"Engine saved: {engine_path} ({engine_size_mb:.1f} MB, build time: {build_time:.1f}s)")
    return engine_path

def benchmark_engine(engine_path, num_warmup=50, num_iter=1000):
    """Benchmark TensorRT engine"""
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Get binding info
    input_name = engine.get_tensor_name(0)
    input_shape = engine.get_tensor_shape(input_name)
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    
    output_names = []
    output_shapes = []
    output_dtypes = []
    num_io = engine.num_io_tensors
    for i in range(num_io):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
            output_shapes.append(engine.get_tensor_shape(name))
            output_dtypes.append(trt.nptype(engine.get_tensor_dtype(name)))
    
    print(f"\nInput:  {input_name} shape={input_shape} dtype={input_dtype}")
    for i, name in enumerate(output_names):
        print(f"Output: {name} shape={output_shapes[i]} dtype={output_dtypes[i]}")
    
    # Allocate buffers
    dummy_input = np.random.randn(*input_shape).astype(input_dtype)
    
    # Set input shape (batch dimension may be -1)
    actual_input_shape = (1,) + tuple(input_shape[1:])  # batch=1
    context.set_input_shape(input_name, actual_input_shape)
    
    # Allocate
    d_input = [None]
    d_outputs = [[] for _ in output_names]
    h_outputs = [[] for _ in output_names]
    
    # Allocate with PyTorch CUDA
    import torch
    
    # Use PyTorch CUDA for allocation
    d_input_val = torch.zeros(actual_input_shape, dtype=torch.float32, device='cuda')
    d_output_vals = []
    for name, shape in zip(output_names, output_shapes):
        actual_shape = tuple(s if s > 0 else 1 for s in shape)
        d_output_vals.append(torch.zeros(actual_shape, dtype=torch.float32, device='cuda'))
    
    context.set_tensor_address(input_name, d_input_val.data_ptr())
    for name, val in zip(output_names, d_output_vals):
        context.set_tensor_address(name, val.data_ptr())
    
    # Warmup
    print(f"\nWarming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        d_input_val.copy_(torch.from_numpy(dummy_input).reshape(actual_input_shape))
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({num_iter} iterations)...")
    timings = []
    for _ in range(num_iter):
        d_input_val.copy_(torch.from_numpy(dummy_input).reshape(actual_input_shape))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)  # ms
    
    timings = np.array(timings)
    avg_ms = np.mean(timings)
    std_ms = np.std(timings)
    fps = 1000.0 / avg_ms
    
    print(f"\n{'='*60}")
    print(f"YOLOv8m TensorRT FP16 Benchmark Results (RTX 4070 Laptop)")
    print(f"{'='*60}")
    print(f"  Average latency: {avg_ms:.2f} ms")
    print(f"  Std latency:     {std_ms:.2f} ms")
    print(f"  Throughput:      {fps:.1f} FPS")
    print(f"  Min latency:     {np.min(timings):.2f} ms")
    print(f"  Max latency:     {np.max(timings):.2f} ms")
    print(f"  Engine size:     {os.path.getsize(engine_path)/1024/1024:.1f} MB")
    print(f"{'='*60}")
    
    return fps

if __name__ == "__main__":
    if not os.path.exists(ENGINE_PATH):
        build_engine(ONNX_PATH, ENGINE_PATH, fp16=True)
    else:
        print(f"Engine already exists: {ENGINE_PATH}")
    
    fps = benchmark_engine(ENGINE_PATH)
