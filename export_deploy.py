"""
export_deploy.py
模型部署导出：PyTorch → ONNX → TensorRT Engine
"""
from ultralytics import YOLO
from pathlib import Path


def export_models():
    """
    导出三种格式，用于部署对比
    """
    weights = 'runs/detect/train/weights/best.pt'

    if not Path(weights).exists():
        print(f"❌ 权重不存在: {weights}")
        return

    print("🚀 加载模型...")
    model = YOLO(weights)

    # ========== 1. 导出 ONNX ==========
    # 通用性强，可在 CPU/GPU/嵌入式设备运行
    print("\n📦 导出 ONNX (FP32)...")
    try:
        model.export(
            format='onnx',
            imgsz=640,
            half=False,  # FP32
            simplify=True,  # 简化计算图
            opset=12,
        )
        print("✅ ONNX 导出成功: runs/detect/train/weights/best.onnx")
    except Exception as e:
        print(f"⚠️ ONNX 导出失败: {e}")

    # ========== 2. 导出 TensorRT Engine ==========
    # 速度最快，NVIDIA GPU 专用（RTX 4070/5090）
    # 自动做算子融合、FP16 量化、内存优化
    print("\n🔥 导出 TensorRT Engine (FP16)...")
    try:
        model.export(
            format='engine',
            imgsz=640,
            half=True,  # FP16 半精度，速度翻倍
            device=0,
            workspace=4,  # 构建时工作区大小(GB)
            verbose=False,
        )
        print("✅ TensorRT 导出成功: runs/detect/train/weights/best.engine")
    except Exception as e:
        print(f"⚠️ TensorRT 导出失败（需安装 tensorrt）: {e}")
        print("💡 提示：先确保 pip install tensorrt 成功，或从 NVIDIA 官网下载")

    print("\n📊 导出完成，接下来运行 benchmark.py 对比速度")


if __name__ == '__main__':
    export_models()