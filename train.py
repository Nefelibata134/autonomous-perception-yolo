"""
train.py
基于 YOLOv8 在 BDD100K 4 类上微调训练
"""
from ultralytics import YOLO


def main():
    # 1. 加载预训练模型（自动下载 yolov8s.pt）
    # 选择 yolov8s 的原因：速度精度平衡，4070 8GB 显存能跑
    model = YOLO('yolov8s.pt')

    # 2. 开始训练
    # 关键参数说明（面试必问）：
    # - data: 数据集配置文件路径
    # - epochs: 训练轮数（BDD100K 子集 50 轮足够，全量建议 100）
    # - batch: 批次大小（4070 8GB 建议 8，如果爆显存改 4）
    # - imgsz: 输入分辨率（640 是 YOLOv8 默认，自动驾驶场景够用了）
    # - device: 0 表示第一张 GPU
    # - workers: 数据加载线程数（WSL 建议 4，Windows 建议 0）
    # - patience: 早停耐心值（10 轮不提升就停，防过拟合）
    # - save_period: 每 10 轮保存一个 checkpoint（防中断）
    results = model.train(
        data='configs/data.yaml',
        epochs=50,
        batch=8,
        imgsz=640,
        device=0,
        workers=4,
        patience=10,
        save_period=10,
        project='runs/detect',
        name='train',
        exist_ok=True,  # 覆盖已有目录
        pretrained=True,
        optimizer='AdamW',  # 比 SGD 收敛快，适合微调
        lr0=0.001,  # 初始学习率
        lrf=0.01,  # 最终学习率 = lr0 * lrf
        augment=True,  # 启用 Mosaic 等增强
        seed=42,  # 保证可复现
    )

    print("\n🎉 训练完成！")
    print(f"最佳权重: runs/detect/train/weights/best.pt")
    print(f"最后权重: runs/detect/train/weights/last.pt")


if __name__ == '__main__':
    main()