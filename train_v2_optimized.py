"""
train_v2_optimized.py
YOLOv8m + 100 epochs + 强数据增强（5090 专用）
目标：BDD100K mAP@50 > 0.75
"""
from ultralytics import YOLO


def main():
    # 升级模型：s → m（参数量翻倍，特征提取能力显著增强）
    model = YOLO('yolov8m.pt')

    print("🚀 启动优化训练：YOLOv8m + 100 epochs + 强增强")

    results = model.train(
        data='configs/data_autodl.yaml',  # 云上路径
        epochs=100,  # 从 50 → 100，充分收敛
        batch=16,  # 5090 32GB 显存，m 模型 batch=16 安全
        imgsz=640,
        device=0,
        workers=16,
        patience=20,  # 早停更宽松，给足 100 epoch 机会
        save_period=10,
        project='runs/detect',
        name='train_v8m_optimized',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,

        # ========== 数据增强优化（核心！）==========
        augment=True,
        mosaic=1.0,  # Mosaic 增强（默认）
        mixup=0.1,  # MixUp 混合增强（新增，提升泛化）
        hsv_h=0.015,  # 色调变化幅度
        hsv_s=0.7,  # 饱和度变化
        hsv_v=0.4,  # 亮度变化
        degrees=5,  # 随机旋转 ±5 度
        translate=0.1,  # 随机平移 10%
        scale=0.5,  # 随机缩放 50%-150%
        shear=2,  # 随机错切 ±2 度
        flipud=0.0,  # 不上下翻转（街景不适用）
        fliplr=0.5,  # 左右翻转 50%
        copy_paste=0.0,  # 关闭 copy-paste（BDD100K 不需要）

        seed=42,
    )

    print("\n🎉 优化训练完成！")
    print("预期 mAP@50: 0.75+")
    print(f"权重: runs/detect/train_v8m_optimized/weights/best.pt")


if __name__ == '__main__':
    main()