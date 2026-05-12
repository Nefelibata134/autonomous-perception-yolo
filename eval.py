"""
eval.py
评估训练好的模型，输出 mAP@50、mAP@50:95、PR 曲线
"""
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path


def evaluate():
    # 加载最佳权重
    model = YOLO('runs/detect/train/weights/best.pt')

    # 在验证集上评估
    # 返回 metrics 对象，包含所有指标
    metrics = model.val(
        data='configs/data.yaml',
        batch=8,
        imgsz=640,
        device=0,
        split='val',  # 明确用验证集
        save_json=True,  # 保存 COCO 格式 JSON，方便后续分析
        project='runs/detect',
        name='val',
        exist_ok=True,
    )

    # 提取关键指标
    map50 = metrics.box.map50  # mAP@50
    map75 = metrics.box.map75  # mAP@75
    map50_95 = metrics.box.map  # mAP@50:95

    print("\n" + "=" * 50)
    print("📊 评估结果（BDD100K 验证集）")
    print("=" * 50)
    print(f"mAP@50:     {map50:.4f}  （目标 > 0.80）")
    print(f"mAP@75:     {map75:.4f}")
    print(f"mAP@50:95:  {map50_95:.4f}")
    print("=" * 50)

    # 保存指标到文件
    import os
    os.makedirs('runs/detect/val', exist_ok=True)
    with open('runs/detect/val/metrics_summary.txt', 'w') as f:
        f.write(f"mAP@50: {map50:.4f}\n")
        f.write(f"mAP@75: {map75:.4f}\n")
        f.write(f"mAP@50:95: {map50_95:.4f}\n")

    # 绘制 PR 曲线
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # 逐类 PR 曲线
        for i, cls_name in enumerate(['car', 'person', 'traffic light', 'traffic sign']):
            # metrics.box.precision: [类别, 置信度阈值]
            # metrics.box.recall: [类别, 置信度阈值]
            # 这里简化处理，直接保存 Ultralytics 生成的 PR 曲线图
            pass

        # 更简单的方法：直接复制 Ultralytics 生成的 PR 曲线
        pr_path = Path('runs/detect/val/PR_curve.png')
        if pr_path.exists():
            print(f"\n✅ PR 曲线已生成: {pr_path}")

        import shutil
        if pr_path.exists():
            shutil.copy(pr_path, 'assets/metrics_pr_curve.png')
            print("✅ PR 曲线已复制到 assets/metrics_pr_curve.png")

    except Exception as e:
        print(f"⚠️ 绘制图表时出错: {e}")


if __name__ == '__main__':
    evaluate()