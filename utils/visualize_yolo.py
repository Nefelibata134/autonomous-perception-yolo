"""
utils/visualize_yolo.py
随机抽取一张图，把 YOLO 标注画上去，人工确认转换正确
"""
import cv2
import random
from pathlib import Path

CLASS_NAMES = {
    0: 'car',
    1: 'person',
    2: 'traffic light',
    3: 'traffic sign'
}

COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]


def visualize(image_path, label_path, save_path=None):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return

    h, w = img.shape[:2]

    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            x_c, y_c, bw, bh = map(float, parts[1:])

            # 反归一化，转回像素坐标
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)

            color = COLORS[cls_id % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = CLASS_NAMES.get(cls_id, str(cls_id))
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if save_path:
            cv2.imwrite(str(save_path), img)
            print(f"✅ 验证图已保存: {save_path}")

        cv2.imshow('YOLO Label Verification', img)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"❌ 标注文件不存在: {label_path}")


if __name__ == "__main__":
    # 随机选一张有标注的图来验证
    img_dir = Path('data/bdd100k/images/100k/train')
    label_dir = Path('data/bdd100k/labels_yolo/train')

    label_files = list(label_dir.glob('*.txt'))
    if not label_files:
        print("❌ 没有找到标注文件，请先运行 dataset_converter.py")
        exit()

    sample_label = random.choice(label_files)
    sample_img = img_dir / f"{sample_label.stem}.jpg"

    if sample_img.exists():
        visualize(sample_img, sample_label, save_path='assets/verify_yolo.jpg')
    else:
        print(f"❌ 图片不存在: {sample_img}")