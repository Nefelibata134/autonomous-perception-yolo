"""
utils/dataset_converter.py
BDD100K JSON 标注 → YOLO 格式 (class x_center y_center width height)
只保留自动驾驶相关的4类：car, person, traffic light, traffic sign
"""
import json
from pathlib import Path
from tqdm import tqdm
import cv2



# BDD100K 原始类别 → 我们的4类ID
CATEGORY_MAP = {
    'car': 0,
    'person': 1,
    'traffic light': 2,
    'traffic sign': 3,
}

def convert_bdd100k_to_yolo(bdd_json_path, img_dir, output_label_dir):
    """
    转换单个JSON文件（train或val）
    """
    bdd_json_path = Path(bdd_json_path)
    img_dir = Path(img_dir)
    output_label_dir = Path(output_label_dir)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取标注: {bdd_json_path}")
    with open(bdd_json_path, 'r') as f:
        annotations = json.load(f)

    converted = 0
    skipped = 0
    no_img = 0

    for ann in tqdm(annotations, desc=f"Converting {bdd_json_path.stem}"):
        img_name = ann['name']  # 例如: 0000f77c-6257be58.jpg
        img_path = img_dir / img_name

        # 必须从原图读取尺寸，才能正确归一化
        if not img_path.exists():
            no_img += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            no_img += 1
            continue

        img_h, img_w = img.shape[:2]
        labels = ann.get('labels', [])
        yolo_lines = []

        for label in labels:
            category = label.get('category', '')
            if category not in CATEGORY_MAP:
                continue  # 跳过我们不关心的类别（如train, rider等）

            box2d = label.get('box2d', {})
            x1 = float(box2d.get('x1', 0))
            y1 = float(box2d.get('y1', 0))
            x2 = float(box2d.get('x2', 0))
            y2 = float(box2d.get('y2', 0))

            # 计算 YOLO 格式（归一化到 0~1）
            x_center = ((x1 + x2) / 2.0) / img_w
            y_center = ((y1 + y2) / 2.0) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            # 边界保护，防止越界
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            cls_id = CATEGORY_MAP[category]
            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # 只有当这张图包含4类目标时，才写入txt
        if yolo_lines:
            out_txt = output_label_dir / f"{Path(img_name).stem}.txt"
            with open(out_txt, 'w') as f:
                f.write('\n'.join(yolo_lines))
            converted += 1
        else:
            skipped += 1

    print(f"\n✅ {bdd_json_path.stem} 转换完成:")
    print(f"   成功转换: {converted} 张（含4类目标）")
    print(f"   无目标跳过: {skipped} 张（4类中无目标）")
    print(f"   图片缺失: {no_img} 张（图片未下载或路径错误）")



if __name__ == "__main__":
    # 转换训练集
    convert_bdd100k_to_yolo(
        bdd_json_path='data/bdd100k/labels/bdd100k_labels_images_train.json',
        img_dir='data/bdd100k/images/100k/train',
        output_label_dir='data/bdd100k/labels_yolo/train'
    )

    # 转换验证集
    convert_bdd100k_to_yolo(
        bdd_json_path='data/bdd100k/labels/bdd100k_labels_images_val.json',
        img_dir='data/bdd100k/images/100k/val',
        output_label_dir='data/bdd100k/labels_yolo/val'
    )