# 🚗 自动驾驶 2D 感知与跟踪系统

基于 **YOLOv8** 的实时车辆/行人/交通标志检测，支持 **DeepSORT** 多目标跟踪与 **BEV** 鸟瞰图视角转换。

> **状态**：活跃开发中（WIP）  
> **目标**：暑假实习前完成端到端感知 Pipeline，支持视频流实时推理

---

## 🛠️ 技术栈

| 模块 | 技术 | 状态 |
|------|------|------|
| 目标检测 | YOLOv8 (Ultralytics) | ✅ 预训练推理跑通 |
| 数据集 | BDD100K | ✅ 已准备 |
| 多目标跟踪 | DeepSORT | ✅ 已集成 |
| BEV 转换 | IPM (Inverse Perspective Mapping) | ✅ 已集成 |
| 推理加速 | TensorRT / ONNX | ⏳ 待优化 |
| 硬件 | NVIDIA RTX 4070 Laptop GPU | ✅ |

---

## 📊 量化指标（目标 vs 当前）

| 指标 | 目标值 | 当前值 | 备注 |
|------|--------|--------|------|
| mAP@50 (BDD100K) | > 0.80 | **0.676** | YOLOv8s，50 epochs |
| 推理速度 (RTX 4070) | > 60 FPS | WIP | batch=1, FP16 |
| 跟踪 IDF1 Score | > 0.70 | ⏳ | DeepSORT |
| BEV 车道线平行度误差 | < 5° | ⏳ | IPM 标定 |

---

## 🚀 快速开始

### 环境要求
- Python 3.9+
- PyTorch 2.x + CUDA 11.8
- NVIDIA GPU（显存 ≥ 8GB）

### 安装

```bash
git clone https://github.com/Nefelibata134/autonomous-perception-yolo.git
cd autonomous-perception-yolo
pip install -r requirements.txt
```

### 单张图推理

```bash
python inference.py --source assets/test.jpg --weights yolov8s.pt --save
```

### 视频推理

```bash
python inference.py --source assets/test_video.mp4 --weights yolov8s.pt --save
```

---

## 📁 项目结构

```
autonomous-perception-yolo/
├── assets/
    ├──demo_detection.jpg
    ├──demo_yolo_labels.jpg
    ├──test.jpg
    ├──confusion_matrix.png
    ├──metrics_pr_curve.png
    ├──training_results.png
    ├──demo_tracking.mp4
                          # 示例图片/视频（不上传大文件）
├── configs/
    ├──data.ymal             # 训练配置文件
├── data/
    ├──bdd100k/
        ├──images/
            ├──10k/
                ├──test/
                ├──train/
                ├──val/
            ├──100k/
                ├──test/
                ├──train/
                ├──val/
        ├──labels/
            ├──100k/
                ├──train/
                ├──val/
            ├──bdd100k_labels_images_train.json
            ├──bdd100k_labels_images_val.json          # 数据集（BDD100K，.gitignore）
├── models/
│   ├── yolo_detector.py     # YOLOv8 封装类
│   ├── deepsort_tracker.py  # DeepSORT 跟踪器
│   └── bev_transform.py     # BEV 视角转换
├── utils/
│   ├── dataset_converter.py # BDD100K → YOLO 格式转换
│   └── visualizer_yolo.py        # 可视化工具
├── inference.py             # 推理入口
├── train.py                 # 训练脚本
├── eval.py                  # 评估脚本（mAP计算）
├── requirements.txt
├──inference_tracjing.py
└── README.md

```

---

## 🎬 演示

### 检测示例（预训练模型推理）
![检测示例](assets/demo_detection.jpg)

### 跟踪示例（DeepSORT 多目标跟踪）
![跟踪示例](assets/demo_tracking.jpg)

### BEV 鸟瞰图（检测+跟踪+IPM）
![BEV示例](assets/demo_bev.jpg)

---

## 📝 开发日志

| 日期         | 里程碑       | 完成内容 |
|------------|-----------|----------|
| 2026-04-27 | WIP Day 1 | YOLOv8 环境搭建，单张图/视频推理跑通 |
| 2026-04-29 | WIP Day 2 | 数据集准备：BDD100K 下载、筛选、格式转换 |
| 2026-05-01 | WIP Day 3 | YOLOv8s 全量训练完成，BDD100K mAP@50=0.676（RTX 5090 云训练 3h） |
| 2026-05-02 | WIP Day 4 | DeepSORT 多目标跟踪集成 |
| 2026-05-03 | WIP Day 5 | BEV 鸟瞰图转换（IPM），前视图+BEV 并排可视化 |
| TBD        | WIP Day 6 | TensorRT / ONNX 导出与推理加速 |
| TBD        | WIP Day 7 | 项目收尾：README 完善，Demo 视频录制，简历包装 |

---

## 📚 参考资料

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
- [DeepSORT Paper](https://arxiv.org/abs/1703.07402)
- [IPM (Inverse Perspective Mapping)](https://en.wikipedia.org/wiki/Inverse_perspective_mapping)

---

## 📧 联系

如有问题或建议，欢迎提 [Issue](https://github.com/Nefelibata134/autonomous-perception-yolo/issues) 或联系作者。
