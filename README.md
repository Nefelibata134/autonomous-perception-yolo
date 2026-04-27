# 🚗 自动驾驶 2D 感知与跟踪系统

基于 **YOLOv8** 的实时车辆/行人/交通标志检测，支持 **DeepSORT** 多目标跟踪与 **BEV** 鸟瞰图视角转换。

> **状态**：活跃开发中（WIP）  
> **目标**：暑假实习前完成端到端感知 Pipeline，支持视频流实时推理

---

## 🛠️ 技术栈

| 模块 | 技术 | 状态 |
|------|------|------|
| 目标检测 | YOLOv8 (Ultralytics) | ✅ 预训练推理跑通 |
| 数据集 | BDD100K | 🔄 准备中 |
| 多目标跟踪 | DeepSORT / ByteTrack | ⏳ 待集成 |
| BEV 转换 | IPM (Inverse Perspective Mapping) | ⏳ 待集成 |
| 推理加速 | TensorRT / ONNX | ⏳ 待优化 |
| 硬件 | NVIDIA RTX 4070 Laptop GPU | ✅ |

---

## 📊 量化指标（目标 vs 当前）

| 指标 | 目标值 | 当前值 | 备注 |
|------|--------|--------|------|
| mAP@50 (BDD100K) | > 0.80 | WIP | 车辆/行人/交通标志 |
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
├── assets/                  # 示例图片/视频（不上传大文件）
├── configs/                 # 训练配置文件
├── data/                    # 数据集（BDD100K，.gitignore）
├── models/
│   ├── yolo_detector.py     # YOLOv8 封装类
│   ├── deepsort_tracker.py  # DeepSORT 跟踪器
│   └── bev_transform.py     # BEV 视角转换
├── utils/
│   ├── dataset_converter.py # BDD100K → YOLO 格式转换
│   └── visualizer.py        # 可视化工具
├── inference.py             # 推理入口
├── train.py                 # 训练脚本
├── eval.py                  # 评估脚本（mAP计算）
├── requirements.txt
└── README.md
```

---

## 🎬 演示

### 检测示例（预训练模型推理）
![检测示例](https://github.com/Nefelibata134/autonomous-perception-yolo/blob/main/assets/demo_detection.jpg)

### 跟踪示例（WIP）
待添加

---

## 📝 开发日志

| 日期 | 里程碑 | 完成内容 |
|------|--------|----------|
| 2026-04-27 | WIP Day 1 | YOLOv8 环境搭建，单张图/视频推理跑通 |
| TBD | WIP Day 2 | 数据集准备：BDD100K 下载、筛选、格式转换 |
| TBD | WIP Day 3 | 训练自定义检测器（车辆/行人/交通标志） |
| TBD | WIP Day 4 | 评估优化：mAP 计算，TensorBoard 可视化 |
| TBD | WIP Day 5 | DeepSORT 多目标跟踪集成 |
| TBD | WIP Day 6 | IPM 鸟瞰图转换，车道线检测 |
| TBD | WIP Day 7 | TensorRT / ONNX 导出与推理加速 |
| TBD | WIP Day 8 | 项目收尾：README 完善，Demo 视频录制，简历包装 |

---

## 📚 参考资料

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
- [DeepSORT Paper](https://arxiv.org/abs/1703.07402)
- [IPM (Inverse Perspective Mapping)](https://en.wikipedia.org/wiki/Inverse_perspective_mapping)

---

## 📧 联系

如有问题或建议，欢迎提 [Issue](https://github.com/Nefelibata134/autonomous-perception-yolo/issues) 或联系作者。
