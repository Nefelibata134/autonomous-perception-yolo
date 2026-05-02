"""
inference_tracking.py
视频流实时检测 + DeepSORT 跟踪
输出：带跟踪 ID 的 Demo 视频（简历展示核心素材）
"""
import cv2
import argparse
from pathlib import Path
from models.yolo_detector import YOLODetector
from models.deepsort_tracker import DeepsortTracker


def track_video(source, output='demo_tracking.mp4', show=False):
    """
    对视频进行目标检测 + 多目标跟踪

    Args:
        source: 输入视频路径（或摄像头索引 0）
        output: 输出视频路径
        show: 是否实时显示（WSL 下 cv2.imshow 可能有问题，建议 False）
    """
    # 初始化模块
    print("🚀 初始化检测器...")
    detector = YOLODetector(
        weights='runs/detect/train/weights/best.pt',
        conf=0.5,
        device=0
    )

    print("🚀 初始化 DeepSORT 跟踪器...")
    tracker = DeepsortTracker(max_age=30, n_init=3)

    # 打开视频
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {source}")
        return

    # 视频属性
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (w, h))

    frame_count = 0
    id_switches = 0  # 统计 ID Switch 次数（面试谈资）
    prev_ids = set()

    print(f"▶️ 开始处理: {source}")
    print(f"   分辨率: {w}x{h}, FPS: {fps}, 总帧数: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 检测
        result = detector.detect(frame)
        detections = detector.get_detections(result)

        # 2. 跟踪
        tracks = tracker.update(detections, frame)

        # 3. 统计 ID Switch（简单版：上一帧有的 ID 这一帧没了，且不是 age out）
        current_ids = {t.track_id for t in tracks if t.is_confirmed()}
        disappeared = prev_ids - current_ids
        # 如果消失的不是因为 max_age（正常消失），可能是 ID Switch
        # 这里简化统计，仅作展示
        prev_ids = current_ids

        # 4. 绘制
        vis_frame = tracker.draw_tracks(frame.copy(), tracks)

        # 添加全局信息
        info_text = f"Frame: {frame_count} | Tracks: {len(current_ids)}"
        cv2.putText(vis_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 5. 写入 + 显示
        out.write(vis_frame)

        if show:
            cv2.imshow('DeepSORT Tracking', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ 用户中断")
                break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   已处理 {frame_count}/{total_frames} 帧，当前跟踪目标: {len(current_ids)}")

    # 释放
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n✅ 跟踪完成！")
    print(f"   输出视频: {output}")
    print(f"   总帧数: {frame_count}")
    print(f"   最终跟踪目标数: {len(current_ids)}")
    print(f"\n💡 提示：将 {output} 复制到 assets/demo_tracking.mp4 用于 README 展示")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='自动驾驶 2D 感知与跟踪系统')
    parser.add_argument('--source', type=str, default='assets/test_video.mp4',
                        help='输入视频路径（默认 assets/test_video.mp4）')
    parser.add_argument('--output', type=str, default='demo_tracking.mp4',
                        help='输出视频路径')
    parser.add_argument('--show', action='store_true',
                        help='是否实时显示窗口（WSL 建议不加）')
    args = parser.parse_args()

    track_video(args.source, args.output, args.show)

    # 自动复制到 assets（如果生成成功）
    if Path(args.output).exists():
        import shutil

        shutil.copy(args.output, 'assets/demo_tracking.mp4')
        print("✅ 已复制到 assets/demo_tracking.mp4")