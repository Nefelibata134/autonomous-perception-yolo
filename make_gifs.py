"""Convert MP4s to compressed GIFs for GitHub README (using OpenCV + PIL)"""
import cv2
import numpy as np
from PIL import Image
import os

PROJECT = "/home/nefelibata/autonomous_driving/autonomous-perception-yolo"

def mp4_to_gif(mp4_path, gif_path, duration=8.0, fps=7, max_size=400):
    cap = cv2.VideoCapture(mp4_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Reading: {os.path.basename(mp4_path)} ({os.path.getsize(mp4_path)/1024/1024:.0f} MB, {total_frames} frames @ {source_fps:.0f} fps)")
    
    # Calculate skip
    max_frames = int(duration * fps)
    if total_frames > max_frames:
        start_frame = total_frames // 3  # skip first third
        frames_needed = min(max_frames, total_frames - start_frame)
        skip = max(1, (total_frames - start_frame) // frames_needed)
    else:
        start_frame = 0
        skip = max(1, int(source_fps / fps))
    
    print(f"  Start frame: {start_frame}, skip every {skip}")
    
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx <= start_frame:
            continue
        if (frame_idx - start_frame) % skip != 0:
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        h, w = frame_rgb.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        frames.append(Image.fromarray(frame_rgb))
        
        if len(frames) >= max_frames:
            break
    
    cap.release()
    
    if not frames:
        print("  WARNING: No frames extracted!")
        return
    
    print(f"  {len(frames)} frames, writing GIF...")
    
    # Save as GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
        quality=85
    )
    
    size_mb = os.path.getsize(gif_path) / 1024 / 1024
    print(f"  → {os.path.basename(gif_path)} ({size_mb:.1f} MB)")
    return size_mb

# Tracking demo
tracking_mp4 = f"{PROJECT}/assets/demo_tracking.mp4"
tracking_gif = f"{PROJECT}/assets/demo_tracking.gif"
mp4_to_gif(tracking_mp4, tracking_gif, duration=6, fps=6, max_size=400)

# BEV demo
bev_mp4 = f"{PROJECT}/assets/demo_bev.mp4"
bev_gif = f"{PROJECT}/assets/demo_bev.gif"
mp4_to_gif(bev_mp4, bev_gif, duration=6, fps=6, max_size=400)

print(f"\nDone! Files in assets/:")
for f in sorted(os.listdir(f"{PROJECT}/assets")):
    size = os.path.getsize(f"{PROJECT}/assets/{f}")
    print(f"  {f:30s} {size/1024/1024:6.1f} MB" if size > 1024*1024 else f"  {f:30s} {size/1024:6.0f} KB")
