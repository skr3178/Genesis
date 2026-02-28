#!/usr/bin/env python3
"""Fix corrupted MP4 video files by re-encoding with OpenCV."""
import argparse
from pathlib import Path

try:
    import cv2
except ImportError:
    print("OpenCV not installed. Install with: pip install opencv-python")
    exit(1)


def fix_video(input_path: str, output_path: str = None):
    """Re-encode a potentially corrupted video file."""
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_fixed.mp4"
    else:
        output_path = Path(output_path)
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {input_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")
    
    if width == 0 or height == 0:
        print("ERROR: Invalid video dimensions - file may be severely corrupted")
        cap.release()
        return False
    
    # Setup output writer with H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("ERROR: Cannot create output video writer")
        cap.release()
        return False
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    
    print(f"\nDone! Wrote {frame_count} frames to {output_path}")
    print(f"Output file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix corrupted MP4 videos")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output video file path (optional)")
    args = parser.parse_args()
    
    fix_video(args.input, args.output)
