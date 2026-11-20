"""
Sports player tracker: YOLO detection + ByteTrack tracking + per-player speed/distance.

Usage:
    python track.py --video clips/play.mp4
    python track.py --video clips/play.mp4 --field-points field_points.json   # real yards/mph
    python track.py --video clips/play.mp4 --yards-per-pixel 0.05             # rough scale

Outputs (in --out, default outputs/<video name>/):
    annotated.mp4   video with boxes, track IDs, speed labels and trails
    tracks.csv      one row per player per frame
    summary.csv     one row per player: distance, max/avg speed
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from metrics import FieldMapper, summarize_tracks

PERSON_CLASS_ID = 0  # COCO "person"


def parse_args():
    p = argparse.ArgumentParser(description="Track players and measure speed/distance.")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO weights (yolov8n.pt is fastest)")
    p.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    p.add_argument("--field-points", default=None,
                   help="JSON with 4+ pixel->field point pairs (see calibrate.py)")
    p.add_argument("--yards-per-pixel", type=float, default=None,
                   help="Rough scale if you have no field points")
    p.add_argument("--min-frames", type=int, default=15,
                   help="Drop tracks shorter than this (removes flicker IDs)")
    p.add_argument("--max-frames", type=int, default=None, help="Stop early (for quick tests)")
    p.add_argument("--out", default=None, help="Output folder")
    return p.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.out) if args.out else Path("outputs") / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    info = sv.VideoInfo.from_video_path(str(video_path))
    fps = info.fps
    mapper = FieldMapper.from_args(args.field_points, args.yards_per_pixel)

    model = YOLO(args.model)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(trace_length=int(fps * 2), position=sv.Position.BOTTOM_CENTER)

    rows = []
    recent = defaultdict(lambda: deque(maxlen=max(2, int(fps // 2))))  # ~0.5 s window per ID

    frames = sv.get_video_frames_generator(str(video_path))
    with sv.VideoSink(str(out_dir / "annotated.mp4"), info) as sink:
        for frame_idx, frame in enumerate(frames):
            if args.max_frames and frame_idx >= args.max_frames:
                break
            t = frame_idx / fps

            # YOLO detects people; ByteTrack (built into Ultralytics) links them across frames
            result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                 classes=[PERSON_CLASS_ID], conf=args.conf, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(result)
            if dets.tracker_id is None:  # no confirmed tracks yet in this frame
                dets = sv.Detections.empty()
                dets.tracker_id = np.array([], dtype=int)

            # Feet position = bottom-center of the box (closest point to the field plane)
            feet = dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            field_xy = mapper.to_field(feet) if len(feet) else np.empty((0, 2))

            labels = []
            for i, tid in enumerate(dets.tracker_id):
                fx, fy = field_xy[i]
                recent[tid].append((t, fx, fy))
                speed = _window_speed(recent[tid])
                rows.append({
                    "frame": frame_idx, "time_s": round(t, 3), "track_id": int(tid),
                    "x_px": float(feet[i][0]), "y_px": float(feet[i][1]),
                    "x_field": float(fx), "y_field": float(fy),
                })
                speed_txt = f"{mapper.speed_label(speed)}" if speed is not None else ""
                labels.append(f"#{tid} {speed_txt}".strip())

            annotated = trace_annotator.annotate(frame.copy(), dets)
            annotated = box_annotator.annotate(annotated, dets)
            annotated = label_annotator.annotate(annotated, dets, labels=labels)
            sink.write_frame(annotated)

            if frame_idx % 50 == 0:
                print(f"frame {frame_idx}/{info.total_frames}  players tracked: {len(dets)}")

    tracks = pd.DataFrame(rows)
    tracks.to_csv(out_dir / "tracks.csv", index=False)
    summary = summarize_tracks(tracks, fps=fps, mapper=mapper, min_frames=args.min_frames)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(f"\nSaved to {out_dir}/")
    print(summary.head(15).to_string(index=False))
    with open(out_dir / "run_info.json", "w") as f:
        json.dump({"video": str(video_path), "fps": fps, "frames": int(tracks["frame"].max() + 1) if len(tracks) else 0,
                   "units": mapper.units, "model": args.model, "conf": args.conf}, f, indent=2)


def _window_speed(window):
    """Speed over the recent window (field units per second), None if too short."""
    if len(window) < 2:
        return None
    (t0, x0, y0), (t1, x1, y1) = window[0], window[-1]
    dt = t1 - t0
    return float(np.hypot(x1 - x0, y1 - y0) / dt) if dt > 0 else None


if __name__ == "__main__":
    main()
