"""Run MediaPipe PoseLandmarker over a video: dump per-frame keypoints and
render a skeleton overlay.

Uses the MediaPipe Tasks API (mediapipe>=1.0 removed the legacy
`mp.solutions.pose` interface that most tutorials still use).

Example:
    python src/extract_pose.py data/raw/OAW01-bottom.mp4 --model full --seconds 20
"""
import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.dirname(__file__))
from pose_topology import (LANDMARK_NAMES, POSE_CONNECTIONS, REGION_COLORS,
                           region_of)
from roi_tracker import ROITracker, median_background


def build_landmarker(model_path, num_poses=1, min_conf=0.5):
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=model_path,
            # macOS: the Metal delegate aborts inside PoseLandmarker
            # (DrishtiMetalHelper 'Service is unavailable'), so pin to CPU.
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_conf,
        min_pose_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def draw_pose(frame, landmarks, scale=1.0):
    """Draw the skeleton on a BGR frame using normalized landmarks."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    vis = [getattr(lm, "visibility", 1.0) for lm in landmarks]

    thick = max(1, int(round(2 * scale)))
    for a, b in POSE_CONNECTIONS:
        if vis[a] < 0.3 or vis[b] < 0.3:
            continue
        color = REGION_COLORS[region_of(b if region_of(b) != "face" else a)]
        cv2.line(frame, pts[a], pts[b], color, thick, cv2.LINE_AA)

    for i, p in enumerate(pts):
        if vis[i] < 0.3:
            continue
        color = REGION_COLORS[region_of(i)]
        r = max(2, int(round(3 * scale)))
        cv2.circle(frame, p, r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p, r, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def extract(video_path, model_path, out_video=None, out_table=None,
            seconds=None, render_scale=0.5, progress_every=100,
            use_roi=True, min_conf=0.3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_target = int(seconds * fps) if seconds else n_total

    writer = None
    if out_video:
        ow, oh = int(w * render_scale), int(h * render_scale)
        writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (ow, oh))

    landmarker = build_landmarker(model_path, min_conf=min_conf)
    tracker = None
    if use_roi:
        print("  building static background model...", flush=True)
        bg = median_background(video_path)
        tracker = ROITracker(w, h, background=bg, target_side=768, min_side=320)
    rows, n_detected, n_roi_hits = [], 0, 0

    for idx in range(n_target):
        ok, frame = cap.read()
        if not ok:
            break
        if tracker is not None:
            det_img, mapping = tracker.prepare(frame)
        else:
            det_img, mapping = frame, None

        rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(idx * 1000.0 / fps)
        res = landmarker.detect_for_video(mp_img, ts_ms)

        has_pose = bool(res.pose_landmarks)
        lms = None
        if has_pose:
            lms = res.pose_landmarks[0]
            if tracker is not None:
                lms = tracker.remap(lms, mapping)
                if mapping is not None:
                    n_roi_hits += 1
        if tracker is not None:
            tracker.update(lms)

        if has_pose:
            n_detected += 1
            world = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
            for j, lm in enumerate(lms):
                wl = world[j] if world else None
                rows.append({
                    "frame": idx,
                    "time_s": idx / fps,
                    "joint_id": j,
                    "joint": LANDMARK_NAMES[j],
                    "x": lm.x, "y": lm.y, "z": lm.z,
                    "visibility": getattr(lm, "visibility", np.nan),
                    "presence": getattr(lm, "presence", np.nan),
                    "wx": wl.x if wl else np.nan,
                    "wy": wl.y if wl else np.nan,
                    "wz": wl.z if wl else np.nan,
                })

        if writer is not None:
            small = cv2.resize(frame, (int(w * render_scale), int(h * render_scale)))
            if has_pose:
                draw_pose(small, lms, scale=render_scale * 2)
            label = f"{os.path.basename(video_path)}  f{idx:05d}  {'POSE' if has_pose else 'NO POSE'}"
            cv2.putText(small, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(small, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(small)

        if progress_every and idx % progress_every == 0:
            print(f"  frame {idx}/{n_target}  detected={n_detected}", flush=True)

    cap.release()
    if writer is not None:
        writer.release()
    landmarker.close()

    df = pd.DataFrame(rows)
    if out_table and not df.empty:
        df.to_parquet(out_table, index=False)
    n_seen = idx + 1
    print(f"  -> {n_seen} frames, pose found in {n_detected} "
          f"({100.0 * n_detected / max(n_seen,1):.1f}%), "
          f"{n_roi_hits} via tracked ROI")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--seconds", type=float, default=None,
                    help="only process the first N seconds")
    ap.add_argument("--render-scale", type=float, default=0.5)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-roi", action="store_true",
                    help="disable ROI tracking (detect on full frames only)")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(f"{args.outdir}/annotated", exist_ok=True)
    os.makedirs(f"{args.outdir}/keypoints", exist_ok=True)

    out_video = None if args.no_video else f"{args.outdir}/annotated/{stem}_pose.mp4"
    out_table = f"{args.outdir}/keypoints/{stem}.parquet"

    print(f"[{stem}] model={args.model}")
    extract(args.video, f"models/pose_landmarker_{args.model}.task",
            out_video=out_video, out_table=out_table,
            seconds=args.seconds, render_scale=args.render_scale,
            use_roi=not args.no_roi, min_conf=args.min_conf)
    print(f"  keypoints -> {out_table}")
    if out_video:
        print(f"  video     -> {out_video}")


if __name__ == "__main__":
    main()
