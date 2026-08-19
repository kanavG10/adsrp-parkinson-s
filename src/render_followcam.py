"""Re-render a clip as a zoomed 'follow cam' locked onto the tracked subject.

Reads keypoints already extracted by extract_pose.py, so this is just crop +
draw + encode -- no re-inference. The full-frame overlay is faithful but the
walker is tiny in 1080x1920; this is the version worth showing an audience.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from extract_pose import draw_pose
from pose_topology import LANDMARK_NAMES


class _LM:
    """Minimal stand-in for a MediaPipe landmark, in crop-local coords."""
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x, y, v):
        self.x, self.y, self.visibility = x, y, v


def smooth(series, win):
    return pd.Series(series).rolling(win, center=True, min_periods=1).mean().values


def render(video, parquet, out_path, out_h=720, aspect=0.62, zoom=1.45,
           smooth_win=25):
    df = pd.read_parquet(parquet)
    X = df.pivot_table(index="frame", columns="joint_id", values="x")
    Y = df.pivot_table(index="frame", columns="joint_id", values="y")
    V = df.pivot_table(index="frame", columns="joint_id", values="visibility")

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Camera path: subject centre and size per frame, heavily smoothed so the
    # crop glides instead of jittering with per-frame landmark noise.
    cx = smooth(((X.min(axis=1) + X.max(axis=1)) / 2 * W).values, smooth_win)
    cy = smooth(((Y.min(axis=1) + Y.max(axis=1)) / 2 * H).values, smooth_win)
    # A walker is tall and narrow, so frame with a portrait crop driven by
    # body height; a square crop wastes most of its width on empty floor.
    body_h = (Y.max(axis=1) - Y.min(axis=1)).values * H
    ch_raw = np.clip(body_h * zoom, 240, H)
    side = smooth(ch_raw, smooth_win)
    frames = X.index.to_numpy()
    lookup = {f: i for i, f in enumerate(frames)}

    out_w = int(round(out_h * aspect / 2) * 2)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i = lookup.get(idx)
        if i is not None:
            hh = side[i] / 2
            hw = side[i] * aspect / 2
            x0 = int(np.clip(cx[i] - hw, 0, W - 1))
            y0 = int(np.clip(cy[i] - hh, 0, H - 1))
            x1 = int(np.clip(cx[i] + hw, x0 + 8, W))
            y1 = int(np.clip(cy[i] + hh, y0 + 8, H))
            crop = frame[y0:y1, x0:x1]
            ch, cw = crop.shape[:2]
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            lms = [_LM((X.iat[i, j] * W - x0) / cw,
                       (Y.iat[i, j] * H - y0) / ch,
                       V.iat[i, j]) for j in range(33)]
            draw_pose(crop, lms, scale=1.6)
            cv2.putText(crop, f"{idx/fps:5.1f}s", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(crop, f"{idx/fps:5.1f}s", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(crop)
        idx += 1
    cap.release()
    writer.release()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--zoom", type=float, default=1.45,
                    help="crop height as a multiple of body height")
    ap.add_argument("--size", type=int, default=720, help="output height")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    pq = args.parquet or f"outputs/keypoints/{stem}.parquet"
    out = args.out or f"outputs/annotated/{stem}_followcam.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("->", render(args.video, pq, out, out_h=args.size, zoom=args.zoom))


if __name__ == "__main__":
    main()
