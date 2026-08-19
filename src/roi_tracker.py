"""Adaptive ROI tracking for pose on small/distant subjects.

The walking videos are 1080x1920 with the subject often only ~100 px tall,
which is below what BlazePose's detector reliably fires on. Strategy:

  1. no ROI yet          -> propose one from motion (static camera, one
                            walking subject), else fall back to whole frame
  2. ROI known           -> crop around the previous pose, upscale, detect there
  3. detection succeeded -> refresh the ROI from the new landmarks
  4. N consecutive misses -> drop the ROI and re-detect on the whole frame

Landmarks found inside a crop are mapped back to whole-frame normalized
coordinates, so downstream code never has to know cropping happened.
World landmarks are root-relative metric values and need no remapping.
"""
import cv2
import numpy as np


def median_background(video_path, n_samples=40):
    """Empty-room background for a static camera: per-pixel median over
    frames sampled across the whole clip. The walker moves, so the median
    keeps the room and drops the person. Beats an adaptive subtractor here,
    which needs a warm-up during which every frame is one big blob."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, max(total - 1, 0), min(n_samples, total)).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


class ROITracker:
    def __init__(self, frame_w, frame_h, margin=0.6, min_side=224,
                 target_side=512, patience=5, vis_thresh=0.3,
                 use_motion=True, min_motion_area=300, background=None,
                 diff_thresh=18):
        self.W, self.H = frame_w, frame_h
        self.margin = margin
        self.min_side = min_side
        self.target_side = target_side
        self.patience = patience
        self.vis_thresh = vis_thresh
        self.roi = None          # (x0, y0, x1, y1) in full-frame pixels
        self.misses = 0
        self.use_motion = use_motion
        self.min_motion_area = min_motion_area
        self.background = background
        self.diff_thresh = diff_thresh
        # Fallback only when no static background was supplied.
        self.bg = (cv2.createBackgroundSubtractorMOG2(
            history=250, varThreshold=25, detectShadows=False)
            if use_motion and background is None else None)

    def _motion_bbox(self, frame):
        """Largest moving blob -> square ROI. Static camera, one walker."""
        if self.background is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, self.background)
            _, mask = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)
        else:
            mask = self.bg.apply(frame)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = cv2.dilate(mask, None, iterations=4)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < self.min_motion_area:
            return None
        x, y, bw, bh = cv2.boundingRect(c)
        cx, cy = x + bw / 2, y + bh / 2
        side = max(bw, bh) * (1 + self.margin)
        side = max(side, self.min_side)
        half = side / 2
        return self._clamp(cx - half, cy - half, cx + half, cy + half)

    def _bbox_from_landmarks(self, landmarks):
        pts = [(lm.x * self.W, lm.y * self.H) for lm in landmarks
               if getattr(lm, "visibility", 1.0) >= self.vis_thresh]
        if len(pts) < 4:
            return None
        xs, ys = zip(*pts)
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = max(x1 - x0, y1 - y0) * (1 + self.margin)
        side = max(side, self.min_side)
        half = side / 2
        return self._clamp(cx - half, cy - half, cx + half, cy + half)

    def _clamp(self, x0, y0, x1, y1):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.W, int(x1)), min(self.H, int(y1))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        return (x0, y0, x1, y1)

    def prepare(self, frame):
        """Return (image_to_detect, mapping_info)."""
        roi = self.roi
        if roi is None and self.use_motion:
            roi = self._motion_bbox(frame)
        elif self.use_motion and self.bg is not None:
            self.bg.apply(frame)   # keep the adaptive model current
        if roi is None:
            return frame, None
        x0, y0, x1, y1 = roi
        crop = frame[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        scale = self.target_side / max(ch, cw)
        if scale > 1:
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                              interpolation=cv2.INTER_CUBIC)
        return crop, (x0, y0, cw, ch)

    def remap(self, landmarks, mapping):
        """Crop-normalized landmarks -> full-frame normalized, in place."""
        if mapping is None:
            return landmarks
        x0, y0, cw, ch = mapping
        for lm in landmarks:
            lm.x = (x0 + lm.x * cw) / self.W
            lm.y = (y0 + lm.y * ch) / self.H
        return landmarks

    def update(self, landmarks):
        """Feed back the (already remapped) landmarks; manages ROI lifecycle."""
        if landmarks is None:
            self.misses += 1
            if self.misses >= self.patience:
                self.roi = None
                self.misses = 0
            return
        self.misses = 0
        box = self._bbox_from_landmarks(landmarks)
        if box is not None:
            self.roi = box
