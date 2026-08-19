# adsrp-parkinson-s

V-JEPA LoRA fine-tuning for Parkinson's gait analysis.

---


Pose-extraction demo ahead of the masking experiments: MediaPipe BlazePose
(33 keypoints) over walking video, producing per-frame keypoint tables,
skeleton-overlay videos, and gait-signal figures.

## Run it in Colab

Open `02_keypoint_extraction.ipynb` with the badge above — it clones this
repo, installs pinned deps, pulls a demo clip and renders the overlay inline.
No GPU required; BlazePose is CPU-only.

## Setup (local)

Python **3.12** in `.venv` (MediaPipe has no 3.13 wheels).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  mediapipe==0.10.35 opencv-python numpy pandas matplotlib pyarrow remotezip
```

Pose models land in `models/` (lite / full / heavy):

```bash
curl -sSL -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

### Two environment traps

- **`mediapipe==1.0.1` crashes on macOS arm64.** `PoseLandmarker` aborts in
  `DrishtiMetalHelper` (`Check failed: service_ Service is unavailable`)
  regardless of whether the CPU delegate is requested. Pinned to `0.10.35`,
  the last 0.10.x with native arm64 wheels.
- **`mp.solutions` no longer exists** in either 0.10.30+ or 1.0. Nearly every
  tutorial online uses `mp.solutions.pose`; that API is gone. This code uses
  the Tasks API (`mediapipe.tasks.python.vision.PoseLandmarker`), and defines
  the skeleton topology and renderer itself (`src/pose_topology.py`).

## Running

```bash
# keypoints + annotated video
.venv/bin/python src/extract_pose.py data/raw/OAW01-bottom.mp4 --model full

# figures + summary stats
.venv/bin/python src/gait_report.py outputs/keypoints/OAW01-bottom.parquet
```

Useful flags: `--seconds N` (quick test), `--no-video`, `--no-roi`,
`--model {lite,full,heavy}`.

## Why ROI tracking exists

The subject walks a long hall in a 1080x1920 frame and is often only ~100 px
tall — well below what BlazePose's person detector fires on. Detection rate on
raw frames was **47%**. Fixes, in the order they mattered:

| configuration | detection rate |
|---|---|
| full frames | 46.9% |
| + ROI crop/upscale tracking | 47.9% |
| + MOG2 motion bootstrap | 68.1% |
| + static median-background bootstrap | **99.8%** |

ROI tracking alone barely helped because it can only engage *after* a
successful full-frame detection — exactly what fails at distance. The
bootstrap is what mattered: because the camera is static, a per-pixel median
over frames sampled across the clip yields an empty-room background, and
thresholded frame differencing localises the walker from frame 0. An adaptive
subtractor (MOG2) needs a warm-up during which every frame is one large blob,
which is why misses clustered entirely in the first 6.3 s.

Landmarks detected inside a crop are mapped back to whole-frame normalised
coordinates, so downstream code never sees the cropping.

## Outputs

- `outputs/keypoints/<clip>.parquet` — one row per (frame, joint): image-space
  `x,y,z`, `visibility`, `presence`, and metric world coordinates `wx,wy,wz`.
- `outputs/annotated/<clip>_pose.mp4` — skeleton overlay, coloured by region.
- `outputs/figures/<clip>_quality.png` — per-joint visibility heatmap over time.
- `outputs/figures/<clip>_gait.png` — ankle height, arm swing, cadence spectrum.
- `outputs/figures/<clip>_masking_conditions.png` — the six pretraining
  conditions drawn on a real tracked skeleton.

**Use world landmarks (`wx,wy,wz`) for gait metrics, not image `x,y`.** The
subject walks toward and away from the camera, so pixel scale drifts and the
perspective envelope swamps the step cycle. World landmarks are metric and
hip-centred, which is why the cadence peak comes out sharp.

## Data

The `gait video drive` Drive folder holds **no video** — it is five shortcuts
to external datasets (figshare walking videos, GAVD clinical annotations,
ProGait, AlphaPose, a Mendeley set).

This demo uses figshare [`10.6084/m9.figshare.19929893`][fs] ("Walking
videos", CC0): 14 subjects, two camera views each, shipped as one 2.35 GB zip.
`remotezip` pulls individual files via HTTP range requests instead of
downloading the whole archive. **These are not Parkinson's videos** — they
demonstrate the pipeline only; severity labels still need a labelled corpus
(CARE-PD / P4DT / GAVD).

[fs]: https://doi.org/10.6084/m9.figshare.19929893.v1

## Masking conditions

`MASK_GROUPS` in `src/pose_topology.py` defines which joint indices each
condition **hides during pretraining**; all 33 are visible at inference.
Grounded in MDS-UPDRS Item 3.10 (stride amplitude, stride speed, foot-lift
height, heel strike, turning, arm swing).

| model | condition | masks |
|---|---|---|
| 1 | random (baseline) | random % of all 33 |
| 2 | `lower_body` | 10 — hips, knees, ankles, heels, foot index |
| 3 | `arm_swing` | 12 — shoulders, elbows, wrists, hands |
| 4 | `gait_full` | 22 — arms + legs (11–32) |
| 5 | `left_side` | 16 — all left-side landmarks |
| 6 | `face_control` | 11 — face/head, negative control |

Note: Model 2 was specced as "12 joints" but the enumerated landmarks 23–32
are **10** — MediaPipe has no separate toe landmarks (heel + foot index cover
foot lift and heel strike). Coded as 10.
