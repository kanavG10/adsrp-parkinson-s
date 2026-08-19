"""BlazePose 33-keypoint topology + the clinically-motivated joint groups
used by the V-JEPA masking study (MDS-UPDRS Item 3.10).

mediapipe>=1.0 removed `mp.solutions`, so the connection list and the
landmark names are defined here rather than imported.
"""

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

POSE_CONNECTIONS = [
    # face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    # arms + hands
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # legs + feet
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Joint groups for the six pretraining conditions. Indices listed here are the
# ones a given model MASKS during pretraining; all 33 are visible at inference.
MASK_GROUPS = {
    "lower_body":   [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    "arm_swing":    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "gait_full":    list(range(11, 33)),
    "left_side":    [1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
    "face_control": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

# BGR colours for rendering, keyed by anatomical region.
REGION_COLORS = {
    "face":  (180, 180, 180),
    "arms":  (80, 175, 250),
    "torso": (200, 160, 90),
    "legs":  (90, 200, 120),
}

def region_of(idx: int) -> str:
    if idx <= 10:
        return "face"
    if idx <= 22:
        return "arms"
    if idx in (23, 24):
        return "torso"
    return "legs"
