"""Scale-normalization, time-windowing, and mirroring of tracked skeletons.

Angles are inherently scale-invariant; for speed/dynamics we normalize *positions*
(pelvis-centered, torso-scaled) so dancers of different sizes/distances compare fairly.
"""

from __future__ import annotations

import numpy as np

from . import config
from .pose import Track


def torso_length(kps: np.ndarray) -> np.ndarray:
    """Per-frame distance from shoulder-midpoint to hip-midpoint. Shape (N,)."""
    ls, rs = config.KP["left_shoulder"], config.KP["right_shoulder"]
    lh, rh = config.KP["left_hip"], config.KP["right_hip"]
    sh_mid = (kps[:, ls, :2] + kps[:, rs, :2]) / 2
    hip_mid = (kps[:, lh, :2] + kps[:, rh, :2]) / 2
    d = np.linalg.norm(sh_mid - hip_mid, axis=1)
    d[d < 1e-6] = np.nan
    return d


def normalize_positions(track: Track) -> np.ndarray:
    """Center on pelvis, scale by torso length. Returns (N, 17, 2), scale-invariant."""
    kps = track.kps.astype(float).copy()
    lh, rh = config.KP["left_hip"], config.KP["right_hip"]
    pelvis = (kps[:, lh, :2] + kps[:, rh, :2]) / 2  # (N, 2)
    scale = torso_length(kps)                       # (N,)
    pos = (kps[:, :, :2] - pelvis[:, None, :]) / scale[:, None, None]
    return pos


def window_track(track: Track, fps: float, start: float | None,
                 end: float | None) -> Track:
    """Return a copy of the track restricted to [start, end] seconds."""
    t = track.frames.astype(float) / fps
    mask = np.ones(len(t), dtype=bool)
    if start is not None:
        mask &= t >= start
    if end is not None:
        mask &= t <= end
    return Track(track.track_id, track.frames[mask], track.kps[mask])


# left/right keypoint pairs to swap when mirroring (COCO-17)
_MIRROR_PAIRS = [
    ("left_eye", "right_eye"), ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"), ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"), ("left_hip", "right_hip"),
    ("left_knee", "right_knee"), ("left_ankle", "right_ankle"),
]


def mirror_track(track: Track, width: int) -> Track:
    """Horizontally mirror a dancer (flip x, swap left/right joints).

    Use when comparing a mirrored formation so 'same move, opposite side' still
    scores as a match.
    """
    kps = track.kps.copy()
    kps[:, :, 0] = width - kps[:, :, 0]
    for a, b in _MIRROR_PAIRS:
        ia, ib = config.KP[a], config.KP[b]
        kps[:, [ia, ib]] = kps[:, [ib, ia]]
    return Track(track.track_id, track.frames, kps)
