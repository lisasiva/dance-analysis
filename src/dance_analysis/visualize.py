"""Preview frames (to identify track ids) and metric comparison plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .audio import BeatGrid
from .metrics import speed_series
from .pose import PoseSequence, Track


def save_track_preview(video: Path, seq: PoseSequence, out: Path) -> Path:
    """Draw each track's skeleton + id on a representative frame so the user can
    tell which track number is them vs. the reference dancer."""
    import cv2

    # pick a frame where the most tracks are simultaneously visible
    from collections import Counter
    counts: Counter = Counter()
    for t in seq.tracks.values():
        counts.update(t.frames.tolist())
    target = counts.most_common(1)[0][0] if counts else 0

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read preview frame from video.")

    colors = [(0, 200, 255), (255, 80, 80), (80, 255, 120), (255, 80, 255)]
    for i, (tid, t) in enumerate(seq.tracks.items()):
        if target not in t.frames:
            continue
        fi = int(np.where(t.frames == target)[0][0])
        kp = t.kps[fi]
        col = colors[i % len(colors)]
        for x, y, c in kp:
            if c >= config.KP_CONF_MIN:
                cv2.circle(frame, (int(x), int(y)), 4, col, -1)
        valid = kp[kp[:, 2] >= config.KP_CONF_MIN]
        if len(valid):
            x0, y0 = int(valid[:, 0].min()), int(valid[:, 1].min())
            cv2.putText(frame, f"track {tid}", (x0, max(20, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    cv2.imwrite(str(out), frame)
    return out


def plot_comparison(me: Track, ref: Track, fps: float, grid: BeatGrid, out: Path) -> Path:
    """Motion-energy of you vs reference with beat markers overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tm, sm = speed_series(me, fps)
    tr, sr = speed_series(ref, fps)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(tm, sm, label="you", color="#00b4d8", lw=1.6)
    ax.plot(tr, sr, label="reference", color="#ff5a5a", lw=1.6, alpha=0.85)
    for b in grid.beats:
        ax.axvline(b, color="#999", lw=0.5, alpha=0.4)
    ax.set_xlabel("seconds")
    ax.set_ylabel("motion energy (normalized)")
    ax.set_title("Motion energy vs. beat grid — gray lines are beats")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
