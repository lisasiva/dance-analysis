"""Video -> tracked multi-person keypoints using YOLO11-pose + ByteTrack.

Output is a PoseSequence: a dict of track_id -> per-frame keypoints, plus fps.
Saved to data/processed/<name>.pose.npy (pickled dict).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


@dataclass
class Track:
    track_id: int
    frames: np.ndarray      # (N,) int frame indices where this track was seen
    kps: np.ndarray         # (N, 17, 3)  -> x, y, confidence


class PoseSequence:
    """All tracked people in one clip."""

    def __init__(self, tracks: dict[int, Track], fps: float, size: tuple[int, int]):
        self.tracks = tracks
        self.fps = fps
        self.size = size  # (width, height)

    # --- io ---
    def save(self, path: Path) -> None:
        obj = {
            "fps": self.fps,
            "size": self.size,
            "tracks": {
                tid: {"frames": t.frames, "kps": t.kps}
                for tid, t in self.tracks.items()
            },
        }
        np.save(path, obj, allow_pickle=True)

    @classmethod
    def load(cls, path: Path) -> "PoseSequence":
        obj = np.load(path, allow_pickle=True).item()
        tracks = {
            tid: Track(tid, d["frames"], d["kps"])
            for tid, d in obj["tracks"].items()
        }
        return cls(tracks, obj["fps"], tuple(obj["size"]))

    # --- convenience ---
    def summary(self) -> list[tuple[int, int, float]]:
        """(track_id, n_frames, seconds_visible) sorted by visibility desc."""
        rows = [
            (tid, len(t.frames), len(t.frames) / self.fps)
            for tid, t in self.tracks.items()
        ]
        return sorted(rows, key=lambda r: r[1], reverse=True)


def extract_pose(video: Path, name: str | None = None,
                 model_name: str = config.POSE_MODEL,
                 conf: float = 0.5) -> PoseSequence:
    """Run pose tracking over a clip and return (and optionally save) a PoseSequence."""
    from ultralytics import YOLO

    model = YOLO(model_name)
    tracks: dict[int, dict] = {}
    fps = 30.0
    size = (0, 0)

    # stream=True yields one Results per frame without loading the whole video.
    results = model.track(
        source=str(video), stream=True, persist=True,
        tracker="bytetrack.yaml", conf=conf, verbose=False,
    )

    for frame_idx, r in enumerate(results):
        if r.orig_shape:
            size = (r.orig_shape[1], r.orig_shape[0])  # (w, h)
        kpts = r.keypoints
        boxes = r.boxes
        if kpts is None or boxes is None or boxes.id is None:
            continue
        ids = boxes.id.int().cpu().numpy()
        xy = kpts.xy.cpu().numpy()            # (P, 17, 2)
        cfd = kpts.conf.cpu().numpy()         # (P, 17)
        for p, tid in enumerate(ids):
            tid = int(tid)
            kp = np.concatenate([xy[p], cfd[p][:, None]], axis=1)  # (17, 3)
            tracks.setdefault(tid, {"frames": [], "kps": []})
            tracks[tid]["frames"].append(frame_idx)
            tracks[tid]["kps"].append(kp)

    # try to recover real fps from the capture
    fps = _probe_fps(video) or fps

    built = {
        tid: Track(tid, np.asarray(d["frames"]), np.asarray(d["kps"]))
        for tid, d in tracks.items()
    }
    seq = PoseSequence(built, fps, size)
    if name:
        seq.save(config.pose_path(name))
    return seq


def _probe_fps(video: Path) -> float | None:
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 0 else None
