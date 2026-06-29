"""Paths and constants shared across the pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path

# ---- paths ------------------------------------------------------------------
# repo root = three parents up from this file (src/dance_analysis/config.py)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = DATA / "reports"

for _d in (RAW, PROCESSED, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)


def raw_path(name: str, ext: str = ".mp4") -> Path:
    return RAW / f"{name}{ext}"


def pose_path(name: str) -> Path:
    return PROCESSED / f"{name}.pose.npy"


def beats_path(name: str) -> Path:
    return PROCESSED / f"{name}.beats.npy"


def report_dir(name: str) -> Path:
    d = REPORTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def meta_path(name: str) -> Path:
    return PROCESSED / f"{name}.meta.json"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").strip().lower()).strip("-")


def make_clip_id(film_date: str, song_title: str, instructor: str) -> str:
    """Standardized clip id: <film-date>-<song-title>-<instructor>."""
    parts = [_slug(film_date), _slug(song_title), _slug(instructor)]
    if not all(parts):
        raise ValueError("film date, song title, and instructor are all required.")
    return "-".join(parts)


def general_feedback_path() -> Path:
    """Feedback that applies to you across all clips (not clip-specific)."""
    return PROCESSED / "general.feedback.txt"


def load_meta(name: str) -> dict:
    """Per-clip metadata: nickname, link, descriptions, team."""
    p = meta_path(name)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_meta(name: str, **fields) -> dict:
    """Merge non-None fields into the clip's metadata file."""
    m = load_meta(name)
    m.update({k: v for k, v in fields.items() if v is not None})
    meta_path(name).write_text(json.dumps(m, indent=2))
    return m


# ---- pose model -------------------------------------------------------------
# nano is fast on CPU; bump to yolo11s-pose / yolo11m-pose for accuracy on a GPU.
POSE_MODEL = "yolo11n-pose.pt"

# COCO-17 keypoint order produced by YOLO pose.
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINTS)}

# Joint angles we track: (joint, point_a, point_b) -> angle at `joint` between a and b.
ANGLES = {
    "l_elbow": ("left_elbow", "left_shoulder", "left_wrist"),
    "r_elbow": ("right_elbow", "right_shoulder", "right_wrist"),
    "l_shoulder": ("left_shoulder", "left_elbow", "left_hip"),
    "r_shoulder": ("right_shoulder", "right_elbow", "right_hip"),
    "l_knee": ("left_knee", "left_hip", "left_ankle"),
    "r_knee": ("right_knee", "right_hip", "right_ankle"),
    "l_hip": ("left_hip", "left_shoulder", "left_knee"),
    "r_hip": ("right_hip", "right_shoulder", "right_knee"),
}

# Confidence below which a keypoint is treated as missing.
KP_CONF_MIN = 0.30

# Body segments, for measuring which parts of the body carry the movement
# ("filling up movement" / using head + chest vs. only limbs).
SEGMENTS = {
    "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    "chest": ["left_shoulder", "right_shoulder"],
    "hips": ["left_hip", "right_hip"],
    "arms": ["left_elbow", "right_elbow", "left_wrist", "right_wrist"],
    "legs": ["left_knee", "right_knee", "left_ankle", "right_ankle"],
}

# ---- metric weights ---------------------------------------------------------
# The dimensions the report ranks. Each clip gets a 0-1 gap per dimension,
# weighted by the target team's profile below.
DIMENSIONS = ["timing", "sharpness", "fluidity", "dynamics",
              "groove", "engagement", "picture", "sync", "rom"]

# Generic default (used when no --team is given).
# `picture` (catching & holding the reference's shapes) is weighted heavily — it's
# the one quality signal that validated. `sync` is tiny (matching a reference's exact
# shape rewards conformity); `rom` is modest (big line/extension isn't a primary street
# win condition).
METRIC_WEIGHTS = {
    "timing": 0.15, "sharpness": 0.15, "fluidity": 0.10, "dynamics": 0.10,
    "groove": 0.15, "engagement": 0.08, "picture": 0.18, "sync": 0.02, "rom": 0.07,
}

# Per-team style emphasis, derived from director feedback. Tunable.
TEAM_WEIGHTS = {
    # Project A: tick-tick-tick, sharp/explosive/clean, deep pocket, strong grooves.
    "project-a": {
        "timing": 0.18, "sharpness": 0.18, "fluidity": 0.03, "dynamics": 0.08,
        "groove": 0.20, "engagement": 0.04, "picture": 0.18, "sync": 0.03, "rom": 0.08,
    },
    # Fine Lines: fluid/gooey, hip & chest, "filling up" movement, held lines.
    "fine-lines": {
        "timing": 0.08, "sharpness": 0.06, "fluidity": 0.20, "dynamics": 0.10,
        "groove": 0.08, "engagement": 0.15, "picture": 0.18, "sync": 0.03, "rom": 0.12,
    },
}

# Target timing offset in ms (where accents should land relative to the beat).
# Positive = behind the beat ("in the pocket"). 0 = on the beat.
TEAM_TIMING_TARGET_MS = {
    "project-a": 60.0,   # deep pocket: almost late, but not late
    "fine-lines": 20.0,  # slightly relaxed, but more on the beat than Project A
}
DEFAULT_TIMING_TARGET_MS = 0.0


def weights_for(team: str | None) -> dict:
    return TEAM_WEIGHTS.get(team, METRIC_WEIGHTS) if team else METRIC_WEIGHTS


def timing_target_for(team: str | None) -> float:
    return TEAM_TIMING_TARGET_MS.get(team, DEFAULT_TIMING_TARGET_MS) if team \
        else DEFAULT_TIMING_TARGET_MS
