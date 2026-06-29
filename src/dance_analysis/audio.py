"""Video -> musical beat grid using librosa.

Pulls the audio track (via librosa/audioread, which needs ffmpeg) and returns
the estimated tempo and beat times in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


@dataclass
class BeatGrid:
    tempo: float            # BPM
    beats: np.ndarray       # beat times in seconds
    duration: float

    def save(self, path: Path) -> None:
        np.save(path, {
            "tempo": self.tempo,
            "beats": self.beats,
            "duration": self.duration,
        }, allow_pickle=True)

    @classmethod
    def load(cls, path: Path) -> "BeatGrid":
        d = np.load(path, allow_pickle=True).item()
        return cls(d["tempo"], d["beats"], d["duration"])

    def eighth_counts(self) -> np.ndarray:
        """Beat times subdivided into 8-count grid markers (each beat split in 2)."""
        if len(self.beats) < 2:
            return self.beats
        mids = (self.beats[:-1] + self.beats[1:]) / 2.0
        grid = np.sort(np.concatenate([self.beats, mids]))
        return grid


def extract_beats(video: Path, name: str | None = None) -> BeatGrid:
    import librosa

    y, sr = librosa.load(str(video), sr=None, mono=True)
    duration = len(y) / sr
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beats = librosa.frames_to_time(beat_frames, sr=sr)

    grid = BeatGrid(float(np.atleast_1d(tempo)[0]), beats, float(duration))
    if name:
        grid.save(config.beats_path(name))
    return grid
