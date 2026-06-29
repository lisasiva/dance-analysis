"""Pull source clips into data/raw/.

Supports Google Drive (share link or file id) and YouTube URLs.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from . import config


def _drive_file_id(url_or_id: str) -> str:
    """Accept a raw id, a /file/d/<id>/ link, or an ?id=<id> link."""
    if "/" not in url_or_id and "=" not in url_or_id:
        return url_or_id
    m = re.search(r"/file/d/([\w-]+)", url_or_id)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([\w-]+)", url_or_id)
    if m:
        return m.group(1)
    raise ValueError(f"Could not parse a Drive file id from: {url_or_id!r}")


def from_drive(url_or_id: str, name: str) -> Path:
    """Download a single Drive file to data/raw/<name>.mp4."""
    import gdown

    file_id = _drive_file_id(url_or_id)
    out = config.raw_path(name)
    gdown.download(id=file_id, output=str(out), quiet=False)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(
            "Drive download produced no file. If the clip is large or restricted, "
            "make sure link-sharing is set to 'Anyone with the link'."
        )
    return out


def from_youtube(url: str, name: str) -> Path:
    """Download a YouTube clip to data/raw/<name>.mp4 via yt-dlp."""
    out = config.raw_path(name)
    # run yt-dlp via the current interpreter so it resolves inside the venv
    subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--no-playlist", "-f", "mp4",
         "-o", str(out), url],
        check=True,
    )
    return out


def ingest(name: str, drive: str | None = None, youtube: str | None = None) -> Path:
    if drive:
        return from_drive(drive, name)
    if youtube:
        return from_youtube(youtube, name)
    raise ValueError("Provide one of --drive or --youtube.")
