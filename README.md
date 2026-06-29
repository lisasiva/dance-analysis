# dance-analysis

A pose + audio pipeline for comparing your **street-style** dancing against a target team,
quantifying the gaps that actually matter between you and a team reference.

## What it measures

Street-style auditions reward a mix of qualities, and different teams weight them
differently. This pipeline scores each and ranks them per the target team:

- **Pocket** — when you *initiate* movement relative to the beat (after = patient/in the pocket, before = rushing).
- **Timing** — accent placement vs. the reference, and how consistent.
- **Range of motion / line & extension** — per body region (head, shoulders, chest, hips, arms, legs). Heavily weighted.
- **Sharpness / attack** — how crisply you hit and freeze.
- **Fluidity** — sustained, gooey movement vs. hit-and-freeze.
- **Dynamic range** — big moves big, stillness still.
- **Groove** — bounce / weight-shift, and whether it locks to the beat.
- **Body engagement** — how many body parts you fire in a single move ("filling it up").
- **Sync** — how closely you match the reference's shape. *Low priority — your own style is fine.*

## Two comparison modes

1. **`same-clip` (preferred)** — one video with both you and a team member doing the
   *same* choreography. Same camera fps and music ⇒ no time-warping ⇒ the most precise sync/timing read.
2. **`two-clips` (fallback)** — separate videos. We extract a beat grid for each and DTW-align
   your run onto the reference. Less precise, but works when you can't film side-by-side.

## Pipeline stages

```
ingest  ->  pose      ->  beats     ->  compare        ->  report
(Drive/     (YOLO11    (librosa      (normalize +         (metrics +
 YouTube)    -pose +     beat grid)    align + metrics)     gap ranking)
            tracking)
```

## Quick start (no command line)

For non-technical users. Needs Python 3 installed.

```
git clone <this repo>
cd dance-analysis
./setup.sh      # installs everything (and ffmpeg) — first run takes a few minutes
./run.sh        # opens the app in your browser
```

Then in the browser: paste a Google Drive or YouTube link, enter the film date, song title,
and instructor, describe yourself + the reference dancer, pick the target team, and click
through the two buttons. The app shows a labeled image of all detected dancers so you can
pick which one is you.

## Setup (command line)

Requires Python 3.11–3.12 recommended (3.13 works but torch wheels are newer), and **ffmpeg**.

```
brew install ffmpeg
cd ~/Desktop/dance-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```
# 1. Pull a clip. The id is built as <date>-<title>-<instructor> and printed back.
python -m dance_analysis ingest --drive "<drive-or-youtube-url>" \
    --date 2026-06-15 --title "Pony" --instructor "Dalia"
#    -> id: 2026-06-15-pony-dalia   (use this id for the next steps)

# 2. Pose + beats + a labeled preview, in one step
python -m dance_analysis prep 2026-06-15-pony-dalia
#    Open the preview to see which track is you and which is the reference.

# 3. Compare two tracks in the same clip (you vs. reference)
python -m dance_analysis compare 2026-06-15-pony-dalia --me 1 --ref 2 --team project-a

# -> writes data/reports/2026-06-15-pony-dalia/  (report.md, metrics.json/csv, plots)
```

## Layout

```
src/dance_analysis/
  config.py     paths + constants (COCO keypoints, hip-hop metric weights)
  ingest.py     Google Drive / YouTube -> data/raw
  pose.py       video -> tracked keypoints  (YOLO11-pose + ByteTrack)
  audio.py      video -> beat grid          (librosa)
  align.py      scale-normalize + DTW time alignment
  metrics.py    angles, speed, sharpness, hit-timing, sync
  visualize.py  skeleton overlays + metric plots
  report.py     metrics -> markdown gap report
  cli.py        argparse entrypoint
data/raw/        downloaded clips
data/processed/  pose + beat artifacts
data/reports/    output per clip
```

## Status

First runnable slice: ingest + pose + beats + same-clip compare + report.
DTW two-clip path is wired in `align.py` and exposed but less battle-tested.
