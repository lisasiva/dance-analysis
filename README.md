# dance-analysis

A pose + audio pipeline for comparing your performance against that of another dancer you admire,
quantifying the gaps that actually matter

## Quick start

Needs Python 3 installed.

```
git clone <this repo>
cd dance-analysis
./setup.sh      # installs everything (and ffmpeg) — first run takes a few minutes
./run.sh        # opens the app in your browser at http://localhost:8501/
```

## What dance-analysis measures

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

## Future Improvements

* Show summary in sidebar of key metrics to focus on now, based on most recent reports
* Generate a training plan to improve key metrics