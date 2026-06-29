# Metrics reference

Every number the pipeline produces, what it means, and exactly how it's computed.
Source of truth is `src/dance_analysis/metrics.py`; this doc explains it in plain terms.

## Foundations

**Pose** — `pose.py` runs YOLO11-pose, giving 17 COCO keypoints (x, y, confidence)
per person per frame, with a ByteTrack id so one dancer stays one "track." Keypoints
with confidence < 0.30 are treated as missing (NaN) so low-quality detections don't
pollute the math.

**Scale normalization** (`align.normalize_positions`) — every dancer is recentered on
their pelvis (hip midpoint) and divided by their **torso length** (shoulder-mid to
hip-mid distance). This makes positions comparable regardless of how tall someone is or
how close they are to the camera. Joint *angles* are already scale-free, so they need no
normalization.

**Beat grid** (`audio.py`) — librosa estimates tempo (BPM) and beat times, then we split
each beat in half to get an **8-count grid** (eighth-note resolution), which is the unit
hip-hop accents actually land on.

---

## Per-dancer metrics (`profile_track`)

### Motion energy / speed  (`speed_series`)
- **What:** how much the whole body is moving at each instant.
- **How:** frame-to-frame change in normalized keypoint positions, averaged over all 17
  joints, divided by the time between frames. One number per frame.
- **Why:** it's the backbone signal — accents, freezes, and dynamics all come from it.

### Dynamic range  (`dynamic_range`)
- **What:** contrast between your biggest movements and your stillest moments.
- **How:** 95th percentile of motion energy minus the 5th percentile.
- **Read:** higher = more dynamic (big moves big, freezes truly still). Flatness is the
  #1 thing that reads as "not performance-ready" in hip-hop.

### Sharpness  (`sharpness`)
- **What:** attack/freeze crispness — how abruptly you accelerate into and stop out of moves.
- **How:** 95th-percentile of the *rate of change* of motion energy, divided by mean motion
  energy (so it's not just "you move a lot"). High = snappy hits and clean dead-stops.

### Accents  (`movement_events`)
- **What:** the timestamps of your movement "hits."
- **How:** prominent local peaks in the motion-energy signal (`scipy.signal.find_peaks`,
  prominence ≈ half the signal's std). `n_accents` is how many were found.

### Timing bias & consistency  (`timing_offsets`)
- **What:** whether your accents land on the beat, and how reliably.
- **How:** each accent is matched to the nearest 8-count grid marker; the signed gap is the
  offset. **Bias** = mean offset in ms (negative = ahead of the beat, positive = behind).
  **Consistency (std)** = standard deviation of those offsets in ms (lower = tighter).
- **Read:** bias near 0 = on the beat. Std is usually the more important one — it's whether
  you're *reliably* on time vs. scattered. At 130 BPM an eighth-note ≈ 230 ms, so a 60 ms
  std is ~¼ of a subdivision.

### Range of motion  (`rom_deg`)
- **What:** how far each joint travels over the clip, in degrees.
- **How:** for 8 joints (left/right elbow, shoulder, knee, hip) we compute the joint angle
  every frame (`angle_series`) and take max − min.

### Symmetry imbalance  (`symmetry_imbalance`)
- **What:** how lopsided your left vs. right range is, per joint pair.
- **How:** |left ROM − right ROM| / larger of the two. 0 = perfectly even; 0.35 = the
  weaker side reaches ~35% less than the stronger side.

---

## Dancer-vs-dancer metrics (`sync_between`) — needs unison choreography

These compare YOU against a REFERENCE dancer **in the same clip**. They only mean what you
think they mean if both dancers are doing the *same move, same facing, same instant*. In a
group formation with different roles/facings, treat these as unreliable (use `--mirror`
and/or a windowed unison section — see README).

### Sync lag  (`lag_ms`)
- **What:** how far ahead/behind the reference your movement is, on average.
- **How:** cross-correlation of the two motion-energy signals; the lag at peak correlation,
  converted to ms. Negative = you lead, positive = you trail.

### Sync correlation  (`corr`)
- **What:** how well the *shape* of your motion matches theirs over time.
- **How:** peak of the normalized cross-correlation. ~1.0 = moving in lockstep; near 0 =
  unrelated (either real desync, or — more often — you're doing different parts).

### Pose difference  (`angle_diff_deg`)
- **What:** average difference in body shape between you and the reference.
- **How:** mean absolute difference across all 8 joint angles, sampled over the overlap.
- **Read:** low = your body positions match theirs. Sensitive to facing — a mirrored
  formation inflates this unless you pass `--mirror`.

---

## How gaps get ranked (`report._gap_scores`)

Each dimension is converted to a 0–1 gap (1 = biggest gap to the reference), then weighted
by `config.METRIC_WEIGHTS` (hip-hop default: timing 35%, sharpness 25%, dynamics 20%,
sync 15%, ROM/symmetry 5%) and sorted. The weights are editable per target team (see README
— different teams, different style emphasis).

**Important honesty note:** the `sync` gap is only trustworthy with unison input. If your
reference is in a different formation spot, demote it mentally until you re-run on a unison
window.
