"""Movement metrics: angles, speed, sharpness, dynamics, hit-timing, sync.

All functions operate on a single Track unless noted. Times are in seconds.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from . import config
from .align import normalize_positions, torso_length
from .audio import BeatGrid
from .pose import Track


def _times(track: Track, fps: float) -> np.ndarray:
    return track.frames.astype(float) / fps


def _angle_at(joint: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees at `joint` between vectors to a and b. Inputs (N,2)."""
    v1 = a - joint
    v2 = b - joint
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos = np.einsum("ij,ij->i", v1, v2) / (n1 * n2 + 1e-9)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def angle_series(track: Track, fps: float) -> dict[str, np.ndarray]:
    """Return {angle_name: degrees per frame}. NaN where keypoints are low-conf."""
    kps = track.kps.astype(float)
    conf = kps[:, :, 2]
    out = {}
    for name, (j, a, b) in config.ANGLES.items():
        ji, ai, bi = config.KP[j], config.KP[a], config.KP[b]
        ang = _angle_at(kps[:, ji, :2], kps[:, ai, :2], kps[:, bi, :2])
        bad = (conf[:, ji] < config.KP_CONF_MIN) | (conf[:, ai] < config.KP_CONF_MIN) \
            | (conf[:, bi] < config.KP_CONF_MIN)
        ang[bad] = np.nan
        out[name] = ang
    return out


def _kp_xy(kps: np.ndarray, name: str) -> np.ndarray:
    return kps[:, config.KP[name], :2]


def _mid(kps: np.ndarray, a: str, b: str) -> np.ndarray:
    return (_kp_xy(kps, a) + _kp_xy(kps, b)) / 2


def _turning_mask(kps: np.ndarray, fps: float) -> np.ndarray:
    """True on frames where the dancer is turning/spinning (so world-frame
    orientation angles are unreliable and should be excluded).

    Detected two ways: (a) the shoulder line is sweeping fast, (b) the shoulders
    foreshorten (width collapses) because the dancer is facing side/back.
    """
    if len(kps) < 2:                       # np.gradient needs >= 2 samples
        return np.zeros(len(kps), dtype=bool)
    sv = _kp_xy(kps, "right_shoulder") - _kp_xy(kps, "left_shoulder")
    tl = torso_length(kps)
    width = np.linalg.norm(sv, axis=1) / tl
    ang = np.unwrap(np.arctan2(sv[:, 1], sv[:, 0]))
    vel = np.abs(np.gradient(ang)) * fps * 180.0 / np.pi   # deg/s
    turning = (vel > 150.0) | (width < 0.45) | ~np.isfinite(width)
    return turning


def region_rom(track: Track, fps: float) -> dict[str, float]:
    """Range of motion in degrees, grouped by body region.

    Turn/spin frames are excluded from the world-frame regions (shoulders,
    chest, hips) so a spin doesn't masquerade as huge articulation. Ranges use
    the 5th-95th percentile spread (robust to single-frame jitter).

    head      - head flexion relative to the torso (turn-invariant)
    shoulders - tilt of the shoulder line (shrugs / drops / rolls)
    chest     - torso lean & rotation relative to vertical
    hips      - pelvic tilt / hip sway
    arms      - elbow + shoulder joint articulation (both sides)
    legs      - knee + hip joint articulation (both sides)
    """
    kps = track.kps.astype(float)
    conf = kps[:, :, 2]
    sh_mid = _mid(kps, "left_shoulder", "right_shoulder")
    hip_mid = _mid(kps, "left_hip", "right_hip")
    still = ~_turning_mask(kps, fps)   # keep only non-turning frames

    def _range(a: np.ndarray, mask: np.ndarray) -> float:
        a = a[mask & np.isfinite(a)]
        if a.size < 4:
            return 0.0
        return float(np.percentile(a, 95) - np.percentile(a, 5))

    # head: angle at shoulder-mid between nose and hip-mid (body-relative)
    nose = _kp_xy(kps, "nose")
    v1, v2 = nose - sh_mid, hip_mid - sh_mid
    head = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", v1, v2) /
        (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9), -1, 1)))
    head_ok = conf[:, config.KP["nose"]] >= config.KP_CONF_MIN

    # world-frame orientations (turn-excluded). Shoulder/hip lines are folded to
    # a tilt MAGNITUDE (deviation from horizontal, 0-90) so facing front vs. back
    # doesn't flip the angle 180 degrees and inflate the range.
    def _tilt(ang_deg: np.ndarray) -> np.ndarray:
        t = np.mod(ang_deg, 180.0)
        return np.minimum(t, 180.0 - t)

    sv = _kp_xy(kps, "right_shoulder") - _kp_xy(kps, "left_shoulder")
    shoulders = _tilt(np.degrees(np.arctan2(sv[:, 1], sv[:, 0])))
    tv = sh_mid - hip_mid
    chest = np.degrees(np.arctan2(tv[:, 0], -tv[:, 1]))
    hv = _kp_xy(kps, "right_hip") - _kp_xy(kps, "left_hip")
    hips = _tilt(np.degrees(np.arctan2(hv[:, 1], hv[:, 0])))

    angles = angle_series(track, fps)

    def _mean_range(keys):
        vals = []
        for k in keys:
            a = angles[k][np.isfinite(angles[k])]
            if a.size >= 4:
                vals.append(np.percentile(a, 95) - np.percentile(a, 5))
        return float(np.mean(vals)) if vals else 0.0

    return {
        "head": _range(head, head_ok),
        "shoulders": _range(shoulders, still),
        "chest": _range(chest, still),
        "hips": _range(hips, still),
        "arms": _mean_range(["l_elbow", "r_elbow", "l_shoulder", "r_shoulder"]),
        "legs": _mean_range(["l_knee", "r_knee", "l_hip", "r_hip"]),
    }


def speed_series(track: Track, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Whole-body motion energy per frame (scale-normalized). Returns (times, speed)."""
    pos = normalize_positions(track)              # (N,17,2)
    t = _times(track, fps)
    dt = np.diff(t)
    dt[dt <= 0] = 1.0 / fps
    disp = np.linalg.norm(np.diff(pos, axis=0), axis=2)   # (N-1, 17)
    speed = np.nanmean(disp, axis=1) / dt                 # (N-1,)
    tc = (t[:-1] + t[1:]) / 2
    return tc, np.nan_to_num(speed)


def dynamic_range(speed: np.ndarray) -> float:
    """Spread between explosive moves and stillness. Higher = more dynamic."""
    if speed.size == 0:
        return 0.0
    hi = np.percentile(speed, 95)
    lo = np.percentile(speed, 5)
    return float(hi - lo)


def sharpness(times: np.ndarray, speed: np.ndarray) -> float:
    """Attack/freeze crispness = peak deceleration magnitude vs mean speed."""
    if speed.size < 3:
        return 0.0
    dt = np.diff(times)
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    accel = np.abs(np.diff(speed) / dt)
    return float(np.percentile(accel, 95) / (np.mean(speed) + 1e-9))


def explosiveness(times: np.ndarray, speed: np.ndarray) -> float:
    """How fast you launch from still into movement (rate of speed rise out of a
    near-still start to the launch peak). Higher = snappier / more explosive.

    Coarse on low-fps video (a launch can take 1-2 frames), so read small
    differences with caution; high-fps clips resolve it far better.
    """
    if speed.size < 6:
        return 0.0
    lo, hi = np.percentile(speed, 30), np.percentile(speed, 70)
    accels, i, n = [], 1, len(speed)
    while i < n:
        if speed[i - 1] <= lo < speed[i]:        # rising out of near-stillness
            start, j = i - 1, i
            while j + 1 < n and speed[j + 1] >= speed[j]:
                j += 1
            if speed[j] >= hi:                   # a real launch (reaches full motion)
                dt = (times[j] - times[start])
                if dt > 0:
                    accels.append((speed[j] - speed[start]) / dt)
            i = j + 1
        else:
            i += 1
    return float(np.median(accels)) if accels else 0.0


def movement_events(times: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Times of accent onsets = prominent local maxima in motion energy."""
    if speed.size < 3:
        return np.array([])
    prom = max(np.std(speed) * 0.5, 1e-6)
    peaks, _ = find_peaks(speed, prominence=prom, distance=max(1, int(len(speed) * 0.02)))
    return times[peaks]


def timing_offsets(event_times: np.ndarray, grid: BeatGrid) -> dict:
    """How far each accent lands from the nearest 8-count grid marker.

    Negative = early (ahead of the beat), positive = late (behind).
    Returns bias (mean), consistency (std), and per-event offsets in seconds.
    """
    markers = grid.eighth_counts()
    if event_times.size == 0 or markers.size == 0:
        return {"offsets": np.array([]), "bias_ms": 0.0, "std_ms": 0.0, "n": 0}
    idx = np.searchsorted(markers, event_times)
    idx = np.clip(idx, 1, len(markers) - 1)
    left = markers[idx - 1]
    right = markers[idx]
    nearest = np.where(event_times - left < right - event_times, left, right)
    offsets = event_times - nearest  # seconds, signed
    return {
        "offsets": offsets,
        "bias_ms": float(np.mean(offsets) * 1000),
        "std_ms": float(np.std(offsets) * 1000),
        "n": int(offsets.size),
    }


def fluidity_index(speed: np.ndarray) -> float:
    """Continuous/gooey movement vs. hit-and-freeze.

    median / 95th-percentile of motion energy: a fluid dancer keeps motion
    flowing (median close to peak); a staccato dancer spikes then freezes
    (median far below peak). 0 = pure tick/freeze, ~1 = sustained flow.
    """
    if speed.size == 0:
        return 0.0
    peak = np.percentile(speed, 95)
    return float(np.median(speed) / (peak + 1e-9))


def segment_energy(track: Track, fps: float) -> dict:
    """Share of total motion carried by each body segment.

    Lets us measure 'filling up movement' — using head & chest, not just limbs.
    """
    pos = normalize_positions(track)                  # (N,17,2)
    disp = np.linalg.norm(np.diff(pos, axis=0), axis=2)  # (N-1,17)
    per_joint = np.nan_to_num(np.nanmean(disp, axis=0))  # (17,)
    seg = {}
    for name, joints in config.SEGMENTS.items():
        idx = [config.KP[j] for j in joints]
        seg[name] = float(np.mean(per_joint[idx]))
    total = sum(seg.values()) + 1e-9
    shares = {k: v / total for k, v in seg.items()}
    return {
        "share": shares,
        # upper-body 'fill': how much head + chest contribute
        "articulation": float(shares["head"] + shares["chest"]),
    }


def _segment_speed_series(track: Track) -> np.ndarray:
    """Per-frame motion speed for each body segment. Returns (5, N-1) in the
    order of config.SEGMENTS."""
    pos = normalize_positions(track)
    disp = np.linalg.norm(np.diff(pos, axis=0), axis=2)   # (N-1, 17)
    rows = []
    for joints in config.SEGMENTS.values():
        idx = [config.KP[j] for j in joints]
        rows.append(np.nan_to_num(np.nanmean(disp[:, idx], axis=1)))
    return np.array(rows)


def body_engagement(track: Track) -> float:
    """How many body segments move AT THE SAME TIME during movement (0-5).

    Measures 'filling up a move' / engaging multiple body parts at once, rather
    than only the share of upper-body motion. A segment counts as engaged in a
    frame if its speed exceeds 20% of the dancer's peak segment speed.
    """
    seg = _segment_speed_series(track)                  # (5, N-1)
    if seg.size == 0:
        return 0.0
    thr = 0.20 * np.percentile(seg, 99)
    active = seg > thr                                  # (5, N-1)
    total = seg.sum(axis=0)
    moving = total > np.median(total)                   # only count moving frames
    if not moving.any():
        return 0.0
    return float(np.mean(active[:, moving].sum(axis=0)))


def hip_articulation(track: Track, fps: float) -> float:
    """Lateral hip movement relative to the torso (sway / twerk / weight-shift).

    The main metrics pelvis-center every frame, which mathematically ERASES this
    motion. Here we measure the hips moving *under a stable torso* (hip-mid minus
    shoulder-mid, scale-normalized), on non-turn frames. Higher = more hip work.
    """
    kps = track.kps.astype(float)
    if len(kps) < 5:
        return 0.0
    still = ~_turning_mask(kps, fps)
    tl = torso_length(kps)
    shm = _mid(kps, "left_shoulder", "right_shoulder")
    hm = _mid(kps, "left_hip", "right_hip")
    sway = (hm[:, 0] - shm[:, 0]) / tl
    s = sway[still]
    s = s[np.isfinite(s)]
    return float(np.std(s)) if s.size >= 4 else 0.0


def _centroid_segment_motion(track: Track) -> dict:
    """Per-segment motion measured relative to the WHOLE-BODY centroid (not the
    pelvis), so hip motion isn't zeroed out. Comparable units across segments."""
    kps = track.kps[:, :, :2].astype(float)
    centroid = np.nanmean(kps, axis=1, keepdims=True)
    tl = torso_length(track.kps.astype(float))[:, None, None]
    pos = (kps - centroid) / tl
    disp = np.linalg.norm(np.diff(pos, axis=0), axis=2)        # (N-1, 17)
    out = {}
    for name, joints in config.SEGMENTS.items():
        idx = [config.KP[j] for j in joints]
        out[name] = float(np.nan_to_num(np.nanmean(disp[:, idx])))
    return out


def movement_distribution(track: Track) -> dict:
    """Where the movement lives: share of motion per segment, plus the fraction
    coming from the CORE (head + chest + hips) vs. the limbs (arms + legs).

    Low core_share = limb-dominant ('marking with feet/arms'); high = dancing
    from the center, which is what reads as advanced in commercial/street styles.
    """
    seg = _centroid_segment_motion(track)
    total = sum(seg.values()) + 1e-9
    shares = {k: v / total for k, v in seg.items()}
    core = shares["head"] + shares["chest"] + shares["hips"]
    return {"share": shares, "core_share": float(core)}


def groove(track: Track, fps: float, grid: BeatGrid) -> dict:
    """Bounce / weight-shift: vertical oscillation of the body and how well it
    locks to the beat. Project A 'find the weight shifts / grooves'.
    """
    kps = track.kps.astype(float)
    lh, rh = config.KP["left_hip"], config.KP["right_hip"]
    y = (kps[:, lh, 1] + kps[:, rh, 1]) / 2
    scale = torso_length(kps)
    yn = y / scale
    t = _times(track, fps)
    good = np.isfinite(yn)
    if good.sum() < 16:
        return {"strength": 0.0, "beat_lock": 0.0}
    # uniform resample then detrend
    ug = np.linspace(t[good][0], t[good][-1], int((t[good][-1] - t[good][0]) * fps))
    yr = np.interp(ug, t[good], yn[good])
    yr = yr - np.polyval(np.polyfit(np.arange(len(yr)), yr, 1), np.arange(len(yr)))
    strength = float(np.std(yr))
    # beat_lock = fraction of the bounce's POWER that sits at the beat frequency
    # (and its half/double). Earlier this used the single dominant frequency, which
    # slow vertical drift (level changes/posture) always hijacks — so it read ~0 for
    # everyone. Band-power is what actually answers "do you bounce on the tempo".
    power = np.abs(np.fft.rfft(yr * np.hanning(len(yr)))) ** 2
    freqs = np.fft.rfftfreq(len(yr), d=1.0 / fps)
    beat_hz = grid.tempo / 60.0

    def _band(f0):
        return power[(freqs > 0.8 * f0) & (freqs < 1.2 * f0)].sum()

    non_drift = power[freqs > 0.3].sum() + 1e-9   # ignore slow drift in the total
    beat_power = _band(beat_hz) + _band(0.5 * beat_hz) + _band(2.0 * beat_hz)
    beat_lock = float(min(1.0, beat_power / non_drift))
    return {"strength": strength, "beat_lock": beat_lock}


def picture_catching(me: Track, ref: Track, fps: float) -> dict | None:
    """Do you hit and HOLD the shapes the reference holds?

    The reference's stillpoints (low-velocity frames) are the 'pictures' the
    choreography calls for. We check how still YOU are at those same frames.
    Moving fast during the reference's holds = you're blowing through the pictures.
    """
    tm, sm = speed_series(me, fps)
    tr, sr = speed_series(ref, fps)
    if tm.size < 10 or tr.size < 10:
        return None
    fm = {int(round(t * fps)): s for t, s in zip(tm, sm)}
    fr = {int(round(t * fps)): s for t, s in zip(tr, sr)}
    common = sorted(set(fm) & set(fr))
    if len(common) < 20:
        return None
    msp = np.array([fm[f] for f in common])
    rsp = np.array([fr[f] for f in common])
    hold = rsp <= np.percentile(rsp, 20)
    if hold.sum() < 3:
        return None
    ref_hold = float(rsp[hold].mean())
    you_hold = float(msp[hold].mean())
    return {
        "ratio": you_hold / (ref_hold + 1e-9),      # you move Nx faster during ref's holds
        "corr": float(np.corrcoef(msp, rsp)[0, 1]),  # stillpoint match (1=perfect, 0=none)
        "n_holds": int(hold.sum()),
    }


def groove_timing(me: Track, ref: Track, fps: float) -> dict | None:
    """Is your bounce EARLY or LATE vs. the reference?

    Cross-correlates the two vertical hip-bounce signals and reports the lag.
    Reference-anchored (same clip), so it cancels the beat-detector's offset and
    is the reliable way to say 'you hit the down a hair before/after the reference'.
    Negative = you lead (early); positive = you trail (late).
    """
    def vbounce(track):
        kps = track.kps.astype(float)
        tl = torso_length(kps)
        y = (kps[:, config.KP["left_hip"], 1] + kps[:, config.KP["right_hip"], 1]) / 2 / tl
        return track.frames / fps, y

    tm, ym = vbounce(me)
    tr, yr = vbounce(ref)
    lo, hi = max(tm.min(), tr.min()), min(tm.max(), tr.max())
    n = int((hi - lo) * fps)
    if n < int(fps):
        return None
    g = np.linspace(lo, hi, n)
    idx = np.arange(n)
    a = np.interp(g, tm, ym)
    b = np.interp(g, tr, yr)
    a = a - np.polyval(np.polyfit(idx, a, 2), idx)   # detrend slow drift
    b = b - np.polyval(np.polyfit(idx, b, 2), idx)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    maxlag = int(0.5 * fps)
    best, best_c = 0, -2.0
    for lag in range(-maxlag, maxlag + 1):
        if lag < 0:
            x, z = a[:lag], b[-lag:]
        elif lag > 0:
            x, z = a[lag:], b[:-lag]
        else:
            x, z = a, b
        if len(x) < 5:
            continue
        c = float(np.dot(x, z) / len(x))
        if c > best_c:
            best_c, best = c, lag
    return {"lag_ms": best / fps * 1000.0, "corr": float(np.clip(best_c, -1, 1))}


def sync_between(me: Track, ref: Track, fps: float) -> dict:
    """Frame-aligned sync of two tracks in the SAME clip.

    Returns best lag (frames/ms) and correlation of their motion-energy signals,
    plus mean absolute joint-angle difference over shared frames.
    """
    tm, sm = speed_series(me, fps)
    tr, sr = speed_series(ref, fps)
    # resample both onto a shared dense time grid
    t0 = max(tm.min(), tr.min())
    t1 = min(tm.max(), tr.max())
    if t1 <= t0:
        return {"lag_ms": None, "corr": None, "angle_diff_deg": None}
    n = int((t1 - t0) * fps)
    if n < 4:
        return {"lag_ms": None, "corr": None, "angle_diff_deg": None}
    grid = np.linspace(t0, t1, n)
    gm = (np.interp(grid, tm, sm) - 0)
    gr = np.interp(grid, tr, sr)
    gm = (gm - gm.mean()) / (gm.std() + 1e-9)
    gr = (gr - gr.mean()) / (gr.std() + 1e-9)
    # search only physically plausible lags (±0.5s), and normalize each lag by its
    # actual overlap count so the score is a real correlation (not biased toward 0).
    max_lag = min(n - 1, int(0.5 * fps))
    best_lag, best_corr = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = gm[:lag], gr[-lag:]
        elif lag > 0:
            a, b = gm[lag:], gr[:-lag]
        else:
            a, b = gm, gr
        if len(a) < 2:
            continue
        c = float(np.dot(a, b) / len(a))
        if c > best_corr:
            best_corr, best_lag = c, lag
    lag_ms = best_lag / fps * 1000
    peak_corr = float(np.clip(best_corr, -1.0, 1.0))

    # angle agreement on the overlap
    am = angle_series(me, fps)
    ar = angle_series(ref, fps)
    diffs = []
    for k in am:
        a = np.interp(grid, _times(me, fps), np.nan_to_num(am[k], nan=np.nanmean(am[k])))
        b = np.interp(grid, _times(ref, fps), np.nan_to_num(ar[k], nan=np.nanmean(ar[k])))
        diffs.append(np.abs(a - b))
    angle_diff = float(np.mean(diffs)) if diffs else None

    return {"lag_ms": float(lag_ms), "corr": peak_corr, "angle_diff_deg": angle_diff}


def pocket(track: Track, fps: float, grid: BeatGrid) -> dict:
    """Rough estimate of when movement INITIATES relative to the beat.

    Detects the rising edge of each movement burst and measures its signed offset
    to the nearest beat. Positive = you start AFTER the beat ('in the pocket' /
    patient); negative = you start BEFORE the beat (anticipating / rushing).

    Rough: limited by frame rate (~33 ms at 30fps) and beat detection. The
    me-vs-reference difference is far more reliable than the absolute value.
    """
    t, s = speed_series(track, fps)
    if t.size < 4 or len(grid.beats) < 2:
        return {"ms": 0.0, "std_ms": 0.0, "n": 0}
    hi, lo = np.percentile(s, 65), np.percentile(s, 35)
    onsets, armed = [], True
    for i in range(len(s)):
        if armed and s[i] >= hi:
            onsets.append(t[i])
            armed = False
        elif not armed and s[i] <= lo:
            armed = True
    if not onsets:
        return {"ms": 0.0, "std_ms": 0.0, "n": 0}
    beats = grid.beats
    offs = np.array([o - beats[np.argmin(np.abs(beats - o))] for o in onsets])
    return {"ms": float(np.mean(offs) * 1000),
            "std_ms": float(np.std(offs) * 1000), "n": len(offs)}


def _mmss(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def stance_spread(track: Track) -> np.ndarray:
    """Horizontal ankle separation / torso length. Bigger = wider stance."""
    kps = track.kps.astype(float)
    la, ra = config.KP["left_ankle"], config.KP["right_ankle"]
    return np.abs(kps[:, ra, 0] - kps[:, la, 0]) / torso_length(kps)


def moments(me: Track, ref: Track | None, fps: float) -> dict:
    """Specific timestamps to go watch, illustrating each gap. `ref` may be None
    (e.g. comparing against a team profile) — reference-relative moments are skipped."""
    out: dict[str, object] = {}

    # leg extension: where the reference's stance most exceeds yours
    cf = np.intersect1d(me.frames, ref.frames) if ref is not None else np.array([])
    if cf.size:
        mi = {f: i for i, f in enumerate(me.frames)}
        ri = {f: i for i, f in enumerate(ref.frames)}
        ms, rs = stance_spread(me), stance_spread(ref)
        best, bf = -np.inf, None
        for f in cf:
            a, b = rs[ri[f]], ms[mi[f]]
            if np.isfinite(a) and np.isfinite(b) and a - b > best:
                best, bf = a - b, f
        if bf is not None:
            out["leg_extension"] = _mmss(bf / fps)

    # fluidity gaps: reference is flowing while you've frozen
    tm, sm = speed_series(me, fps)
    tr, sr = speed_series(ref, fps) if ref is not None else (np.array([]), np.array([]))
    if tm.size and tr.size:
        t0, t1 = max(tm.min(), tr.min()), min(tm.max(), tr.max())
        if t1 > t0:
            g = np.linspace(t0, t1, int((t1 - t0) * fps))
            gm, gr = np.interp(g, tm, sm), np.interp(g, tr, sr)
            gap = (gr > np.percentile(gr, 60)) & (gm < np.percentile(gm, 30))
            # collapse contiguous runs, keep the longest few
            runs, i = [], 0
            while i < len(gap):
                if gap[i]:
                    j = i
                    while j < len(gap) and gap[j]:
                        j += 1
                    runs.append((j - i, g[i]))
                    i = j
                else:
                    i += 1
            runs.sort(reverse=True)               # longest gaps first
            top = sorted(runs[:3], key=lambda r: r[1])   # then chronological
            out["fluidity_gaps"] = [_mmss(s) for _, s in top]

    # groove: longest steady moderate-motion stretch (where to feel weight shifts)
    if tm.size > int(4 * fps):
        win = int(4 * fps)
        best = None
        for st in range(0, len(sm) - win, int(fps)):
            seg = sm[st:st + win]
            if np.mean(seg) > np.median(sm):
                score = np.median(seg) / (np.percentile(seg, 95) + 1e-9)
                if best is None or score > best[0]:
                    best = (score, tm[st], tm[st + win])
        if best:
            out["groove_section"] = f"{_mmss(best[1])}-{_mmss(best[2])}"

    # your sharpest hit (a strength to reference)
    ev = movement_events(tm, sm)
    if ev.size:
        idx = [int(np.argmin(np.abs(tm - e))) for e in ev]
        out["your_sharp_hit"] = _mmss(ev[int(np.argmax([sm[i] for i in idx]))])

    return out


def profile_track(track: Track, fps: float, grid: BeatGrid) -> dict:
    """Full single-dancer metric profile."""
    t, s = speed_series(track, fps)
    events = movement_events(t, s)
    timing = timing_offsets(events, grid)
    angles = angle_series(track, fps)
    rom = {k: float(np.nanmax(v) - np.nanmin(v)) for k, v in angles.items()
           if np.isfinite(np.nanmax(v))}
    # left/right symmetry: compare paired ROM
    sym = {}
    for base in ("elbow", "shoulder", "knee", "hip"):
        l, r = rom.get(f"l_{base}"), rom.get(f"r_{base}")
        if l and r:
            sym[base] = float(abs(l - r) / (max(l, r) + 1e-9))
    seg = segment_energy(track, fps)
    grv = groove(track, fps, grid)
    pkt = pocket(track, fps, grid)
    dist = movement_distribution(track)
    return {
        "n_frames": int(len(track.frames)),
        "seconds": float(len(track.frames) / fps),
        "dynamic_range": dynamic_range(s),
        "sharpness": sharpness(t, s),
        "explosiveness": explosiveness(t, s),
        "fluidity": fluidity_index(s),
        "groove_strength": grv["strength"],
        "groove_beat_lock": grv["beat_lock"],
        "articulation": seg["articulation"],
        "segment_share": seg["share"],
        "engagement": body_engagement(track),
        "hip_articulation": hip_articulation(track, fps),
        "core_share": dist["core_share"],
        "segment_motion": dist["share"],
        "n_accents": int(events.size),
        "timing_bias_ms": timing["bias_ms"],
        "timing_std_ms": timing["std_ms"],
        "pocket_ms": pkt["ms"],
        "pocket_std_ms": pkt["std_ms"],
        "rom_deg": rom,
        "rom_region": region_rom(track, fps),
        "symmetry_imbalance": sym,
    }
