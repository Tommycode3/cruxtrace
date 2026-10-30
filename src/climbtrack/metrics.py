"""Per-climb movement metrics: pauses, moves, smoothness (LDLJ, SPARC), explosiveness."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import preprocess as pp


# ------------------------------------------------------------------ smoothness
def sparc(speed: np.ndarray, fs: float, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral arc length (Balasubramanian et al. 2012, 2015).

    Follows the reference implementation: zero-padded magnitude spectrum,
    normalised to its maximum, adaptive cut-off = last frequency <= fc where the
    normalised magnitude is >= amp_th, arc length of the curve with the
    frequency axis normalised to [0, 1]. More negative = less smooth.
    """
    x = np.asarray(speed, float)
    if len(x) < 4 or not np.any(x):
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(x))) + padlevel))
    f = np.arange(nfft) * fs / nfft
    M = np.abs(np.fft.fft(x, nfft))
    M = M / M.max()
    sel = f <= fc
    f_sel, M_sel = f[sel], M[sel]
    idx = np.flatnonzero(M_sel >= amp_th)
    f_sel, M_sel = f_sel[idx[0] : idx[-1] + 1], M_sel[idx[0] : idx[-1] + 1]
    if len(f_sel) < 2:
        return 0.0
    return float(-np.sum(np.sqrt((np.diff(f_sel) / (f_sel[-1] - f_sel[0])) ** 2 + np.diff(M_sel) ** 2)))


def ldlj_v(vel: np.ndarray, fs: float) -> float:
    """Log dimensionless jerk from velocity (Hogan & Sternad 2009; Balasubramanian 2015):
    LDLJ = -ln( T^3 / v_peak^2 * integral ||j||^2 dt ). Minimum-jerk reach = -ln(720/1.875^2) ~ -5.32."""
    v = np.atleast_2d(np.asarray(vel, float).T).T
    T = (len(v) - 1) / fs
    vpk = np.max(np.linalg.norm(v, axis=1))
    j = np.gradient(np.gradient(v, 1 / fs, axis=0), 1 / fs, axis=0)
    integ = np.trapezoid(np.sum(j**2, axis=1), dx=1 / fs)
    return float(-np.log(T**3 / vpk**2 * integ))


def ldlj_a(acc: np.ndarray, fs: float) -> float:
    """LDLJ from acceleration, the IMU-friendly variant (Melendez-Calderon et al. 2021):
    LDLJ_A = -ln( T / a_peak^2 * integral ||da/dt||^2 dt ). Input should be
    gravity-free acceleration with its mean removed."""
    a = np.atleast_2d(np.asarray(acc, float).T).T
    a = a - a.mean(axis=0)
    T = (len(a) - 1) / fs
    apk = np.max(np.linalg.norm(a, axis=1))
    if apk <= 0:
        return float("nan")
    j = np.gradient(a, 1 / fs, axis=0)
    integ = np.trapezoid(np.sum(j**2, axis=1), dx=1 / fs)
    return float(-np.log(T / apk**2 * integ))


# ------------------------------------------------------------------ per climb
@dataclass
class ClimbMetrics:
    start_s: float
    end_s: float
    duration_s: float
    n_moves: int
    moves_per_min: float
    n_pauses: int
    pause_time_s: float
    pause_frac: float
    longest_pause_s: float
    ldlj_a_median: float
    sparc_median: float
    peak_acc_ms2: float
    p95_jerk_ms3: float
    n_bursts: int  # dynamic-move proxies

    def as_dict(self):
        return asdict(self)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.r_[False, mask, False].astype(np.int8)
    e = np.flatnonzero(np.diff(m))
    return list(zip(e[::2], e[1::2]))


def climb_metrics(acc, fs, start, end, gyr=None, ghat=None, pause_min_s: float = 2.0,
                  still_acc: float = 0.6, still_rot: float = 0.35, burst_acc: float = 15.0) -> ClimbMetrics:
    """Metrics for one climb interval [start, end) (s).

    * motion activity a(t) = ||a_lin|| low-passed at 3 Hz; rotation r(t) = ||omega||
      (gyro) or gravity tilt rate (accelerometer only).
    * pause = run >= pause_min_s with a(t) < still_acc AND r(t) < still_rot.
    * moves = active runs between stills, >= 0.3 s; smoothness computed per move.
    * bursts = local maxima of ||a_lin|| above burst_acc (~1.5 g), >= 0.5 s apart.
    """
    i0, i1 = int(start * fs), int(end * fs)
    a = acc[i0:i1]
    g = (ghat if ghat is not None else pp.gravity(acc, fs, gyr))[i0:i1]
    lin = pp.linear_acc(a, g)
    lin_n = np.linalg.norm(lin, axis=1)
    act = pp.lowpass(lin_n, fs, 3.0) if len(lin_n) > 30 else lin_n
    rot = np.linalg.norm(gyr[i0:i1], axis=1) if gyr is not None else pp.tilt_rate(g, fs)
    still = (act < still_acc) & (rot < still_rot)
    pauses = [(s, e) for s, e in _runs(still) if (e - s) / fs >= pause_min_s]
    moves = [(s, e) for s, e in _runs(~still) if (e - s) / fs >= 0.3]
    ld, sp = [], []
    for s, e in moves:
        ld.append(ldlj_a(lin[s:e], fs))
        sp.append(sparc(rot[s:e] if gyr is not None else act[s:e], fs))
    jerk = np.linalg.norm(np.gradient(pp.lowpass(lin, fs, 10.0) if len(lin) > 30 else lin, 1 / fs, axis=0), axis=1)
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(lin_n, height=burst_acc, distance=max(1, int(0.5 * fs)))
    dur = (i1 - i0) / fs
    pt = sum(e - s for s, e in pauses) / fs
    return ClimbMetrics(
        start_s=round(start, 2), end_s=round(end, 2), duration_s=round(dur, 2),
        n_moves=len(moves), moves_per_min=round(60 * len(moves) / dur, 2) if dur else 0.0,
        n_pauses=len(pauses), pause_time_s=round(pt, 2), pause_frac=round(pt / dur, 3) if dur else 0.0,
        longest_pause_s=round(max([(e - s) / fs for s, e in pauses], default=0.0), 2),
        ldlj_a_median=round(float(np.nanmedian(ld)), 3) if ld else float("nan"),
        sparc_median=round(float(np.nanmedian(sp)), 3) if sp else float("nan"),
        peak_acc_ms2=round(float(lin_n.max()), 2) if len(lin_n) else 0.0,
        p95_jerk_ms3=round(float(np.percentile(jerk, 95)), 1) if len(jerk) else 0.0,
        n_bursts=len(peaks),
    )


@dataclass
class SessionSummary:
    duration_min: float
    n_climbs: int
    time_on_wall_s: float
    wall_fraction: float
    median_climb_s: float
    median_rest_s: float
    work_rest_ratio: float

    def as_dict(self):
        return asdict(self)


def session_summary(climbs, duration_s: float) -> SessionSummary:
    durs = [b - a for a, b in climbs]
    rests = [climbs[k + 1][0] - climbs[k][1] for k in range(len(climbs) - 1)]
    tow = float(sum(durs))
    return SessionSummary(
        duration_min=round(duration_s / 60, 1), n_climbs=len(climbs), time_on_wall_s=round(tow, 1),
        wall_fraction=round(tow / duration_s, 3) if duration_s else 0.0,
        median_climb_s=round(float(np.median(durs)), 1) if durs else 0.0,
        median_rest_s=round(float(np.median(rests)), 1) if rests else 0.0,
        work_rest_ratio=round(float(np.sum(durs) / np.sum(rests)), 2) if rests else float("nan"),
    )
