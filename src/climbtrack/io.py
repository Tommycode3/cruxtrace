"""Loading IMU recordings: generic CSV and Garmin/COROS-style FIT files."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

G = 9.80665

_T_KEYS = ("t", "time", "timestamp", "seconds_elapsed", "time_s", "loggingtime(txt)")
_ACC_KEYS = [
    ("ax", "ay", "az"),
    ("acc_x", "acc_y", "acc_z"),
    ("accelerometerx", "accelerometery", "accelerometerz"),
    ("accelerometeraccelerationx(g)", "accelerometeraccelerationy(g)", "accelerometeraccelerationz(g)"),
    ("x", "y", "z"),
]
_GYR_KEYS = [
    ("gx", "gy", "gz"),
    ("gyr_x", "gyr_y", "gyr_z"),
    ("gyro_x", "gyro_y", "gyro_z"),
    ("gyroscopex", "gyroscopey", "gyroscopez"),
    ("gyrorotationx(rad/s)", "gyrorotationy(rad/s)", "gyrorotationz(rad/s)"),
]


@dataclass
class Recording:
    fs: float
    t: np.ndarray  # s, uniform, starts at 0
    acc: np.ndarray  # (N,3) m/s^2
    gyr: np.ndarray | None  # (N,3) rad/s or None
    t0: float = 0.0  # original start time (s, source clock)
    source: str = ""


def _find(cols: list[str], options) -> list[int] | None:
    low = [c.strip().lower() for c in cols]
    for opt in options:
        if all(o in low for o in opt):
            return [low.index(o) for o in opt]
    return None


def _to_seconds(t: np.ndarray) -> np.ndarray:
    t = t - t[0]
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    if dt > 1e6:  # ns
        return t / 1e9
    if dt > 0.9:  # ms (assume >= ~1 Hz is never the case for IMU data)
        return t / 1e3
    return t


def resample(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear resampling onto a uniform grid (IMU exports are often jittery)."""
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    keep = np.r_[True, np.diff(t) > 0]
    t, x = t[keep], x[keep]
    tu = np.arange(0.0, t[-1], 1.0 / fs)
    return tu, np.column_stack([np.interp(tu, t, x[:, k]) for k in range(x.shape[1])])


def load_csv(path: str, fs: float | None = None, acc_unit: str = "auto") -> Recording:
    """Load a CSV with a time column and 3 accelerometer columns (gyro optional).

    Time may be in s, ms or ns (auto-detected from the median step). Acceleration
    unit is auto-detected: if the median |a| is ~1 the data are in g.
    """
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    data = data[~np.isnan(data).any(axis=1)] if data.ndim == 2 else data
    ti = _find(header, [(k,) for k in _T_KEYS])
    ai = _find(header, _ACC_KEYS)
    if ti is None or ai is None:
        raise ValueError(f"{path}: need a time column {_T_KEYS} and accelerometer columns, got {header}")
    gi = _find(header, _GYR_KEYS)
    t = _to_seconds(data[:, ti[0]])
    acc = data[:, ai]
    if acc_unit == "g" or (acc_unit == "auto" and np.median(np.linalg.norm(acc, axis=1)) < 3.0):
        acc = acc * G
    elif acc_unit == "mg":
        acc = acc * G / 1000
    x = acc if gi is None else np.column_stack([acc, data[:, gi]])
    if gi is not None and np.nanpercentile(np.abs(data[:, gi]), 99) > 40:  # deg/s
        x[:, 3:] = np.deg2rad(x[:, 3:])
    if fs is None:
        fs = float(np.round(1.0 / np.median(np.diff(t))))
    tu, xu = resample(t, x, fs)
    return Recording(fs, tu, xu[:, :3], None if gi is None else xu[:, 3:], float(data[0, ti[0]]), path)


def load_fit(path: str, fs: float = 25.0) -> Recording:
    """Load raw accelerometer (and gyroscope) messages from a .FIT file.

    Requires the optional dependency ``fitdecode``. Only devices/apps that log
    ``accelerometer_data`` (FIT global msg 165) will work; most watches store only
    summary data unless a developer/raw-data logging app is used.
    """
    try:
        import fitdecode
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'climbtrack[fit]' (fitdecode) to read FIT files") from e

    ts, acc, gts, gyr = [], [], [], []
    with fitdecode.FitReader(path) as fr:
        for frame in fr:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name not in ("accelerometer_data", "gyroscope_data"):
                continue
            base = frame.get_value("timestamp").timestamp()
            ms = frame.get_value("timestamp_ms", fallback=0) or 0
            off = frame.get_value("sample_time_offset", fallback=None)
            if off is None:
                continue
            off = np.atleast_1d(off).astype(float)
            t = base + ms / 1000 + off / 1000
            if frame.name == "accelerometer_data":
                x = [np.atleast_1d(frame.get_value(f"calibrated_accel_{k}", fallback=None)) for k in "xyz"]
                if x[0][0] is None:
                    x = [np.atleast_1d(frame.get_value(f"accel_{k}")) for k in "xyz"]
                ts.append(t)
                acc.append(np.column_stack(x).astype(float))
            else:
                x = [np.atleast_1d(frame.get_value(f"calibrated_gyro_{k}", fallback=None)) for k in "xyz"]
                if x[0][0] is None:
                    continue
                gts.append(t)
                gyr.append(np.column_stack(x).astype(float))
    if not acc:
        raise ValueError(f"{path}: no accelerometer_data messages (device did not log raw IMU)")
    t = np.concatenate(ts)
    a = np.concatenate(acc)
    scale = G / 1000 if np.median(np.linalg.norm(a, axis=1)) > 100 else (G if np.median(np.linalg.norm(a, axis=1)) < 3 else 1.0)
    a = a * scale
    t0 = t.min()
    tu, au = resample(t - t0, a, fs)
    g = None
    if gyr:
        gt = np.concatenate(gts) - t0
        gv = np.deg2rad(np.concatenate(gyr))  # FIT calibrated gyro is deg/s
        gtu, gvu = resample(gt, gv, fs)
        g = np.column_stack([np.interp(tu, gtu, gvu[:, k]) for k in range(3)])
    return Recording(fs, tu, au, g, float(t0), path)


def load(path: str, fs: float | None = None) -> Recording:
    if path.lower().endswith(".fit"):
        return load_fit(path, fs or 25.0)
    return load_csv(path, fs)


def load_labels(path: str) -> list[tuple[float, float]]:
    """Read a labels CSV (start_s,end_s[,label]); only 'climb' rows (or unlabelled) are kept."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("label", "climb").strip().lower() in ("climb", ""):
                out.append((float(row["start_s"]), float(row["end_s"])))
    return sorted(out)
