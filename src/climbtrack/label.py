"""Minimal labelling: a terminal stopwatch for climb start/stop + tap-sync alignment.

Protocol (docs/DATA_COLLECTION.md): start the IMU recording, then start
``climbtrack label``; tap the watch face hard 3 times while pressing ``s``.
Press Enter at the start and at the end of every climb (feet off / feet back
on the ground). Afterwards ``climbtrack align`` finds the 3 taps in the IMU
signal and shifts the labels onto the recording clock.
"""

from __future__ import annotations

import csv
import sys
import time

import numpy as np

from . import preprocess as pp


def run_labeller(out_path: str, clock=time.time, inp=sys.stdin, out=sys.stdout) -> list[tuple[float, float]]:
    out.write(
        "climbtrack label — keys: [Enter] toggle climb start/stop · s+Enter mark 3-tap sync · "
        "u+Enter undo last · q+Enter quit\n"
    )
    t_sync, start, climbs = None, None, []
    for line in inp:
        cmd, now = line.strip().lower(), clock()
        if cmd == "q":
            break
        if cmd == "s":
            t_sync = now
            out.write(f"  sync marked at {time.strftime('%H:%M:%S', time.localtime(now))}\n")
        elif cmd == "u":
            if start is not None:
                start = None
                out.write("  open climb discarded\n")
            elif climbs:
                climbs.pop()
                out.write("  last climb removed\n")
        elif cmd == "":
            if start is None:
                start = now
                out.write(f"  CLIMB {len(climbs) + 1} start\n")
            else:
                climbs.append((start, now))
                out.write(f"  CLIMB {len(climbs)} end ({now - start:.0f} s)\n")
                start = None
        out.flush()
    ref = t_sync if t_sync is not None else (climbs[0][0] if climbs else clock())
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "label", "clock"])
        w.writerow([0.0, 0.0, "sync" if t_sync is not None else "nosync", f"{ref:.3f}"])
        for a, b in climbs:
            w.writerow([f"{a - ref:.3f}", f"{b - ref:.3f}", "climb", ""])
    out.write(f"saved {len(climbs)} climbs -> {out_path}\n")
    return [(a - ref, b - ref) for a, b in climbs]


def find_sync_taps(acc: np.ndarray, fs: float, search_s: float = 300.0, thr_g: float = 2.0,
                   n_taps: int = 3, max_span_s: float = 3.0) -> float | None:
    """Time (s) of the first tap of the first burst of ``n_taps`` spikes above
    ``thr_g`` g (dynamic magnitude) within ``max_span_s``; None if not found."""
    from scipy.signal import find_peaks

    n = min(len(acc), int(search_s * fs))
    d = pp.dyn_magnitude(acc[:n])
    pk, _ = find_peaks(d, height=thr_g * pp.G, distance=max(1, int(0.12 * fs)))
    for i in range(len(pk) - n_taps + 1):
        if (pk[i + n_taps - 1] - pk[i]) / fs <= max_span_s:
            return pk[i] / fs
    return None


def align_labels(labels_path: str, acc: np.ndarray, fs: float, out_path: str, offset_s: float | None = None) -> float:
    """Rewrite labels relative to the IMU clock. ``offset_s`` overrides tap detection."""
    rows = list(csv.DictReader(open(labels_path)))
    if offset_s is None:
        if not rows or rows[0]["label"] != "sync":
            raise ValueError("labels have no sync mark; pass --offset seconds")
        offset_s = find_sync_taps(acc, fs)
        if offset_s is None:
            raise ValueError("3-tap sync not found in the first 5 min of IMU data; pass --offset")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "label"])
        for r in rows:
            if r["label"] == "climb":
                w.writerow([f"{float(r['start_s']) + offset_s:.3f}", f"{float(r['end_s']) + offset_s:.3f}", "climb"])
    return offset_s
