"""Gravity estimation and orientation-free signals."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfiltfilt

G = 9.80665


def lowpass(x: np.ndarray, fs: float, fc: float, order: int = 2) -> np.ndarray:
    sos = butter(order, fc, fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def gravity_lowpass(acc: np.ndarray, fs: float, fc: float = 0.3) -> np.ndarray:
    """Gravity unit vector in the sensor frame from a zero-phase low-pass of the
    accelerometer (accelerometer-only fallback)."""
    g = lowpass(acc, fs, fc)
    return g / np.linalg.norm(g, axis=1, keepdims=True).clip(1e-9)


def gravity_complementary(acc: np.ndarray, gyr: np.ndarray, fs: float, tau: float = 1.0) -> np.ndarray:
    """First-order complementary filter on the gravity direction.

    Propagate with the gyroscope (dg/dt = g x omega in the body frame), then pull
    towards the normalised accelerometer with gain (1-alpha), alpha = tau/(tau+dt).
    """
    dt = 1.0 / fs
    alpha = tau / (tau + dt)
    an = acc / np.linalg.norm(acc, axis=1, keepdims=True).clip(1e-9)
    g0 = an[: max(1, int(fs))].mean(axis=0)
    x, y, z = (g0 / np.linalg.norm(g0)).tolist()
    out = np.empty_like(acc)
    A, W = an.tolist(), (gyr * dt).tolist()
    for k in range(len(A)):
        wx, wy, wz = W[k]
        # first-order rotation of g by -w*dt: g <- g + g x (w dt)
        x, y, z = x + (y * wz - z * wy), y + (z * wx - x * wz), z + (x * wy - y * wx)
        ax, ay, az = A[k]
        x, y, z = alpha * x + (1 - alpha) * ax, alpha * y + (1 - alpha) * ay, alpha * z + (1 - alpha) * az
        n = math.sqrt(x * x + y * y + z * z)
        x, y, z = x / n, y / n, z / n
        out[k] = (x, y, z)
    return out


def gravity(acc: np.ndarray, fs: float, gyr: np.ndarray | None = None) -> np.ndarray:
    return gravity_complementary(acc, gyr, fs) if gyr is not None else gravity_lowpass(acc, fs)


def linear_acc(acc: np.ndarray, ghat: np.ndarray) -> np.ndarray:
    """Gravity-removed acceleration a_lin = a - G * g_hat (sensor frame)."""
    return acc - G * ghat


def dyn_magnitude(acc: np.ndarray) -> np.ndarray:
    """Orientation-free dynamic acceleration | ||a|| - G |."""
    return np.abs(np.linalg.norm(acc, axis=1) - G)


def tilt_rate(ghat: np.ndarray, fs: float, lag_s: float = 0.2) -> np.ndarray:
    """Angular speed of the gravity direction (rad/s), mounting-invariant."""
    L = max(1, int(round(lag_s * fs)))
    d = np.clip(np.einsum("ij,ij->i", ghat[L:], ghat[:-L]), -1, 1)
    r = np.arccos(d) / (L / fs)
    return np.r_[r, np.full(L, r[-1] if len(r) else 0.0)]


def elevation(ghat: np.ndarray, axis: int = 0, sign: float | None = None) -> tuple[np.ndarray, float]:
    """Forearm elevation above horizontal (rad).

    With the sensor at rest f = R^T (0,0,G), so the forearm axis component of
    g_hat is sin(elevation). The forearm axis sign depends on which wrist the
    watch is on; if ``sign`` is None it is chosen so that the session median
    elevation is negative (people spend most of a session with arms down).
    """
    e = np.arcsin(np.clip(ghat[:, axis], -1, 1))
    if sign is None:
        sign = -1.0 if np.median(e) > 0 else 1.0
    return sign * e, sign
