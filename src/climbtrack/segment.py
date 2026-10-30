"""Climb / not-climb segmentation: window features + sticky 2-state Gaussian HMM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from . import preprocess as pp

BASE_FEATURES = ("elev_mean", "elev_up_frac", "tilt_rate", "log_dyn_rms", "periodicity", "orient_spread")
# Context features: the same quantities averaged over ~CONTEXT_S seconds. A climb is a
# minutes-long *regime*; holds and shake-outs inside it look like rest at 4 s scale.
CONTEXT = ("elev_up_frac", "tilt_rate", "log_dyn_rms", "periodicity")
HMM_FEATURES = ("elev_up_frac_ctx", "tilt_rate_ctx", "log_dyn_rms_ctx")
WALK_PERIODICITY = 0.5
CONTEXT_S = 15.0
UP_DEG = 30.0  # "hand above shoulder-ish": forearm elevation threshold
FEATURES = BASE_FEATURES + tuple(f"{c}_ctx" for c in CONTEXT)


@dataclass
class Windows:
    centers: np.ndarray  # s
    X: np.ndarray  # (W, F)
    win: float
    hop: float


def _periodicity(x: np.ndarray, fs: float, fmin: float = 1.3, fmax: float = 2.9) -> float:
    """Max normalised autocorrelation of x at lags corresponding to fmin..fmax
    (step frequency band): ~1 for walking, ~0 for climbing / rest."""
    x = x - x.mean()
    den = np.dot(x, x)
    if den < 1e-9:
        return 0.0
    lo, hi = int(fs / fmax), int(np.ceil(fs / fmin))
    hi = min(hi, len(x) - 1)
    if hi <= lo:
        return 0.0
    r = [np.dot(x[:-L], x[L:]) / den for L in range(lo, hi + 1)]
    return float(max(0.0, max(r)))


def _elevation_sign(acc, fs, ghat, axis: int) -> float | None:
    """Choose the forearm-axis sign from walking bouts (arm hangs down while
    walking). Returns None (fallback: session median) if too little walking."""
    W = int(4 * fs)
    mag = np.linalg.norm(acc, axis=1)
    e = np.arcsin(np.clip(ghat[:, axis], -1, 1))
    votes = []
    for s in range(0, len(acc) - W, W):
        if _periodicity(mag[s : s + W], fs) > 0.6 and np.std(mag[s : s + W]) > 0.5:
            votes.append(np.median(e[s : s + W]))
    if len(votes) >= 5:
        return -1.0 if np.median(votes) > 0 else 1.0
    return None


def window_features(acc, fs, gyr=None, win: float = 4.0, hop: float = 1.0, forearm_axis: int = 0,
                    elev_sign: float | None = None, ghat=None) -> Windows:
    # Segmentation features deliberately use the accelerometer-only gravity
    # estimate (0.3 Hz low-pass) even when a gyroscope exists: it behaves the same
    # on every device and ignores sub-second fidgets. ``gyr`` is accepted for API
    # symmetry only.
    ghat = pp.gravity_lowpass(acc, fs) if ghat is None else ghat
    if elev_sign is None:
        elev_sign = _elevation_sign(acc, fs, ghat, forearm_axis)
    elev, _ = pp.elevation(ghat, forearm_axis, elev_sign)
    tr = pp.tilt_rate(ghat, fs)
    dm = pp.dyn_magnitude(acc)
    mag = np.linalg.norm(acc, axis=1)
    W, H = int(win * fs), int(hop * fs)
    starts = np.arange(0, max(1, len(acc) - W + 1), H)
    X = np.empty((len(starts), len(BASE_FEATURES)))
    for i, s in enumerate(starts):
        sl = slice(s, s + W)
        g = ghat[sl]
        X[i] = (
            elev[sl].mean(),
            (elev[sl] > np.deg2rad(UP_DEG)).mean(),
            np.log(tr[sl].mean() + 0.05),
            np.log(np.sqrt(np.mean(dm[sl] ** 2)) + 0.05),
            _periodicity(mag[sl], fs),
            1.0 - np.linalg.norm(g.mean(axis=0)),
        )
    k = max(1, int(round(CONTEXT_S / hop)))
    ctx = [median_filter(X[:, BASE_FEATURES.index(c)], size=k, mode="nearest") for c in CONTEXT]
    X = np.column_stack([X, *ctx])
    centers = (starts + W / 2) / fs
    return Windows(centers, X, win, hop)


# --------------------------------------------------------------------------- HMM
class GaussianHMM:
    """K-state HMM with diagonal Gaussian emissions (Rabiner 1989).

    The transition matrix is *fixed* from expected state durations (a sticky
    prior: p_stay = 1 - hop/mean_duration); only emissions are learned by EM
    (Baum-Welch) on the session itself, so the model adapts to each
    person/device without labelled data.
    """

    def __init__(self, mean_dur_s=(120.0, 20.0, 60.0), hop: float = 1.0, n_iter: int = 30,
                 var_floor: float = 1e-2, temper: float = 0.25):
        K = len(mean_dur_s)
        A = np.empty((K, K))
        for i, d in enumerate(mean_dur_s):
            ps = 1 - hop / d
            A[i] = (1 - ps) / (K - 1)
            A[i, i] = ps
        self.K, self.A = K, A
        self.pi = np.full(K, 1.0 / K)
        # temper < 1 compensates for overlapping windows (observations are not
        # conditionally independent), otherwise the HMM is over-confident.
        self.temper, self.n_iter, self.var_floor = temper, n_iter, var_floor

    def _logB(self, X, forbid=None):
        out = np.empty((len(X), self.K))
        for k in range(self.K):
            v = self.var[k]
            out[:, k] = -0.5 * self.temper * (np.sum((X - self.mu[k]) ** 2 / v, axis=1) + np.sum(np.log(2 * np.pi * v)))
        if forbid is not None:
            out[forbid, 1:] = -1e6  # state 0 (rest) only
        return out

    def _fb(self, logB):
        """Scaled forward-backward (Rabiner 1989, Sec. V-A)."""
        T, K = logB.shape
        mx = logB.max(axis=1, keepdims=True)
        B = np.exp(logB - mx)
        A, AT = self.A, self.A.T.copy()
        al = np.empty((T, K))
        c = np.empty(T)
        p = self.pi * B[0]
        for t in range(T):
            if t:
                p = (AT @ al[t - 1]) * B[t]
            c[t] = p.sum()
            al[t] = p / c[t]
        be = np.empty((T, K))
        be[-1] = 1.0
        for t in range(T - 2, -1, -1):
            be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
        g = al * be
        g /= g.sum(axis=1, keepdims=True)
        return g, float(np.sum(np.log(c)) + mx.sum())

    def _mstep(self, X, gamma, keep=None):
        if keep is not None:
            X, gamma = X[keep], gamma[keep]
        w = gamma.sum(axis=0) + 1e-9
        self.mu = (gamma.T @ X) / w[:, None]
        self.var = np.stack([(gamma[:, k : k + 1] * (X - self.mu[k]) ** 2).sum(0) / w[k] for k in range(self.K)])
        self.var = np.maximum(self.var, self.var_floor)

    def fit(self, X, init_resp: np.ndarray, forbid=None):
        """init_resp: (T, K) initial soft responsibilities."""
        gamma = init_resp / init_resp.sum(axis=1, keepdims=True)
        self.ll_ = []
        for _ in range(self.n_iter):
            self._mstep(X, gamma, None if forbid is None else ~forbid)
            gamma, ll = self._fb(self._logB(X, forbid))
            done = bool(self.ll_) and abs(ll - self.ll_[-1]) < 1e-5 * abs(ll)
            self.ll_.append(ll)
            if done:
                break
        self.posterior_ = gamma
        return self

    def viterbi(self, X, forbid=None):
        logB = self._logB(X, forbid)
        logA = np.log(self.A)
        T = len(X)
        d = np.empty((T, self.K))
        bp = np.zeros((T, self.K), dtype=int)
        d[0] = np.log(self.pi) + logB[0]
        for t in range(1, T):
            m = d[t - 1][:, None] + logA
            bp[t] = m.argmax(axis=0)
            d[t] = m.max(axis=0) + logB[t]
        s = np.empty(T, dtype=int)
        s[-1] = d[-1].argmax()
        for t in range(T - 2, -1, -1):
            s[t] = bp[t + 1, s[t + 1]]
        return s


def heuristic_score(X: np.ndarray) -> np.ndarray:
    """Unsupervised prior climb score in [0,1] used to initialise EM.

    Climbing = forearm up a lot + frequent re-orientation + little step
    periodicity. Each term is robust-z-scored within the session.
    """
    def rz(v):
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
        return (v - med) / mad

    f = dict(zip(FEATURES, X.T))
    z = (0.5 * rz(f["elev_up_frac_ctx"]) + 1.0 * rz(f["tilt_rate_ctx"]) + 1.0 * rz(f["log_dyn_rms_ctx"])
         - 1.0 * rz(f["periodicity_ctx"]))
    return 1.0 / (1.0 + np.exp(-np.clip(z - np.percentile(z, 70), -30, 30)))


def intervals_from_mask(mask: np.ndarray, centers: np.ndarray, hop: float) -> list[tuple[float, float]]:
    m = np.r_[False, mask.astype(bool), False]
    e = np.flatnonzero(np.diff(m.astype(np.int8)))
    return [(float(centers[a] - hop / 2), float(centers[b - 1] + hop / 2)) for a, b in zip(e[::2], e[1::2])]


def clean_intervals(iv, min_dur: float = 10.0, merge_gap: float = 10.0):
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if out and a - out[-1][1] <= merge_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out if b - a >= min_dur]


@dataclass
class Segmentation:
    climbs: list[tuple[float, float]]
    windows: Windows
    posterior: np.ndarray
    method: str


def segment(acc, fs, gyr=None, method: str = "hmm", min_dur: float = 10.0, merge_gap: float = 10.0,
            forearm_axis: int = 0, elev_sign: float | None = None, ghat=None) -> Segmentation:
    win = window_features(acc, fs, gyr, forearm_axis=forearm_axis, elev_sign=elev_sign, ghat=ghat)
    X = win.X
    score = heuristic_score(X)
    if method == "threshold":
        # hysteresis on the heuristic score (baseline)
        on, mask = False, np.zeros(len(score), bool)
        for i, s in enumerate(score):
            on = s > 0.6 if not on else s > 0.4
            mask[i] = on
        post = score
    elif method == "hmm":
        H = X[:, [FEATURES.index(c) for c in HMM_FEATURES]]
        Z = (H - H.mean(0)) / (H.std(0) + 1e-9)
        # walking gate: rhythmic gait is never climbing; those windows are
        # excluded from EM and forced to the rest state in decoding.
        walk = X[:, FEATURES.index("periodicity_ctx")] > WALK_PERIODICITY
        hmm = GaussianHMM(mean_dur_s=(150.0, 60.0), hop=win.hop)
        hmm.fit(Z, np.column_stack([1 - score, score]) + 1e-3, forbid=walk)
        k_climb = int(np.argmax(hmm.mu @ np.array([1.0, 1.0, 1.0])))
        mask = hmm.viterbi(Z, forbid=walk) == k_climb
        post = hmm.posterior_[:, k_climb]
    else:
        raise ValueError(method)
    iv = clean_intervals(intervals_from_mask(mask, win.centers, win.hop), min_dur, merge_gap)
    iv = [ab for ab in iv if _plausible_climb(ab, win)]
    return Segmentation(iv, win, post, method)


# Interval-level plausibility gate (chosen on synthetic *dev* subjects 0-3 only).
MIN_UP_FRAC = 0.2  # mean fraction of time with forearm > UP_DEG over the candidate
MIN_LOG_DYN = -0.6  # mean log RMS dynamic acceleration (log m/s^2) over the candidate


def _plausible_climb(ab, win: Windows) -> bool:
    m = (win.centers >= ab[0]) & (win.centers <= ab[1])
    if not m.any():
        return False
    X = win.X[m]
    return (X[:, FEATURES.index("elev_up_frac")].mean() >= MIN_UP_FRAC
            and X[:, FEATURES.index("log_dyn_rms")].mean() >= MIN_LOG_DYN)
