"""Physically-grounded synthetic wrist-IMU climbing sessions.

THIS IS A TEST FIXTURE, NOT DATA. Every number produced from it is a property of
the assumptions below, not of real climbers. It exists so the pipeline has a
verifier (known ground truth) until real labelled sessions are recorded
(see docs/DATA_COLLECTION.md).

Model (all assumptions, documented in README "Synthetic generator"):

* Sensor frame: x along the forearm (towards the hand), y lateral, z out of the
  watch face. Orientation R(t) = Rz(yaw) Ry(pitch) Rx(roll); forearm elevation
  above horizontal = -pitch.
* Accelerometer = specific force  f_b = R^T (a_w + g e_z)  (+ bias, white noise).
* Gyroscope = body angular rate from ZYX Euler-angle rates (+ bias, noise).
* Rest: arm hanging or hands on lap, slow drift, occasional fidget/gesture.
* Walking: pendulum arm swing at stride frequency (~0.9 Hz) plus vertical
  bounce / heel-strike at step frequency (~1.8 Hz).
* Belaying (rope sessions): arm raised to chest height, rope-pull cycles of
  ~0.6 Hz. Labelled REST: it is a deliberate confounder.
* Climbing: alternating *move* (minimum-jerk re-orientation of the forearm to a
  new hold, 0.4-1.2 s, with a minimum-jerk hand displacement of 0.2-0.7 m) and
  *hold* (0.6-4 s static with 8-12 Hz isometric tremor). Occasional shake-outs
  (arm dropped, 3-5 Hz shaking) and chalk dips inside the climb. Forearm
  elevation on the wall is mostly above horizontal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.80665


@dataclass
class Subject:
    """Per-'subject' style parameters (inter-individual variability)."""

    seed: int
    move_time: tuple[float, float] = (0.4, 1.2)
    hold_time: tuple[float, float] = (0.6, 4.0)
    reach: tuple[float, float] = (0.2, 0.7)
    elev_wall: tuple[float, float] = (-10.0, 85.0)  # deg, forearm on the wall
    tremor: float = 0.12  # m/s^2
    shakeout_p: float = 0.08  # per move
    fidget_rate: float = 1 / 25  # per s at rest
    noise: float = 0.04  # accel white noise, m/s^2

    @classmethod
    def random(cls, seed: int) -> "Subject":
        r = np.random.default_rng(10_000 + seed)
        mt = r.uniform(0.35, 0.6)
        return cls(
            seed=seed,
            move_time=(mt, mt + r.uniform(0.5, 0.9)),
            hold_time=(r.uniform(0.4, 0.9), r.uniform(2.5, 5.0)),
            reach=(0.2, r.uniform(0.5, 0.8)),
            elev_wall=(r.uniform(-25, 0), r.uniform(70, 88)),
            tremor=r.uniform(0.06, 0.2),
            shakeout_p=r.uniform(0.04, 0.12),
            fidget_rate=r.uniform(1 / 40, 1 / 12),
            noise=r.uniform(0.02, 0.06),
        )


@dataclass
class SynthSession:
    fs: float
    t: np.ndarray
    acc: np.ndarray  # (N,3) m/s^2, specific force, sensor frame
    gyr: np.ndarray  # (N,3) rad/s
    climbs: list[tuple[float, float]]  # ground-truth [start, end) in s
    activity: np.ndarray  # per-sample int code, see ACTIVITY
    meta: dict = field(default_factory=dict)


ACTIVITY = {0: "rest", 1: "walk", 2: "belay", 3: "climb"}


def _minjerk(n: int) -> np.ndarray:
    tau = np.linspace(0.0, 1.0, n, endpoint=False) if n > 1 else np.zeros(1)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _minjerk_acc(n: int, T: float) -> np.ndarray:
    tau = np.linspace(0.0, 1.0, n, endpoint=False)
    return (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2


class _Track:
    """Accumulates Euler angles (rad) and world linear acceleration sample-wise."""

    def __init__(self, fs: float, rng: np.random.Generator):
        self.fs, self.rng = fs, rng
        self.ang: list[np.ndarray] = []
        self.lin: list[np.ndarray] = []
        self.act: list[np.ndarray] = []
        self.cur = np.array([0.0, np.deg2rad(80.0), 0.0])  # yaw, pitch, roll

    def n(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.fs)))

    def hold(self, seconds: float, code: int, lin_std: float = 0.0, osc=None):
        n = self.n(seconds)
        a = np.tile(self.cur, (n, 1))
        lin = np.zeros((n, 3))
        if lin_std > 0:
            lin += self.rng.normal(0, lin_std, (n, 3))
        if osc is not None:
            osc(a, lin, n)
        self._push(a, lin, code)

    def move(self, target: np.ndarray, seconds: float, code: int, disp: np.ndarray | None = None):
        n = self.n(seconds)
        s = _minjerk(n)[:, None]
        a = self.cur + (target - self.cur) * s
        lin = np.zeros((n, 3))
        if disp is not None:
            lin += disp[None, :] * _minjerk_acc(n, seconds)[:, None]
        self.cur = target.copy()
        self._push(a, lin, code)

    def _push(self, a, lin, code):
        self.ang.append(a)
        self.lin.append(lin)
        self.act.append(np.full(len(a), code, dtype=np.int8))


def _orient(elev_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    return np.deg2rad([yaw_deg, -np.clip(elev_deg, -85, 85), roll_deg])


def _rest(tr: _Track, sub: Subject, seconds: float):
    rng = tr.rng
    posture = _orient(rng.choice([-80.0, -70.0, -5.0, 10.0]), rng.uniform(-180, 180), rng.uniform(-60, 60))
    tr.move(posture, rng.uniform(0.8, 1.5), 0)
    left = seconds
    while left > 0:
        chunk = min(left, rng.exponential(1 / sub.fidget_rate))
        tr.hold(chunk, 0, lin_std=0.03)
        left -= chunk
        if left > 3 and rng.random() < 0.8:  # fidget / gesture / drink / phone
            el = rng.uniform(-40, 60)
            g = _orient(el, np.rad2deg(tr.cur[0]) + rng.uniform(-60, 60), rng.uniform(-80, 80))
            back = tr.cur.copy()
            t1 = rng.uniform(0.4, 1.0)
            tr.move(g, t1, 0, disp=rng.normal(0, 0.15, 3))
            tr.hold(rng.uniform(0.5, 4.0), 0, lin_std=0.08)
            tr.move(back, t1, 0, disp=rng.normal(0, 0.15, 3))
            left -= 2 * t1 + 2


def _walk(tr: _Track, sub: Subject, seconds: float):
    rng = tr.rng
    f_step = rng.uniform(1.6, 2.1)
    tr.move(_orient(-78, rng.uniform(-180, 180), rng.uniform(-20, 20)), 0.6, 1)
    amp = np.deg2rad(rng.uniform(12, 30))
    ph0 = rng.uniform(0, 2 * np.pi)
    fs = tr.fs

    def osc(a, lin, n):
        tt = np.arange(n) / fs
        a[:, 1] += amp * np.sin(np.pi * f_step * tt + ph0)  # stride = step/2
        lin[:, 2] += 2.0 * np.sin(2 * np.pi * f_step * tt) + 1.2 * np.maximum(0, np.sin(2 * np.pi * f_step * tt)) ** 8 * 4
        lin[:, 0] += 0.8 * np.sin(2 * np.pi * f_step * tt + 0.5)

    tr.hold(seconds, 1, lin_std=0.15, osc=osc)


def _belay(tr: _Track, sub: Subject, seconds: float):
    rng = tr.rng
    tr.move(_orient(rng.uniform(5, 35), rng.uniform(-180, 180), rng.uniform(-30, 30)), 0.8, 2)
    left = seconds
    fs = tr.fs
    while left > 0:
        chunk = min(left, rng.uniform(4, 15))
        if rng.random() < 0.6:  # taking in rope
            f = rng.uniform(0.4, 0.8)
            amp = np.deg2rad(rng.uniform(20, 45))

            def osc(a, lin, n, f=f, amp=amp):
                tt = np.arange(n) / fs
                a[:, 1] += amp * np.sin(2 * np.pi * f * tt)
                lin[:, 0] += 1.5 * np.sin(2 * np.pi * f * tt)

            tr.hold(chunk, 2, lin_std=0.1, osc=osc)
        else:
            tr.hold(chunk, 2, lin_std=0.05)
        left -= chunk


def _climb(tr: _Track, sub: Subject, seconds: float):
    rng = tr.rng
    el_lo, el_hi = sub.elev_wall
    yaw0 = rng.uniform(-180, 180)
    fs = tr.fs
    left = seconds
    while left > 0:
        mt = rng.uniform(*sub.move_time)
        target = _orient(rng.uniform(el_lo, el_hi), yaw0 + rng.uniform(-40, 40), rng.uniform(-70, 70))
        d = rng.normal(0, 1, 3)
        d[2] = abs(d[2]) + 0.5  # moves mostly go up
        d *= rng.uniform(*sub.reach) / np.linalg.norm(d)
        tr.move(target, mt, 3, disp=d)
        ht = rng.uniform(*sub.hold_time)
        ft = rng.uniform(8, 12)
        tremor = sub.tremor

        def osc(a, lin, n, ft=ft, tremor=tremor):
            tt = np.arange(n) / fs
            lin += tremor * np.sin(2 * np.pi * ft * tt)[:, None] * np.array([0.6, 1.0, 0.4])

        tr.hold(ht, 3, lin_std=0.04, osc=osc)
        left -= mt + ht
        if left > 6 and rng.random() < sub.shakeout_p:  # shake-out / chalk on the wall
            back = tr.cur.copy()
            tr.move(_orient(rng.uniform(-80, -50), yaw0, rng.uniform(-40, 40)), 0.6, 3, disp=np.array([0, 0, -0.4]))
            fsh = rng.uniform(3, 5)
            dur = rng.uniform(2, 7)

            def shake(a, lin, n, fsh=fsh):
                tt = np.arange(n) / fs
                a[:, 2] += np.deg2rad(25) * np.sin(2 * np.pi * fsh * tt)
                lin[:, 1] += 6.0 * np.sin(2 * np.pi * fsh * tt)

            tr.hold(dur, 3, lin_std=0.1, osc=shake)
            tr.move(back, 0.6, 3, disp=np.array([0, 0, 0.4]))
            left -= dur + 1.2


def _render(tr: _Track, sub: Subject, rng: np.random.Generator):
    ang = np.concatenate(tr.ang)
    lin = np.concatenate(tr.lin)
    act = np.concatenate(tr.act)
    fs = tr.fs
    yaw, pitch, roll = ang.T
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    # R = Rz(yaw) Ry(pitch) Rx(roll), stacked (N,3,3)
    R = np.empty((len(yaw), 3, 3))
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    fw = lin + np.array([0.0, 0.0, G])
    acc = np.einsum("nji,nj->ni", R, fw)  # R^T f_w
    dy, dp, dr = (np.gradient(x, 1 / fs) for x in (yaw, pitch, roll))
    gyr = np.stack([dr - dy * sp, dp * cr + dy * cp * sr, -dp * sr + dy * cp * cr], axis=1)
    acc += rng.normal(0, sub.noise, acc.shape) + rng.normal(0, 0.05, 3)
    gyr += rng.normal(0, 0.01, gyr.shape) + rng.normal(0, 0.005, 3)
    return acc, gyr, act


def generate_session(
    seed: int = 0,
    n_climbs: int | None = None,
    kind: str = "boulder",
    fs: float = 50.0,
    subject: Subject | None = None,
) -> SynthSession:
    """Generate one session. ``kind`` is 'boulder' (short problems, walking) or
    'rope' (longer routes, belaying between them)."""
    rng = np.random.default_rng(seed)
    sub = subject or Subject.random(seed)
    if n_climbs is None:
        n_climbs = int(rng.integers(6, 13) if kind == "boulder" else rng.integers(3, 7))
    tr = _Track(fs, rng)
    _rest(tr, sub, rng.uniform(60, 180))
    for i in range(n_climbs):
        if rng.random() < 0.7:
            _walk(tr, sub, rng.uniform(5, 30))
        _rest(tr, sub, rng.uniform(8, 40))  # inspecting the problem
        dur = rng.uniform(15, 100) if kind == "boulder" else rng.uniform(90, 300)
        _climb(tr, sub, dur)
        _rest(tr, sub, rng.uniform(5, 20))  # jump/lower off, catch breath
        if kind == "rope" and rng.random() < 0.8:
            _belay(tr, sub, rng.uniform(90, 300))
        else:
            _rest(tr, sub, rng.uniform(60, 300))
    acc, gyr, act = _render(tr, sub, rng)
    t = np.arange(len(acc)) / fs
    is_c = act == 3
    edges = np.flatnonzero(np.diff(np.r_[0, is_c.astype(np.int8), 0]))
    climbs = [(edges[k] / fs, edges[k + 1] / fs) for k in range(0, len(edges), 2)]
    return SynthSession(fs, t, acc, gyr, climbs, act, {"seed": seed, "kind": kind, "subject": sub.seed})


def to_csv(s: SynthSession, path: str, labels_path: str | None = None):
    data = np.column_stack([s.t, s.acc, s.gyr])
    np.savetxt(path, data, delimiter=",", fmt="%.5f", header="t,ax,ay,az,gx,gy,gz", comments="")
    if labels_path:
        with open(labels_path, "w") as f:
            f.write("start_s,end_s,label\n")
            for a, b in s.climbs:
                f.write(f"{a:.3f},{b:.3f},climb\n")
