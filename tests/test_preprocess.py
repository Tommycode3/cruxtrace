import numpy as np

from climbtrack import preprocess as pp, synth


def test_gravity_estimates_agree_with_truth_at_rest():
    s = synth.generate_session(seed=1, n_climbs=1, kind="boulder")
    rest = s.activity == 0
    for g in (pp.gravity_lowpass(s.acc, s.fs), pp.gravity_complementary(s.acc, s.gyr, s.fs)):
        assert np.allclose(np.linalg.norm(g, axis=1), 1, atol=1e-6)
        ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", g, s.acc / np.linalg.norm(s.acc, axis=1, keepdims=True)), -1, 1)))
        assert np.median(ang[rest]) < 3.0


def test_complementary_tracks_rotation_without_accel_help():
    # pure gyro rotation about z by 90 deg: gravity along x should rotate in the body frame
    fs, T = 100.0, 1.0
    n = int(fs * T)
    gyr = np.tile([0, 0, np.pi / 2 / T], (n, 1))
    acc = np.tile([9.81, 0, 0], (n, 1))
    g = pp.gravity_complementary(acc, gyr, fs, tau=1e9)  # ~pure integration
    assert np.allclose(g[-1], [0, -1, 0], atol=0.05)


def test_dyn_magnitude_is_rotation_invariant():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 3, (500, 3)) + [0, 0, 9.81]
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert np.allclose(pp.dyn_magnitude(a), pp.dyn_magnitude(a @ q.T))
