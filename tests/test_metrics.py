import numpy as np
import pytest

from climbtrack import metrics


def minjerk_speed(T, D, fs):
    t = np.arange(0, T + 1e-9, 1 / fs)
    tau = t / T
    return D / T * 30 * tau**2 * (1 - tau) ** 2


@pytest.mark.parametrize("T,D", [(0.5, 0.2), (1.0, 0.5), (2.0, 1.3)])
def test_ldlj_v_minjerk_is_analytic_and_invariant(T, D):
    # integral of jerk^2 for min-jerk = 720 D^2 / T^5, v_peak = 1.875 D/T  ->  -ln(720/1.875^2)
    v = minjerk_speed(T, D, 2000.0)
    assert metrics.ldlj_v(v, 2000.0) == pytest.approx(-np.log(720 / 1.875**2), abs=0.02)


def test_smoothness_decreases_with_submovements():
    fs = 200.0
    one = minjerk_speed(1.0, 0.5, fs)
    two = np.r_[minjerk_speed(0.5, 0.25, fs), minjerk_speed(0.5, 0.25, fs)[1:]]
    assert metrics.sparc(two, fs) < metrics.sparc(one, fs)
    assert metrics.ldlj_v(two, fs) < metrics.ldlj_v(one, fs)
    acc1, acc2 = np.gradient(one, 1 / fs), np.gradient(two, 1 / fs)
    assert metrics.ldlj_a(acc2, fs) < metrics.ldlj_a(acc1, fs)


def test_sparc_minjerk_reference_value():
    # SPARC of a single minimum-jerk speed profile is ~ -1.4 (scale/duration invariant)
    for T in (0.5, 1.0, 3.0):
        assert metrics.sparc(minjerk_speed(T, 1.0, 500.0), 500.0) == pytest.approx(-1.40, abs=0.05)


def test_session_summary():
    s = metrics.session_summary([(10, 40), (100, 160)], 600)
    assert s.n_climbs == 2 and s.time_on_wall_s == 90 and s.median_rest_s == 60
    assert s.work_rest_ratio == pytest.approx(1.5)
