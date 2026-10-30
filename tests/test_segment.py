import numpy as np
import pytest

from climbtrack import evaluate, segment, synth


@pytest.mark.parametrize("seed,kind", [(5, "boulder"), (6, "rope")])
def test_hmm_recovers_synthetic_climbs(seed, kind):
    s = synth.generate_session(seed=seed, kind=kind, n_climbs=4)
    r = segment.segment(s.acc, s.fs)
    e = evaluate.event_scores(s.climbs, r.climbs)
    assert e["recall"] >= 0.75 and e["precision"] >= 0.6
    assert e["mean_iou"] >= 0.7


def test_no_climbs_in_pure_rest_and_walking():
    rng = np.random.default_rng(0)
    tr = synth._Track(50.0, rng)
    sub = synth.Subject.random(0)
    for _ in range(4):
        synth._rest(tr, sub, 120)
        synth._walk(tr, sub, 30)
    acc, gyr, act = synth._render(tr, sub, rng)
    r = segment.segment(acc, 50.0)
    assert len(r.climbs) <= 1


def test_hmm_forward_backward_matches_bruteforce():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 2))
    hmm = segment.GaussianHMM(mean_dur_s=(3.0, 2.0), hop=1.0, temper=1.0)
    hmm.mu = np.array([[0.0, 0.0], [1.0, 1.0]])
    hmm.var = np.ones((2, 2))
    g, ll = hmm._fb(hmm._logB(X))
    # brute force over all 2^6 paths
    import itertools

    logB = hmm._logB(X)
    tot, marg = -np.inf, np.full((6, 2), -np.inf)
    for path in itertools.product([0, 1], repeat=6):
        lp = np.log(hmm.pi[path[0]]) + logB[0, path[0]] + sum(
            np.log(hmm.A[path[t - 1], path[t]]) + logB[t, path[t]] for t in range(1, 6))
        tot = np.logaddexp(tot, lp)
        for t, k in enumerate(path):
            marg[t, k] = np.logaddexp(marg[t, k], lp)
    assert ll == pytest.approx(tot, rel=1e-9)
    assert np.allclose(g, np.exp(marg - tot))


def test_clean_intervals():
    assert segment.clean_intervals([(0, 20), (25, 40), (100, 105)], min_dur=10, merge_gap=10) == [(0, 40)]
