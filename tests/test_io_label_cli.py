import io as sio

import numpy as np
import pytest

from climbtrack import cli, io, label, synth


def test_csv_roundtrip_units_and_ms(tmp_path):
    t = np.arange(0, 20, 0.02)
    acc_g = np.column_stack([np.zeros_like(t), np.zeros_like(t), np.ones_like(t)])
    p = tmp_path / "x.csv"
    np.savetxt(p, np.column_stack([t * 1000, acc_g]), delimiter=",", header="timestamp,ax,ay,az", comments="")
    r = io.load_csv(str(p))
    assert r.fs == 50 and r.gyr is None
    assert np.allclose(np.linalg.norm(r.acc, axis=1), io.G)


def test_labeller_and_tap_alignment(tmp_path):
    clock = iter([100.0, 110.0, 150.0, 160.0]).__next__
    out = tmp_path / "live.csv"
    climbs = label.run_labeller(str(out), clock=clock, inp=sio.StringIO("s\n\n\nq\n"), out=sio.StringIO())
    assert climbs == [(10.0, 50.0)]
    fs = 50.0
    acc = np.tile([0, 0, io.G], (int(300 * fs), 1)).astype(float)
    for k in range(3):  # 3 taps starting at 42 s
        acc[int((42 + 0.4 * k) * fs), 2] += 4 * io.G
    off = label.align_labels(str(out), acc, fs, str(tmp_path / "aligned.csv"))
    assert off == pytest.approx(42.0, abs=0.05)
    assert io.load_labels(str(tmp_path / "aligned.csv"))[0] == pytest.approx((52.0, 92.0), abs=0.05)


def test_cli_end_to_end(tmp_path, capsys):
    s = synth.generate_session(seed=3, n_climbs=3)
    csvp, lab = tmp_path / "s.csv", tmp_path / "l.csv"
    synth.to_csv(s, str(csvp), str(lab))
    cli.main(["analyze", str(csvp), "--labels", str(lab), "-o", str(tmp_path / "out"), "--synthetic"])
    md = (tmp_path / "out" / "s_report.md").read_text()
    assert "SYNTHETIC DATA" in md and (tmp_path / "out" / "s_timeline.png").exists()
