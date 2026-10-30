"""Command-line interface: ``climbtrack {analyze,synth,label,align}``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__


def _analyze(a):
    from . import evaluate, io, metrics, preprocess, report, segment

    rec = io.load(a.input, a.fs)
    ghat = preprocess.gravity(rec.acc, rec.fs, rec.gyr)
    seg = segment.segment(rec.acc, rec.fs, rec.gyr, method=a.method, min_dur=a.min_climb, merge_gap=a.merge_gap,
                          forearm_axis="xyz".index(a.forearm_axis), elev_sign=None if a.forearm_sign == "auto" else float(a.forearm_sign))
    cms = [metrics.climb_metrics(rec.acc, rec.fs, s, e, rec.gyr, ghat) for s, e in seg.climbs]
    summ = metrics.session_summary(seg.climbs, rec.t[-1])
    gt, scores = None, None
    if a.labels:
        gt = io.load_labels(a.labels)
        scores = {**evaluate.event_scores(gt, seg.climbs), **evaluate.frame_scores(gt, seg.climbs, rec.t[-1])}
    name = a.name or os.path.splitext(os.path.basename(a.input))[0]
    md, png = report.write_report(a.out, name, rec, seg, cms, summ, scores, gt, synthetic=a.synthetic)
    print(json.dumps({"summary": summ.as_dict(), "scores": scores}, indent=2, default=str))
    print(f"report: {md}\nfigure: {png}")


def _synth(a):
    from . import synth

    s = synth.generate_session(a.seed, kind=a.kind, fs=a.fs)
    synth.to_csv(s, a.out, a.labels)
    print(f"SYNTHETIC session: {len(s.climbs)} climbs, {s.t[-1] / 60:.1f} min -> {a.out}" + (f", labels -> {a.labels}" if a.labels else ""))


def _label(a):
    from . import label

    label.run_labeller(a.out)


def _align(a):
    from . import io, label

    rec = io.load(a.imu, a.fs)
    off = label.align_labels(a.labels, rec.acc, rec.fs, a.out, a.offset)
    print(f"offset {off:.2f} s -> {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="climbtrack", description="Climbing session tracking from a wrist IMU.")
    p.add_argument("--version", action="version", version=__version__)
    sp = p.add_subparsers(dest="cmd", required=True)

    q = sp.add_parser("analyze", help="segment a session and write a report")
    q.add_argument("input", help="IMU CSV (t,ax,ay,az[,gx,gy,gz]) or .fit with accelerometer_data")
    q.add_argument("-o", "--out", default="out")
    q.add_argument("--labels", help="ground-truth CSV start_s,end_s[,label] to score against")
    q.add_argument("--fs", type=float, help="resample to this rate (default: native)")
    q.add_argument("--method", choices=["hmm", "threshold"], default="hmm")
    q.add_argument("--min-climb", type=float, default=10.0, help="min climb duration, s")
    q.add_argument("--merge-gap", type=float, default=10.0, help="merge climbs separated by less, s")
    q.add_argument("--forearm-axis", choices=list("xyz"), default="x", help="sensor axis along the forearm")
    q.add_argument("--forearm-sign", choices=["auto", "1", "-1"], default="auto",
                   help="-1 if the forearm axis points to the elbow; auto = infer from walking bouts")
    q.add_argument("--name")
    q.add_argument("--synthetic", action="store_true", help="stamp report/figure as synthetic")
    q.set_defaults(func=_analyze)

    q = sp.add_parser("synth", help="write a SYNTHETIC session CSV (+labels) for testing")
    q.add_argument("-o", "--out", required=True)
    q.add_argument("--labels")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--kind", choices=["boulder", "rope"], default="boulder")
    q.add_argument("--fs", type=float, default=50.0)
    q.set_defaults(func=_synth)

    q = sp.add_parser("label", help="live stopwatch labeller (Enter = climb start/stop)")
    q.add_argument("out")
    q.set_defaults(func=_label)

    q = sp.add_parser("align", help="shift live labels onto the IMU clock via 3-tap sync")
    q.add_argument("imu")
    q.add_argument("labels")
    q.add_argument("-o", "--out", required=True)
    q.add_argument("--offset", type=float, help="manual offset (s) instead of tap detection")
    q.add_argument("--fs", type=float)
    q.set_defaults(func=_align)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
