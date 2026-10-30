"""Session report: timeline figure (PNG) + Markdown + JSON."""

from __future__ import annotations

import json
import math
import os

import numpy as np

from . import preprocess as pp


def timeline_figure(path, t, acc, seg, climbs_gt=None, title="", banner=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1, 0.6]})
    tm = t / 60
    step = max(1, len(t) // 20000)  # keep the figure light
    dyn = pp.dyn_magnitude(acc)
    ax[0].plot(tm[::step], dyn[::step], lw=0.4, color="#555")
    ax[0].set_ylabel("| ||a|| - g |  (m/s²)")
    ax[0].set_ylim(0, np.percentile(dyn, 99.5) * 1.2 + 1e-6)
    for a, b in seg.climbs:
        ax[0].axvspan(a / 60, b / 60, color="#e4572e", alpha=0.18, lw=0)
    w = seg.windows
    ax[1].plot(w.centers / 60, seg.posterior, color="#e4572e", lw=1)
    ax[1].set_ylabel("P(climb)")
    ax[1].set_ylim(-0.05, 1.05)
    rows = [("detected", seg.climbs, "#e4572e")]
    if climbs_gt is not None:
        rows.append(("ground truth", climbs_gt, "#29335c"))
    for k, (name, iv, col) in enumerate(rows):
        for a, b in iv:
            ax[2].barh(k, (b - a) / 60, left=a / 60, color=col, height=0.6)
    ax[2].set_yticks(range(len(rows)), [r[0] for r in rows])
    ax[2].set_xlabel("time (min)")
    ax[2].set_ylim(-0.6, len(rows) - 0.4)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    if banner:
        fig.text(0.5, 0.995, banner, ha="center", va="top", fontsize=9, color="#b00", weight="bold")
    ax[0].set_title(title, fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97 if banner else 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fmt(v):
    if isinstance(v, float):
        return "–" if math.isnan(v) else f"{v:g}"
    return str(v)


def write_report(outdir, name, rec, seg, climb_metrics, summary, scores=None, climbs_gt=None, synthetic=False):
    os.makedirs(outdir, exist_ok=True)
    banner = "SYNTHETIC DATA — generator output, not a real session" if synthetic else None
    png = os.path.join(outdir, f"{name}_timeline.png")
    timeline_figure(png, rec.t, rec.acc, seg, climbs_gt, title=f"{name} — {seg.method}", banner=banner)
    L = [f"# Session report: {name}", ""]
    if synthetic:
        L += ["> **SYNTHETIC DATA.** Produced by `climbtrack.synth`; numbers describe the generator, not a climber.", ""]
    L += [f"Source: `{os.path.basename(rec.source) or name}` · fs = {rec.fs:g} Hz · gyroscope: {'yes' if rec.gyr is not None else 'no'}", "",
          f"![timeline]({os.path.basename(png)})", "", "## Session", "", "| metric | value |", "|---|---|"]
    L += [f"| {k} | {_fmt(v)} |" for k, v in summary.as_dict().items()]
    if scores:
        L += ["", "## Agreement with labels", "", "| metric | value |", "|---|---|"]
        L += [f"| {k} | {_fmt(round(v, 3) if isinstance(v, float) else v)} |" for k, v in scores.items()]
    L += ["", "## Climbs", ""]
    if climb_metrics:
        keys = list(climb_metrics[0].as_dict())
        L += ["| # | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        for i, m in enumerate(climb_metrics, 1):
            L.append(f"| {i} | " + " | ".join(_fmt(v) for v in m.as_dict().values()) + " |")
    else:
        L.append("No climbs detected.")
    L += ["", "Metric definitions: see README § Movement metrics. Smoothness values are only comparable "
          "within the same device, sampling rate and wrist."]
    md = os.path.join(outdir, f"{name}_report.md")
    with open(md, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(outdir, f"{name}_report.json"), "w") as f:
        json.dump({"synthetic": synthetic, "summary": summary.as_dict(), "climbs": [m.as_dict() for m in climb_metrics],
                   "scores": scores}, f, indent=2, default=lambda o: None)
    return md, png
